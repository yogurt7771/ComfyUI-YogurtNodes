"""stable-diffusion.cpp 原生 API：POST /sdcpp/v1/img_gen 提交任务 → 轮询 jobs/{id}；中断或超时时取消服务端任务。

请求体以服务端默认参数为底，只发送节点上设置过的字段；extra 中的字段按原生 schema 覆盖。
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..framework import GenerationRequest, GenerationResult, ProviderConfig, Route, cancellable_sleep, register_protocol
from .http import connect, data_url, decode_image, ensure_ok, parse_extra
from .http import request as http_request
from .openai import image_prompt, resolve_size

_POLL_INTERVAL = 1.0
_TERMINAL = ("completed", "failed", "cancelled")


def img_gen_body(request: GenerationRequest) -> dict[str, Any]:
    params = request.params
    body: dict[str, Any] = {"prompt": image_prompt(request), "seed": request.seed or -1}
    width, height = int(params.get("width") or 0), int(params.get("height") or 0)
    if not (width or height):
        match = re.fullmatch(r"(\d+)x(\d+)", resolve_size(params))
        if match:
            width, height = int(match.group(1)), int(match.group(2))
    if width:
        body["width"] = width
    if height:
        body["height"] = height
    if params.get("n"):
        body["batch_count"] = int(params["n"])
    if request.images:
        body["ref_images"] = [data_url(image) for image in request.images]
    body.update(parse_extra(params.get("extra")))
    # 原生接口省略尺寸时用服务端默认值（通常 512），img2img 应跟随 init 图
    if "width" not in body and "height" not in body and body.get("init_image"):
        body["width"], body["height"] = decode_image(body["init_image"]).size
    return body


@register_protocol("sdcpp_img_gen")
async def run_sdcpp_img_gen(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    headers = conn.bearer if conn.api_key else {}
    submitted = ensure_ok(
        await http_request(conn, "POST", conn.url("sdcpp/v1/img_gen"), headers=headers, json=img_gen_body(request)),
        f"{conn.name} img_gen",
    )
    job_id = str(submitted.get("id") or "")
    if not job_id:
        raise RuntimeError(f"{conn.name} did not return a job id: {str(submitted)[:300]}")
    job = await _wait(conn, headers, job_id, float(provider.get("task_timeout", 1800) or 1800))
    return _result(conn, job)


async def _wait(conn, headers: dict[str, str], job_id: str, task_timeout: float) -> dict[str, Any]:
    job_url = conn.url(f"sdcpp/v1/jobs/{job_id}")
    deadline = time.monotonic() + task_timeout
    job: dict[str, Any] = {}
    try:
        while job.get("status") not in _TERMINAL:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{conn.name} job {job_id} did not finish within {task_timeout:.0f}s")
            await cancellable_sleep(_POLL_INTERVAL)
            job = ensure_ok(await http_request(conn, "GET", job_url, headers=headers), f"{conn.name} job")
    except BaseException:
        # ComfyUI 中断（InterruptProcessingException 继承 BaseException）或超时：服务端任务不会自己停
        await _cancel(conn, headers, job_id)
        raise
    return job


async def _cancel(conn, headers: dict[str, str], job_id: str) -> None:
    try:
        await http_request(conn, "POST", conn.url(f"sdcpp/v1/jobs/{job_id}/cancel"), headers=headers)
    except Exception:  # noqa: BLE001 - 尽力取消，网络错误不掩盖原始异常
        pass


def _result(conn, job: dict[str, Any]) -> GenerationResult:
    status = job.get("status")
    if status != "completed":
        error = job.get("error") or {}
        detail = error.get("message") or error.get("code") or "no details"
        raise RuntimeError(f"{conn.name} job {job.get('id')} {status}: {detail}")
    items = (job.get("result") or {}).get("images") or []
    images = [
        decode_image(item["b64_json"])
        for item in sorted(items, key=lambda item: item.get("index", 0))
        if item.get("b64_json")
    ]
    if not images:
        raise RuntimeError(f"{conn.name} job {job.get('id')} completed without images")
    return GenerationResult(images=images)
