"""放大协议：Topaz Image API 与 Magnific（异步任务 + 轮询）。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ...utils.topaz_client import TopazClient
from ..framework import (
    GenerationRequest,
    GenerationResult,
    ProviderConfig,
    Route,
    cancellable_sleep,
    register_protocol,
)
from .http import connect, data_url, download_image, ensure_ok, is_set, parse_extra
from .http import request as http_request

# 这些参数 build_topaz_form_data 不认识，按表单字段透传
_TOPAZ_PASSTHROUGH = (
    "seed", "color_preservation", "face_preservation", "enhancement_strength",
    "grain_model", "grain_strength", "grain_size", "grain_density",
)
_TOPAZ_GRAIN = ("grain_model", "grain_strength", "grain_size", "grain_density")


def _topaz_kwargs(params: dict[str, Any], request: GenerationRequest, route: Route) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "scale_factor": float(params.get("scale_factor", 2.0)),
        "output_width": int(params.get("output_width", 0) or 0),
        "output_height": int(params.get("output_height", 0) or 0),
        "crop_to_fill": bool(params.get("crop_to_fill", False)),
        "output_format": params.get("output_format", "png"),
    }
    for key in ("sharpen", "denoise", "fix_compression", "strength", "face_enhancement_strength",
                "face_enhancement_creativity", "creativity", "texture", "detail"):
        if params.get(key) is not None:
            kwargs[key] = params[key]
    face = params.get("face_enhancement")
    if face is not None:
        kwargs["face_enhancement"] = str(face).lower() if isinstance(face, bool) else face
    if is_set(params.get("subject_detection")):
        kwargs["subject_detection"] = params["subject_detection"]
    prompt = str(params.get("prompt", "") or "").strip()
    if "prompt" in params:
        kwargs["prompt"] = prompt
        if route.options.get("autoprompt"):
            kwargs["autoprompt"] = "false" if prompt else "true"

    extra = {}
    grain = params.get("grain")
    for key in _TOPAZ_PASSTHROUGH:
        if key in _TOPAZ_GRAIN and not grain:
            continue
        value = params.get(key)
        if value is not None:
            extra[key] = str(value).lower() if isinstance(value, bool) else value
    if grain:
        extra["grain"] = "true"
    if request.seed and "seed" in params:
        extra["seed"] = request.seed
    extra.update(parse_extra(params.get("extra")))
    kwargs["extra"] = extra
    return kwargs


@register_protocol("topaz")
async def run_topaz(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    client = TopazClient(
        api_key=conn.api_key,
        base_url=conn.base_url,
        proxy_url=provider.get("proxy_url", ""),
        timeout=int(provider.get("timeout", 0) or 0),
    )
    kwargs = _topaz_kwargs(request.params, request, route)
    task_timeout = int(provider.get("task_timeout", 900) or 900)
    result = GenerationResult()
    for image in request.images:
        result.images.append(
            await asyncio.to_thread(
                client.upscale_image,
                image=image,
                model_type=route.options.get("endpoint", "standard"),
                model=route.model,
                task_timeout=task_timeout,
                **kwargs,
            )
        )
    return result


def _magnific_payload(route: Route, params: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in route.options.get("fields", ()):
        value = params.get(key)
        if value is None:
            continue
        if key == "prompt" and not str(value).strip():
            continue
        if key == "scale_factor" and route.options.get("integer_scale"):
            value = int(str(value).rstrip("x"))
        payload[key] = value
    payload.update(parse_extra(params.get("extra")))
    return payload


@register_protocol("magnific")
async def run_magnific(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    endpoint = conn.url(route.model)
    headers = {"x-magnific-api-key": conn.api_key, "Content-Type": "application/json"}
    task_timeout = float(provider.get("task_timeout", 900) or 900)
    result = GenerationResult()
    for image in request.images:
        payload = {"image": data_url(image).split(",", 1)[1], **_magnific_payload(route, request.params)}
        body = ensure_ok(await http_request(conn, "POST", endpoint, headers=headers, json=payload), "Magnific")
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        task_id = str(data.get("task_id", "") or "")
        if not task_id:
            raise RuntimeError(f"Magnific did not return a task_id: {str(body)[:300]}")
        deadline = time.monotonic() + task_timeout
        while str(data.get("status", "")).upper() != "COMPLETED":
            if str(data.get("status", "")).upper() in ("FAILED", "ERROR", "CANCELLED"):
                raise RuntimeError(f"Magnific task {task_id} ended with status {data.get('status')}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Magnific task {task_id} timed out after {task_timeout:.0f}s")
            await cancellable_sleep(3)
            body = ensure_ok(await http_request(conn, "GET", f"{endpoint}/{task_id}", headers=headers), "Magnific")
            data = body.get("data") if isinstance(body.get("data"), dict) else body
        generated = data.get("generated") or []
        if not generated:
            raise RuntimeError(f"Magnific task {task_id} completed without an image")
        result.images.append((await download_image(conn, str(generated[0]))).convert("RGB"))
    return result
