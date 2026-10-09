"""火山方舟 / BytePlus（Seedream）与阿里云百炼 DashScope（Qwen Image / Wan）生图协议。"""

from __future__ import annotations

from typing import Any

from ..framework import GenerationRequest, GenerationResult, ProviderConfig, Route, register_protocol
from .http import connect, data_url, decode_image, download_image, is_set, parse_extra
from .openai import image_prompt, request_json, resolve_size


@register_protocol("ark_images")
async def run_ark_images(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    params = request.params
    size = resolve_size(params)
    payload: dict[str, Any] = {
        "model": route.model,
        "prompt": image_prompt(request),
        "response_format": "b64_json",
        "watermark": bool(params.get("watermark", False)),
    }
    if is_set(size):
        payload["size"] = size
    if request.seed:
        payload["seed"] = request.seed
    if request.images:
        urls = [data_url(image) for image in request.images]
        payload["image"] = urls[0] if len(urls) == 1 else urls
    max_images = int(params.get("max_images") or 1)
    if "max_images" in params:
        payload["sequential_image_generation"] = "auto" if max_images > 1 else "disabled"
        if max_images > 1:
            payload["sequential_image_generation_options"] = {"max_images": max_images}
    if params.get("prompt_optimization") == "fast":
        payload["optimize_prompt_options"] = {"mode": "fast"}
    elif "thinking" in params and not request.images:
        payload["optimize_prompt_options"] = {"thinking": "enabled" if params["thinking"] else "disabled"}
    payload.update(parse_extra(params.get("extra")))

    body = await request_json(conn, conn.url("images/generations"), payload, f"{conn.name} Seedream")
    result = GenerationResult()
    errors = []
    for item in body.get("data") or []:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            result.images.append(decode_image(item["b64_json"]))
        elif item.get("url"):
            result.images.append(await download_image(conn, item["url"]))
        elif item.get("error"):
            errors.append(str(item["error"]))
    if not result.images:
        raise RuntimeError(f"{conn.name} returned no image: {errors or str(body)[:500]}")
    if errors:
        result.text = f"{len(errors)} image(s) failed: {'; '.join(errors)[:500]}"
    return result


_DASHSCOPE_PARAMETERS = ("n", "negative_prompt", "prompt_extend", "watermark", "thinking_mode", "enable_sequential")


@register_protocol("dashscope_image")
async def run_dashscope_image(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    params = request.params
    content: list[dict[str, Any]] = [{"image": data_url(image)} for image in request.images]
    prompt = image_prompt(request)
    if not prompt:
        raise ValueError("Prompt is required")
    content.append({"text": prompt})
    parameters: dict[str, Any] = {}
    for key in _DASHSCOPE_PARAMETERS:
        value = params.get(key)
        if value is not None and value != "":
            parameters[key] = value
    size = resolve_size(params, separator="*")
    if is_set(size):
        parameters["size"] = size
    if request.seed:
        parameters["seed"] = request.seed
    payload: dict[str, Any] = {
        "model": route.model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": parameters,
    }
    for key, value in parse_extra(params.get("extra")).items():
        (payload if key in ("model", "input") else parameters)[key] = value

    body = await request_json(
        conn, conn.url("services/aigc/multimodal-generation/generation"), payload, f"{conn.name} image"
    )
    choices = (body.get("output") or {}).get("choices") or []
    urls = [
        item.get("image") or item.get("url")
        for choice in choices
        for item in ((choice.get("message") or {}).get("content") or [])
        if isinstance(item, dict) and (item.get("image") or item.get("url"))
    ]
    if not urls:
        raise RuntimeError(f"{conn.name} returned no image: {body.get('code')} {body.get('message') or str(body)[:500]}")
    return GenerationResult(images=[await download_image(conn, url) for url in urls])
