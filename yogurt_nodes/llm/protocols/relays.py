"""聚合 / 中转平台的生图协议：OpenRouter Images API、xAI Images、GRSAI、FreedomGPT。"""

from __future__ import annotations

import asyncio
from math import gcd
from typing import Any

from ...utils.grsai_client import GRSAIClient
from ..framework import GenerationRequest, GenerationResult, ProviderConfig, Route, register_protocol
from .http import connect, data_url, download_image, ensure_ok, is_set, parse_extra
from .http import request as http_request
from .openai import (
    build_messages,
    image_fields,
    image_prompt,
    openrouter_routing,
    parse_image_data,
    request_json,
)


@register_protocol("openrouter_images")
async def run_openrouter_images(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    params = request.params
    payload: dict[str, Any] = {"model": route.model, "prompt": image_prompt(request), **image_fields(request)}
    payload.pop("moderation", None)
    for key in ("resolution", "aspect_ratio", "quality"):
        if is_set(params.get(key)):
            payload[key] = params[key]
    if request.seed and route.options.get("seed", True):
        payload["seed"] = request.seed
    if request.images:
        payload["input_references"] = [
            {"type": "image_url", "image_url": {"url": data_url(image)}} for image in request.images
        ]
    payload.update(openrouter_routing(provider))
    if is_set(params.get("moderation")):
        options = payload.setdefault("provider", {}).setdefault("options", {})
        options.setdefault("openai", {})["moderation"] = params["moderation"]
    payload.update(parse_extra(params.get("extra")))
    body = await request_json(conn, conn.url("images"), payload, f"{conn.name} image generation")
    return await parse_image_data(conn, body)


@register_protocol("xai_images")
async def run_xai_images(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    params = request.params
    payload: dict[str, Any] = {"model": route.model, "prompt": image_prompt(request), "response_format": "b64_json"}
    if params.get("n"):
        payload["n"] = int(params["n"])
    if is_set(params.get("aspect_ratio")):
        payload["aspect_ratio"] = params["aspect_ratio"]
    if is_set(params.get("resolution")):
        payload["resolution"] = str(params["resolution"]).lower()
    if is_set(params.get("quality")):
        payload["quality"] = params["quality"]
    if request.seed:
        payload["seed"] = request.seed
    endpoint = "images/generations"
    if request.images:
        endpoint = "images/edits"
        payload["images"] = [{"type": "image_url", "url": data_url(image, "JPEG")} for image in request.images]
    payload.update(parse_extra(params.get("extra")))
    body = await request_json(conn, conn.url(endpoint), payload, f"{conn.name} image")
    return await parse_image_data(conn, body)


def _ratio_from_size(size: str) -> str:
    try:
        width, height = (int(part) for part in size.lower().split("x"))
    except ValueError:
        return "auto"
    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


@register_protocol("grsai_draw")
async def run_grsai(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    params = request.params
    aspect_ratio = params.get("aspect_ratio") or "auto"
    if aspect_ratio == "auto" and is_set(params.get("size")) and params.get("size") != "Custom":
        aspect_ratio = _ratio_from_size(params["size"])
    image_size = params.get("resolution") or "auto"
    client = GRSAIClient(
        api_key=conn.api_key,
        base_url=conn.base_url,
        proxy_url=provider.get("proxy_url", ""),
        timeout=int(provider.get("timeout", 0) or 0),
    )
    images, text, _history = await asyncio.to_thread(
        client.generate_image,
        model_name=route.model,
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        images=request.images,
        aspect_ratio=aspect_ratio,
        image_size=image_size if image_size in ("1K", "2K", "4K") else "auto",
        retry_count=1,
        max_wait_seconds=int(provider.get("max_wait_seconds", 600) or 600),
        extra=parse_extra(params.get("extra")),
    )
    if not images:
        raise RuntimeError(f"GRSAI returned no image. {text[:300]}")
    return GenerationResult(images=list(images), text=text or "")


@register_protocol("freedomgpt_chat")
async def run_freedomgpt_chat(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    params = request.params
    messages = build_messages(request)
    payload: dict[str, Any] = {"model": route.model, "stream": False}
    if messages and messages[0]["role"] == "system":
        payload["customPrompt"] = True
        payload["prompt"] = messages.pop(0)["content"]
    payload["messages"] = messages
    for key in ("temperature", "top_p", "top_k"):
        if params.get(key) is not None:
            payload[key] = params[key]
    if params.get("max_output_tokens"):
        payload["max_tokens"] = int(params["max_output_tokens"])
    if request.seed:
        payload["seed"] = request.seed
    payload.update(parse_extra(params.get("extra")))
    body = await request_json(conn, conn.url("chat/completions"), payload, f"{conn.name} chat")
    choices = body.get("choices") or []
    text = (choices[0].get("message") or {}).get("content", "") if choices else ""
    if not text:
        raise RuntimeError(f"{conn.name} returned empty text: {str(body)[:500]}")
    return GenerationResult(text=text.strip())


@register_protocol("freedomgpt_image")
async def run_freedomgpt_image(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    params = request.params
    payload: dict[str, Any] = {
        "model": route.model,
        "prompt": image_prompt(request),
        "numberOfImages": int(params.get("n") or 1),
    }
    if request.seed:
        payload["seed"] = request.seed
    if request.images:
        payload["images"] = [data_url(image, "JPEG") for image in request.images]
    payload.update(parse_extra(params.get("extra")))
    response = await http_request(
        conn, "POST", conn.url("images/generations"),
        headers={**conn.bearer, "Content-Type": "application/json"}, json=payload,
    )
    body = ensure_ok(response, f"{conn.name} image generation")
    result = GenerationResult()
    for item in body.get("data") or []:
        if isinstance(item, dict) and item.get("url"):
            result.images.append(await download_image(conn, item["url"]))
    if not result.images:
        raise RuntimeError(f"{conn.name} returned no image: {str(body)[:500]}")
    return result
