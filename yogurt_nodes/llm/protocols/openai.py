"""OpenAI 风格协议：chat/completions（OpenAI / 兼容 / xAI / OpenRouter）与 images API。"""

from __future__ import annotations

import re
from typing import Any

from ..framework import PROVIDERS, GenerationRequest, GenerationResult, ProviderConfig, Route, register_protocol
from .http import connect, data_url, decode_image, download_image, ensure_ok, is_set, parse_extra, png_bytes
from .http import request as http_request

# 各家 effort 取值不完全一致；OpenRouter 统一后不认识 max
_OPENROUTER_EFFORT = {"max": "xhigh", "off": "none"}


def build_messages(request: GenerationRequest, image_detail: str = "") -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    for role, text in request.history:
        if text:
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": text})
    content: list[dict[str, Any]] = []
    if request.prompt:
        content.append({"type": "text", "text": request.prompt})
    for image in request.images:
        image_url: dict[str, Any] = {"url": data_url(image, "JPEG")}
        if image_detail and image_detail != "auto":
            image_url["detail"] = image_detail
        content.append({"type": "image_url", "image_url": image_url})
    if not content:
        raise ValueError("Prompt or images are required")
    messages.append({"role": "user", "content": content})
    return messages


def _reasoning(params: dict[str, Any], provider: ProviderConfig) -> dict[str, Any]:
    """thinking_level / thinking_budget / reasoning_effort → 请求字段。"""
    openrouter = provider.route_key == "openrouter"
    effort = params.get("reasoning_effort")
    if not is_set(effort) and is_set(params.get("thinking_level")):
        effort = str(params["thinking_level"]).lower()
    budget = params.get("thinking_budget")

    if openrouter:
        if budget is not None:
            if int(budget) == 0:
                return {"reasoning": {"effort": "none"}}
            return {"reasoning": {"max_tokens": int(budget)} if int(budget) > 0 else {"enabled": True}}
        if is_set(effort):
            return {"reasoning": {"effort": _OPENROUTER_EFFORT.get(effort, effort)}}
        return {}
    if is_set(effort) and effort != "off":
        return {"reasoning_effort": effort}
    return {}


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


@register_protocol("openai_chat")
async def run_openai_chat(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    params = request.params
    payload: dict[str, Any] = {
        "model": route.model,
        "messages": build_messages(request, params.get("image_detail", "")),
    }
    tokens_field = route.options.get("max_tokens_field") or provider_field(provider, "max_tokens_field", "max_tokens")
    if params.get("max_output_tokens"):
        payload[tokens_field] = int(params["max_output_tokens"])
    for key in ("temperature", "top_p", "top_k", "frequency_penalty", "presence_penalty"):
        if params.get(key) is not None:
            payload[key] = params[key]
    if is_set(params.get("verbosity")):
        payload["verbosity"] = params["verbosity"]
    reasoning = _reasoning(params, provider)
    payload.update(reasoning)
    reasoning_off = reasoning.get("reasoning_effort") == "none" or reasoning.get("reasoning", {}).get("effort") == "none"
    if reasoning and not reasoning_off and route.options.get("drop_sampling_with_reasoning"):
        payload.pop("temperature", None)
        payload.pop("top_p", None)
    if request.seed:
        payload["seed"] = request.seed
    payload.update(openrouter_routing(provider))
    payload.update(parse_extra(params.get("extra")))

    response = await request_json(conn, conn.url("chat/completions"), payload, f"{conn.name} chat")
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"{conn.name} returned no choices: {str(response)[:500]}")
    message = choices[0].get("message") or {}
    text = _message_text(message)
    thought = message.get("reasoning") or message.get("reasoning_content") or ""
    if not text:
        reason = choices[0].get("finish_reason")
        raise RuntimeError(f"{conn.name} returned empty text (finish_reason={reason})")
    return GenerationResult(text=text, thought=thought if isinstance(thought, str) else str(thought))


def provider_field(provider: ProviderConfig, key: str, default: Any) -> Any:
    return PROVIDERS[provider.kind].connection.get(key, default)


def openrouter_routing(provider: ProviderConfig) -> dict[str, Any]:
    if provider.route_key != "openrouter":
        return {}
    order = [item.strip() for item in str(provider.get("provider_order", "")).split(",") if item.strip()]
    if not order:
        return {}
    return {"provider": {"order": order, "allow_fallbacks": bool(provider.get("allow_fallbacks", True))}}


async def request_json(conn, url: str, payload: dict[str, Any], label: str) -> dict[str, Any]:
    response = await http_request(conn, "POST", url, headers={**conn.bearer, "Content-Type": "application/json"}, json=payload)
    return ensure_ok(response, label)


_SIZE_RE = re.compile(r"(\d+)\s*[x*]\s*(\d+)")


def resolve_size(params: dict[str, Any], separator: str = "x") -> str:
    """size 预设（可带说明文字）/ Custom → 'WxH'；'auto'、'2K' 等档位原样返回。"""
    size = str(params.get("size", "auto") or "auto")
    if size == "Custom":
        width, height = int(params.get("custom_width", 1024)), int(params.get("custom_height", 1024))
    else:
        match = _SIZE_RE.search(size)
        if not match:
            return size
        width, height = int(match.group(1)), int(match.group(2))
    return f"{width}{separator}{height}"


def image_fields(request: GenerationRequest) -> dict[str, Any]:
    params = request.params
    fields: dict[str, Any] = {}
    size = resolve_size(params)
    if is_set(size):
        fields["size"] = size
    for key in ("quality", "background", "moderation", "output_format"):
        if is_set(params.get(key)):
            fields[key] = params[key]
    if params.get("output_format") in ("jpeg", "webp") and params.get("output_compression") is not None:
        fields["output_compression"] = int(params["output_compression"])
    if params.get("n"):
        fields["n"] = int(params["n"])
    return fields


def image_prompt(request: GenerationRequest) -> str:
    if request.system_prompt:
        return f"{request.system_prompt}\n\n{request.prompt}".strip()
    return request.prompt


@register_protocol("openai_images")
async def run_openai_images(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    conn = connect(provider)
    fields: dict[str, Any] = {"model": route.model, "prompt": image_prompt(request), **image_fields(request)}
    fields.update(parse_extra(request.params.get("extra")))

    if request.images:
        files = [
            ("image[]" if len(request.images) > 1 else "image", (f"image_{i}.png", png_bytes(image), "image/png"))
            for i, image in enumerate(request.images)
        ]
        data = {key: str(value) for key, value in fields.items()}
        response = await http_request(conn, "POST", conn.url("images/edits"), headers=conn.bearer, data=data, files=files)
        body = ensure_ok(response, f"{conn.name} image edit")
    else:
        body = await request_json(conn, conn.url("images/generations"), fields, f"{conn.name} image generation")
    return await parse_image_data(conn, body)


async def parse_image_data(conn, body: dict[str, Any]) -> GenerationResult:
    result = GenerationResult()
    for item in body.get("data") or []:
        if not isinstance(item, dict):
            continue
        if item.get("b64_json"):
            result.images.append(decode_image(item["b64_json"]))
        elif item.get("url"):
            result.images.append(await download_image(conn, item["url"]))
        if item.get("revised_prompt"):
            result.text = item["revised_prompt"]
    if not result.images:
        raise RuntimeError(f"{conn.name} returned no image: {str(body)[:500]}")
    return result
