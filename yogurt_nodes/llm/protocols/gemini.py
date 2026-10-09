"""Gemini generateContent 协议（Google AI Studio / Vertex AI / GenAI 兼容中转）。"""

from __future__ import annotations

import io
from io import BytesIO
from typing import Any

from google.genai import types
from PIL import Image

from ...utils.gemini_client import GeminiClient
from ..framework import GenerationRequest, GenerationResult, ProviderConfig, Route, register_protocol
from .http import parse_extra

_DEFAULT_SAFETY = "BLOCK_NONE"


def build_client(provider: ProviderConfig) -> GeminiClient:
    kind = provider.kind
    if kind == "genai_compatible" and not (provider.get("api_key") and provider.get("base_url")):
        # 兼容渠道不回退到 api_key.json 的官方 key，避免把官方 key 发给第三方
        raise ValueError("Google GenAI Compatible: base_url and api_key are required.")
    common = {
        "api_key": provider.get("api_key", ""),
        "base_url": provider.get("base_url", ""),
        "proxy_url": provider.get("proxy_url", ""),
        "timeout": int(provider.get("timeout", 0) or 0),
    }
    if kind == "vertex_ai":
        credentials = provider.get("credentials_json", "")
        if common["api_key"] and not credentials:
            return GeminiClient(use_vertex_api_key=True, **common)
        return GeminiClient(
            use_vertex_ai=True,
            vertex_ai_json=credentials or None,
            vertex_ai_project=provider.get("project_id", "") or None,
            vertex_ai_region=provider.get("location", "") or None,
            **{key: value for key, value in common.items() if key != "api_key"},
        )
    return GeminiClient(**common)


def _image_part(image: Image.Image) -> types.Part:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")


def build_contents(request: GenerationRequest) -> list[types.Content]:
    contents: list[types.Content] = []
    for role, message in request.history:
        if message:
            contents.append(
                types.Content(role="model" if role == "assistant" else "user", parts=[types.Part.from_text(text=message)])
            )
    parts: list[types.Part] = []
    if request.prompt:
        parts.append(types.Part.from_text(text=request.prompt))
    parts.extend(_image_part(image) for image in request.images)
    if not parts:
        raise ValueError("Prompt or images are required")
    contents.append(types.Content(role="user", parts=parts))
    return contents


def _thinking_config(params: dict[str, Any], route: Route) -> types.ThinkingConfig | None:
    level = params.get("thinking_level")
    budget = params.get("thinking_budget")
    include = route.options.get("include_thoughts", True)
    if level and level != "AUTO":
        return types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, level), include_thoughts=include)
    if budget is not None:
        if int(budget) == 0:
            return types.ThinkingConfig(thinking_budget=0)
        return types.ThinkingConfig(thinking_budget=int(budget), include_thoughts=include)
    if level == "AUTO":
        return types.ThinkingConfig(include_thoughts=include)
    return None


def build_config(route: Route, request: GenerationRequest, client: GeminiClient) -> types.GenerateContentConfig:
    params = request.params
    config = types.GenerateContentConfig()
    for key in ("temperature", "top_p", "top_k", "max_output_tokens"):
        value = params.get(key)
        if value is not None and key not in route.options.get("omit", ()):
            setattr(config, key, value)
    if request.seed:
        config.seed = request.seed
    if request.system_prompt:
        config.system_instruction = [types.Part.from_text(text=request.system_prompt)]

    thinking = _thinking_config(params, route)
    if thinking is not None:
        config.thinking_config = thinking

    safety_level = params.get("safety_level", _DEFAULT_SAFETY)
    if safety_level and safety_level != "DEFAULT":
        config.safety_settings = client._get_safety_settings(False, safety_level)

    if request.task == "image":
        modalities = params.get("response_modalities", "IMAGE+TEXT")
        config.response_modalities = ["IMAGE"] if modalities == "IMAGE" else ["TEXT", "IMAGE"]
        image_config = types.ImageConfig()
        aspect_ratio = params.get("aspect_ratio", "auto")
        if aspect_ratio and aspect_ratio != "auto":
            image_config.aspect_ratio = aspect_ratio
        resolution = params.get("resolution")
        if resolution and resolution != "auto":
            image_config.image_size = resolution
        config.image_config = image_config
    for key, value in parse_extra(params.get("extra")).items():
        setattr(config, key, value)
    return config


def parse_response(response: Any, model: str) -> GenerationResult:
    candidates = getattr(response, "candidates", None) or []
    content = candidates[0].content if candidates else None
    if content is None or not content.parts:
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        raise ValueError(
            f"Model {model} returned no content (finish_reason={reason}, prompt_feedback={feedback})"
        )
    result = GenerationResult()
    for part in content.parts:
        is_thought = getattr(part, "thought", None) is True
        if getattr(part, "text", None):
            if is_thought:
                result.thought += part.text
            else:
                result.text += part.text
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None) and not is_thought:
            result.images.append(Image.open(BytesIO(inline.data)).convert("RGB"))
    return result


@register_protocol("gemini")
async def run_gemini(provider: ProviderConfig, route: Route, request: GenerationRequest) -> GenerationResult:
    client = build_client(provider)
    try:
        client._ensure_vertex_credentials()
        config = build_config(route, request, client)
        response = await client._generate_content_async(route.model, build_contents(request), config)
    finally:
        await client.close_async()
    result = parse_response(response, route.model)
    if request.task == "image" and not result.images:
        raise ValueError(f"Model {route.model} returned no image. Text: {result.text[:300]!r}")
    if request.task == "text" and not result.text:
        raise ValueError(f"Model {route.model} returned empty text")
    return result
