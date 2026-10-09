"""Google 模型系列：Nano Banana（生图）与 Gemini（文本 / 图像理解）。"""

from __future__ import annotations

from ..framework import FamilySpec, ModelSpec, Route, combo, integer, seed
from .common import ASPECT_RATIOS, ASPECT_RATIOS_EXTENDED, EXTRA, aspect_ratio, temperature, top_p

SAFETY_LEVEL = combo(
    "safety_level",
    ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "OFF", "DEFAULT"],
    default="BLOCK_NONE",
    advanced=True,
    tooltip="Safety threshold for all harm categories; DEFAULT sends no safety settings.",
)


def _gemini(model_id: str, openrouter: str = "", grsai: str = "", **options) -> dict[str, Route]:
    """Google 官方路由；openrouter / grsai 给出各平台上的模型 id（为空表示该平台不提供）。"""
    routes = {"gemini": Route("gemini", model_id, options)}
    if openrouter:
        is_image_model = "image" in model_id or "banana" in model_id
        routes["openrouter"] = Route("openrouter_images" if is_image_model else "openai_chat", openrouter)
    if grsai:
        routes["grsai"] = Route("grsai_draw", grsai)
    return routes


def _thinking_level(options, default=None):
    return combo(
        "thinking_level",
        options,
        default=default,
        tooltip="How much the model reasons before answering; higher is slower and costs more tokens.",
    )


def _modalities(default: str = "IMAGE"):
    return combo(
        "response_modalities",
        ["IMAGE", "IMAGE+TEXT"],
        default=default,
        advanced=True,
        tooltip="IMAGE+TEXT also returns the model's text reply.",
    )


def _resolution(options):
    return combo("resolution", options, tooltip="Output resolution.")


NANO_BANANA = FamilySpec(
    task="image",
    default_provider="google_ai",
    max_images=14,
    system_prompt="",
    params=(seed(), SAFETY_LEVEL, EXTRA),
    models=(
        ModelSpec(
            "Nano Banana 2.1",
            routes=_gemini("gemini-nano-banana-2.1", openrouter="google/gemini-nano-banana-2.1"),
            params=(
                aspect_ratio(ASPECT_RATIOS_EXTENDED),
                _resolution(["1K", "2K", "4K"]),
                _thinking_level(["MINIMAL", "MEDIUM", "HIGH"], "MEDIUM"),
                _modalities(),
            ),
        ),
        ModelSpec(
            "Nano Banana 2",
            routes=_gemini("gemini-3.1-flash-image", openrouter="google/gemini-3.1-flash-image", grsai="nano-banana-2"),
            params=(
                aspect_ratio(ASPECT_RATIOS_EXTENDED),
                _resolution(["1K", "2K", "4K"]),
                _thinking_level(["MINIMAL", "HIGH"], "MINIMAL"),
                _modalities(),
                temperature(),
                top_p(),
            ),
        ),
        ModelSpec(
            "Nano Banana 2 Lite",
            routes=_gemini("gemini-3.1-flash-lite-image", openrouter="google/gemini-3.1-flash-lite-image"),
            params=(
                aspect_ratio(ASPECT_RATIOS_EXTENDED),
                _resolution(["1K"]),
                _thinking_level(["MINIMAL", "HIGH"], "MINIMAL"),
                _modalities(),
                temperature(),
                top_p(),
            ),
        ),
        ModelSpec(
            "Nano Banana Pro",
            routes=_gemini("gemini-3-pro-image", openrouter="google/gemini-3-pro-image", grsai="nano-banana-pro"),
            params=(
                aspect_ratio(ASPECT_RATIOS),
                _resolution(["1K", "2K", "4K"]),
                _modalities("IMAGE+TEXT"),
                temperature(),
                top_p(),
            ),
        ),
        ModelSpec(
            "Nano Banana",
            routes=_gemini("gemini-2.5-flash-image", openrouter="google/gemini-2.5-flash-image", grsai="nano-banana"),
            params=(
                aspect_ratio(ASPECT_RATIOS),
                _modalities("IMAGE+TEXT"),
                temperature(),
                top_p(),
            ),
        ),
    ),
    custom_params=(
        aspect_ratio(ASPECT_RATIOS_EXTENDED),
        _resolution(["auto", "1K", "2K", "4K"]),
        _thinking_level(["AUTO", "MINIMAL", "LOW", "MEDIUM", "HIGH"], "AUTO"),
        _modalities("IMAGE+TEXT"),
        temperature(),
        top_p(),
    ),
)


def _max_output_tokens():
    return integer(
        "max_output_tokens",
        32768,
        min=16,
        max=65536,
        advanced=True,
        tooltip="Includes thinking tokens; raise it if replies come back empty or truncated.",
    )


def _text_params(levels=None, default=None, sampling=True, budget=None):
    params = []
    if levels:
        params.append(_thinking_level(levels, default))
    if budget is not None:
        low, high, value = budget
        params.append(
            integer("thinking_budget", value, min=low, max=high,
                    tooltip="Thinking token budget; -1 lets the model decide, 0 disables it where allowed.")
        )
    if sampling:
        params += [temperature(), top_p()]
    params.append(_max_output_tokens())
    return tuple(params)


GEMINI = FamilySpec(
    task="text",
    default_provider="google_ai",
    max_images=16,
    system_prompt="",
    history=True,
    params=(seed(), SAFETY_LEVEL, EXTRA),
    models=(
        ModelSpec("Gemini 3.8 Flash", _gemini("gemini-3.8-flash", openrouter="google/gemini-3.8-flash"),
                  _text_params(["LOW", "MEDIUM", "HIGH"], "MEDIUM", sampling=False)),
        ModelSpec("Gemini 3.7 Flash", _gemini("gemini-3.7-flash", openrouter="google/gemini-3.7-flash"), _text_params(["LOW", "MEDIUM", "HIGH"], "MEDIUM")),
        ModelSpec("Gemini 3.6 Flash", _gemini("gemini-3.6-flash", openrouter="google/gemini-3.6-flash"), _text_params(["LOW", "MEDIUM", "HIGH"], "MEDIUM")),
        ModelSpec("Gemini 3.5 Flash", _gemini("gemini-3.5-flash", openrouter="google/gemini-3.5-flash"),
                  _text_params(["MINIMAL", "LOW", "MEDIUM", "HIGH"], "MEDIUM")),
        ModelSpec("Gemini 3.5 Flash-Lite", _gemini("gemini-3.5-flash-lite", openrouter="google/gemini-3.5-flash-lite"),
                  _text_params(["MINIMAL", "LOW", "MEDIUM", "HIGH"], "LOW")),
        ModelSpec("Gemini 3.1 Pro", _gemini("gemini-3.1-pro-preview", openrouter="google/gemini-3.1-pro-preview"), _text_params(["LOW", "HIGH"], "HIGH")),
        ModelSpec("Gemini 3.1 Flash-Lite", _gemini("gemini-3.1-flash-lite", openrouter="google/gemini-3.1-flash-lite"), _text_params(["LOW", "HIGH"], "LOW")),
        ModelSpec("Gemini 2.5 Pro", _gemini("gemini-2.5-pro", openrouter="google/gemini-2.5-pro"), _text_params(budget=(-1, 32768, -1))),
        ModelSpec("Gemini 2.5 Flash", _gemini("gemini-2.5-flash", openrouter="google/gemini-2.5-flash"), _text_params(budget=(-1, 24576, -1))),
        ModelSpec("Gemini 2.5 Flash-Lite", _gemini("gemini-2.5-flash-lite", openrouter="google/gemini-2.5-flash-lite"), _text_params(budget=(-1, 24576, 0))),
    ),
    custom_params=_text_params(["AUTO", "MINIMAL", "LOW", "MEDIUM", "HIGH"], "AUTO"),
)
