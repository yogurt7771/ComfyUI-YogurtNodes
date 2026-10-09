"""xAI 模型系列：Grok Imagine（生图）与 Grok（文本 / 图像理解）。"""

from __future__ import annotations

from ..framework import FamilySpec, ModelSpec, Route, combo, integer, seed
from .common import EXTRA, count, temperature, top_p

GROK_ASPECT_RATIOS = (
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9",
    "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1",
)


def _image_routes(model_id: str, openrouter: bool = False) -> dict[str, Route]:
    routes = {"xai": Route("xai_images", model_id)}
    if openrouter:
        routes["openrouter"] = Route("openrouter_images", f"x-ai/{model_id}")
    return routes


def _image_params(quality: bool = False):
    params = [
        combo("aspect_ratio", GROK_ASPECT_RATIOS, tooltip="'auto' follows the reference image or model default."),
        combo("resolution", ("1K", "2K"), tooltip="Output resolution."),
    ]
    if quality:
        params.append(combo("quality", ("medium", "low"), tooltip="Generation quality."))
    return tuple(params)


GROK_IMAGINE = FamilySpec(
    task="image",
    default_provider="xai",
    max_images=3,
    system_prompt="",
    params=(count(10), seed(), EXTRA),
    models=(
        ModelSpec("Grok Imagine 2.0", _image_routes("grok-imagine-image-2.0", openrouter=True),
                  _image_params(quality=True), max_images=3),
        ModelSpec("Grok Imagine Quality", _image_routes("grok-imagine-image-quality", openrouter=True),
                  _image_params(), max_images=3),
        ModelSpec("Grok Imagine Pro", _image_routes("grok-imagine-image-pro"), _image_params(), max_images=1),
        ModelSpec("Grok Imagine", _image_routes("grok-imagine-image"), _image_params(), max_images=3),
    ),
    custom_params=_image_params(quality=True),
)


def _text_routes(model_id: str) -> dict[str, Route]:
    return {
        "xai": Route("openai_chat", model_id),
        "openrouter": Route("openai_chat", f"x-ai/{model_id}"),
    }


def _text_params(effort: bool = True):
    params = []
    if effort:
        params.append(
            combo("reasoning_effort", ("default", "none", "low", "medium", "high"),
                  tooltip="Reasoning effort; 'default' leaves it to the model.")
        )
    params += [
        temperature(),
        top_p(1.0),
        integer("max_output_tokens", 16384, min=16, max=262144, advanced=True,
                tooltip="Includes reasoning tokens."),
    ]
    return tuple(params)


GROK = FamilySpec(
    task="text",
    default_provider="xai",
    max_images=16,
    system_prompt="",
    history=True,
    params=(seed(), combo("image_detail", ("auto", "low", "high"), advanced=True,
                          tooltip="Image detail level for vision."), EXTRA),
    models=(
        ModelSpec("Grok 4.7", _text_routes("grok-4.7"), _text_params()),
        ModelSpec("Grok 4.6", _text_routes("grok-4.6"), _text_params()),
        ModelSpec("Grok 4.5", _text_routes("grok-4.5"), _text_params()),
        ModelSpec("Grok 4.3", _text_routes("grok-4.3"), _text_params()),
        ModelSpec("Grok 4.20", _text_routes("grok-4.20"), _text_params(effort=False)),
    ),
    custom_params=_text_params(),
)
