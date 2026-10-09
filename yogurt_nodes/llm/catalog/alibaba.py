"""阿里模型系列：Qwen Image 与 Wan Image（生图 / 编辑），经百炼 DashScope 或 OpenRouter 调用。"""

from __future__ import annotations

from ..framework import FamilySpec, ModelSpec, Route, boolean, combo, integer, seed, text
from .common import EXTRA, count

_QWEN_SIZES = (
    "auto", "1024*1024", "1536*1536", "2048*2048", "1664*928", "928*1664",
    "1472*1104", "1104*1472", "1584*1056", "1056*1584", "2560*1440", "1440*2560", "Custom",
)


def _negative():
    return text("negative_prompt", multiline=True, tooltip="What to avoid in the image.")


def _custom_size(maximum: int):
    return (
        integer("custom_width", 1024, min=256, max=maximum, step=16, tooltip="Used when size is Custom."),
        integer("custom_height", 1024, min=256, max=maximum, step=16, tooltip="Used when size is Custom."),
    )


def _qwen_params():
    return (
        combo("size", _QWEN_SIZES, tooltip="Output size; total area 512x512 to 2560x2560, ratio 1:8 to 8:1."),
        *_custom_size(2560),
        _negative(),
        boolean("prompt_extend", True, tooltip="Let the service rewrite the prompt for better results."),
    )


def _qwen_routes(model_id: str, openrouter_id: str = "") -> dict[str, Route]:
    routes = {"dashscope": Route("dashscope_image", model_id)}
    if openrouter_id:
        routes["openrouter"] = Route("openrouter_images", f"qwen/{openrouter_id}")
    return routes


QWEN_IMAGE = FamilySpec(
    task="image",
    default_provider="dashscope",
    max_images=3,
    system_prompt="",
    params=(
        count(6),
        seed(),
        boolean("watermark", False, advanced=True, tooltip="Add a watermark."),
        EXTRA,
    ),
    models=(
        ModelSpec("Qwen Image 3.0 Pro", _qwen_routes("qwen-image-3.0-pro", "qwen-image-3-pro"), _qwen_params()),
        ModelSpec("Qwen Image 3.0", _qwen_routes("qwen-image-3.0", "qwen-image-3"), _qwen_params()),
        ModelSpec("Qwen Image 2.0 Pro", _qwen_routes("qwen-image-2.0-pro"), _qwen_params()),
    ),
    custom_params=_qwen_params(),
)


def _wan_params():
    return (
        combo("size", ("auto", "1K", "2K", "Custom"), tooltip="Resolution tier, or Custom width x height."),
        *_custom_size(4096),
        _negative(),
        boolean("thinking_mode", True, tooltip="Reason about the prompt before drawing."),
        boolean("enable_sequential", False, advanced=True, tooltip="Generate a related image set."),
    )


WAN_IMAGE = FamilySpec(
    task="image",
    default_provider="dashscope",
    max_images=9,
    system_prompt="",
    params=(
        count(4),
        seed(),
        boolean("watermark", False, advanced=True, tooltip="Add a watermark."),
        EXTRA,
    ),
    models=(ModelSpec("Wan 2.7 Image", {"dashscope": Route("dashscope_image", "wan2.7-image")}, _wan_params()),),
    custom_params=_wan_params(),
)
