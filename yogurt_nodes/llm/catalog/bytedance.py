"""字节跳动模型系列：Seedream（生图），经火山方舟、BytePlus 或 OpenRouter 调用。"""

from __future__ import annotations

from ..framework import FamilySpec, ModelSpec, Route, boolean, combo, integer, seed
from .common import EXTRA

_RATIOS = ("1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2", "21:9")
_TIERS = {
    "1K": ((1024, 1024), (864, 1152), (1152, 864), (1312, 736), (736, 1312), (832, 1248), (1248, 832), (1568, 672)),
    "1.5K": ((1536, 1536), (1344, 1792), (1792, 1344), (2048, 1152), (1152, 2048), (1248, 1872), (1872, 1248),
             (2352, 1008)),
    "2K": ((2048, 2048), (1728, 2304), (2304, 1728), (2848, 1600), (1600, 2848), (1664, 2496), (2496, 1664),
           (3136, 1344)),
    "3K": ((3072, 3072), (2592, 3456), (3456, 2592), (4096, 2304), (2304, 4096), (2496, 3744), (3744, 2496),
           (4704, 2016)),
    "4K": ((4096, 4096), (3520, 4704), (4704, 3520), (5504, 3040), (3040, 5504), (3328, 4992), (4992, 3328),
           (6240, 2656)),
}


def _sizes(*tiers: str) -> tuple[str, ...]:
    labels = [
        f"({tier}) {width}x{height} ({ratio})"
        for tier in tiers
        for (width, height), ratio in zip(_TIERS[tier], _RATIOS)
    ]
    return tuple(labels) + ("Custom",)


def _routes(byteplus_id: str, openrouter_id: str = "") -> dict[str, Route]:
    routes = {
        "byteplus_ark": Route("ark_images", byteplus_id),
        "volcengine_ark": Route("ark_images", f"doubao-{byteplus_id}"),
    }
    if openrouter_id:
        routes["openrouter"] = Route("openrouter_images", f"bytedance-seed/{openrouter_id}")
    return routes


def _params(sizes, max_side=6240, batch=True, thinking=True, fast=False, max_refs=10):
    params = [
        combo("size", sizes, tooltip="Output size preset; Custom uses the width and height below."),
        integer("custom_width", 2048, min=1024, max=max_side, step=2, tooltip="Used when size is Custom."),
        integer("custom_height", 2048, min=1024, max=max_side, step=2, tooltip="Used when size is Custom."),
    ]
    if batch:
        params.append(integer("max_images", 1, min=1, max=15 - 1, tooltip=(
            "Upper bound of images to generate; above 1 the model may return a related set. "
            f"Reference images plus generated images cannot exceed 15 (up to {max_refs} references).")))
    if thinking:
        params.append(boolean("thinking", True, advanced=True,
                              tooltip="Prompt-optimization reasoning for text-to-image; ignored with reference images."))
    if fast:
        params.append(combo("prompt_optimization", ("standard", "fast"), advanced=True,
                            tooltip="'fast' shortens generation when reference images are provided."))
    return tuple(params)


SEEDREAM = FamilySpec(
    task="image",
    default_provider="volcengine_ark",
    max_images=14,
    params=(
        seed(),
        boolean("watermark", False, advanced=True, tooltip="Add an 'AI generated' watermark."),
        EXTRA,
    ),
    models=(
        ModelSpec("Seedream 5.0 Pro", _routes("seedream-5-0-pro-260628", "seedream-5-0-pro"),
                  _params(_sizes("1K", "2K"), max_side=4514, batch=False, fast=True), max_images=10),
        ModelSpec("Seedream 5.0 Flash", _routes("seedream-5-0-flash-260915", "seedream-5-0-flash"),
                  _params(_sizes("1K", "1.5K", "2K"), max_side=4514, batch=False, thinking=False), max_images=10),
        ModelSpec("Seedream 5.0 Lite", _routes("seedream-5-0-260128", "seedream-5-0-lite"),
                  _params(_sizes("2K", "3K", "4K"), max_refs=14), max_images=14),
        ModelSpec("Seedream 4.5", _routes("seedream-4-5-251128", "seedream-4.5"),
                  _params(_sizes("2K", "4K")), max_images=10),
        ModelSpec("Seedream 4.0", _routes("seedream-4-0-250828"), _params(_sizes("1K", "2K", "4K")), max_images=10),
    ),
    custom_params=_params(_sizes("1K", "2K", "4K"), max_refs=14),
)
