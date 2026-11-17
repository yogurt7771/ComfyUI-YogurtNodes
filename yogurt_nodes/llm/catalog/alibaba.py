"""阿里模型系列：Qwen Image 与 Wan Image（生图 / 编辑），经百炼 DashScope 或 OpenRouter 调用。
模型版本在 models/qwen_image.json、models/wan_image.json。"""

from __future__ import annotations

from ..framework import FamilySpec, boolean, seed
from .common import EXTRA, count

QWEN_IMAGE = FamilySpec(
    task="image",
    catalog="qwen_image",
    default_provider="dashscope",
    max_images=3,
    system_prompt="",
    params=(
        count(6),
        seed(),
        boolean("watermark", False, advanced=True, tooltip="Add a watermark."),
        EXTRA,
    ),
)

WAN_IMAGE = FamilySpec(
    task="image",
    catalog="wan_image",
    default_provider="dashscope",
    max_images=9,
    system_prompt="",
    params=(
        count(4),
        seed(),
        boolean("watermark", False, advanced=True, tooltip="Add a watermark."),
        EXTRA,
    ),
)
