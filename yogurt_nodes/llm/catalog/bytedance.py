"""字节跳动模型系列：Seedream（生图），经火山方舟、BytePlus 或 OpenRouter 调用。模型版本在 models/seedream.json。"""

from __future__ import annotations

from ..framework import FamilySpec, boolean, seed
from .common import EXTRA

SEEDREAM = FamilySpec(
    task="image",
    catalog="seedream",
    default_provider="volcengine_ark",
    max_images=14,
    params=(
        seed(),
        boolean("watermark", False, advanced=True, tooltip="Add an 'AI generated' watermark."),
        EXTRA,
    ),
)
