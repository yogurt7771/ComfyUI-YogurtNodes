"""xAI 模型系列：Grok Imagine（生图）与 Grok（文本 / 图像理解）。模型版本在 models/grok_imagine.json、models/grok.json。"""

from __future__ import annotations

from ..framework import FamilySpec, combo, seed
from .common import EXTRA, count

GROK_IMAGINE = FamilySpec(
    task="image",
    catalog="grok_imagine",
    default_provider="xai",
    max_images=3,
    system_prompt="",
    params=(count(10), seed(), EXTRA),
)

GROK = FamilySpec(
    task="text",
    catalog="grok",
    default_provider="xai",
    max_images=16,
    system_prompt="",
    history=True,
    params=(seed(), combo("image_detail", ("auto", "low", "high"), advanced=True,
                          tooltip="Image detail level for vision."), EXTRA),
)
