"""Google 模型系列：Nano Banana（生图）与 Gemini（文本 / 图像理解）。模型版本在 models/nano_banana.json、models/gemini.json。"""

from __future__ import annotations

from ..framework import FamilySpec, combo, seed
from .common import EXTRA

SAFETY_LEVEL = combo(
    "safety_level",
    ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE", "OFF", "DEFAULT"],
    default="BLOCK_NONE",
    advanced=True,
    tooltip="Safety threshold for all harm categories; DEFAULT sends no safety settings.",
)

NANO_BANANA = FamilySpec(
    task="image",
    catalog="nano_banana",
    default_provider="google_ai",
    max_images=14,
    system_prompt="",
    params=(seed(), SAFETY_LEVEL, EXTRA),
)

GEMINI = FamilySpec(
    task="text",
    catalog="gemini",
    default_provider="google_ai",
    max_images=16,
    system_prompt="",
    history=True,
    params=(seed(), SAFETY_LEVEL, EXTRA),
)
