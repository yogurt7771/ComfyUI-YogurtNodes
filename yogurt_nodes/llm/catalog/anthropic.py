"""Anthropic 模型系列：Claude（文本 / 图像理解），经 OpenRouter 或 OpenAI 兼容中转调用。模型版本在 models/claude.json。"""

from __future__ import annotations

from ..framework import FamilySpec, seed
from .common import EXTRA

CLAUDE = FamilySpec(
    task="text",
    catalog="claude",
    default_provider="openrouter",
    max_images=20,
    system_prompt="",
    history=True,
    params=(seed(), EXTRA),
)
