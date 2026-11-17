"""OpenAI 模型系列：GPT Image（生图）与 GPT（文本 / 图像理解）。模型版本在 models/gpt_image.json、models/gpt.json。"""

from __future__ import annotations

from ..framework import FamilySpec, combo, integer, seed
from .common import EXTRA, count

GPT_IMAGE = FamilySpec(
    task="image",
    catalog="gpt_image",
    default_provider="openai",
    max_images=16,
    system_prompt="",
    params=(
        count(10),
        seed(),
        combo("output_format", ("png", "jpeg", "webp"), advanced=True, tooltip="Encoded format returned by the API."),
        integer("output_compression", 100, min=0, max=100, advanced=True, tooltip="jpeg / webp compression level."),
        combo("moderation", ("auto", "low"), advanced=True, tooltip="Content moderation strictness."),
        EXTRA,
    ),
)

GPT = FamilySpec(
    task="text",
    catalog="gpt",
    default_provider="openai",
    max_images=16,
    system_prompt="",
    history=True,
    params=(
        seed(),
        combo("image_detail", ("auto", "low", "high"), advanced=True, tooltip="Image detail level for vision."),
        EXTRA,
    ),
)
