"""通用节点：任意供应商上的任意模型（自填 model_id），用于尚未声明的模型。"""

from __future__ import annotations

from ..framework import FamilySpec, combo, integer, seed, text
from .common import ASPECT_RATIOS_EXTENDED, EXTRA, count, temperature, top_p

CUSTOM_IMAGE = FamilySpec(
    task="image",
    default_provider="openai",
    max_images=16,
    system_prompt="",
    params=(count(10), seed(), EXTRA),
    models=(),
    custom_params=(
        combo("aspect_ratio", ASPECT_RATIOS_EXTENDED, tooltip="Sent only when the protocol supports it."),
        combo("resolution", ("auto", "512", "1K", "2K", "4K"), tooltip="Resolution tier, where supported."),
        text("size", default="auto", tooltip="Explicit size such as 1024x1024, where supported."),
        combo("quality", ("auto", "low", "medium", "high"), tooltip="Quality, where supported."),
    ),
)

CUSTOM_CHAT = FamilySpec(
    task="text",
    default_provider="openai",
    max_images=16,
    system_prompt="",
    history=True,
    params=(seed(), EXTRA),
    models=(),
    custom_params=(
        combo("reasoning_effort", ("default", "none", "minimal", "low", "medium", "high", "xhigh"),
              tooltip="Sent only when not 'default'."),
        temperature(),
        top_p(1.0),
        integer("max_output_tokens", 8192, min=16, max=262144, advanced=True, tooltip="Maximum output tokens."),
    ),
)
