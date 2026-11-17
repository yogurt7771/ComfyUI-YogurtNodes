"""通用节点：任意供应商上的任意模型（自填 model_id），用于尚未声明的模型。
只有 Custom model 一个选项，它的参数在 models/custom_image.json、models/custom_chat.json。"""

from __future__ import annotations

from ..framework import FamilySpec, seed
from .common import EXTRA, count

CUSTOM_IMAGE = FamilySpec(
    task="image",
    catalog="custom_image",
    default_provider="openai",
    max_images=16,
    system_prompt="",
    params=(count(10), seed(), EXTRA),
    extra_input=True,
)

CUSTOM_CHAT = FamilySpec(
    task="text",
    catalog="custom_chat",
    default_provider="openai",
    max_images=16,
    system_prompt="",
    history=True,
    params=(seed(), EXTRA),
)
