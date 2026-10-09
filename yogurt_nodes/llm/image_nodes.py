"""生图节点：每个模型系列一个节点，版本差异由 catalog 中的声明决定。"""

from . import protocols  # noqa: F401
from .catalog import alibaba, bytedance, gemini, generic, openai, xai
from .framework import ModelFamilyNode


class NanoBananaGenerateImage(ModelFamilyNode):
    """Nano Banana node.

    Nano Banana image generation and editing (2.1 / 2 / 2 Lite / Pro / 1).
    """
    _NODE_NAME = "Nano Banana"
    DESCRIPTION = "Nano Banana image generation and editing (2.1 / 2 / 2 Lite / Pro / 1)."
    FAMILY = gemini.NANO_BANANA


class GPTImageGenerateImage(ModelFamilyNode):
    """GPT Image node.

    OpenAI GPT Image generation and editing (2.5 / 2 / 1.5 / 1 / 1 Mini).
    """
    _NODE_NAME = "GPT Image"
    DESCRIPTION = "OpenAI GPT Image generation and editing (2.5 / 2 / 1.5 / 1 / 1 Mini)."
    FAMILY = openai.GPT_IMAGE


class GrokImagineGenerateImage(ModelFamilyNode):
    """Grok Imagine node.

    xAI Grok Imagine image generation and editing.
    """
    _NODE_NAME = "Grok Imagine"
    DESCRIPTION = "xAI Grok Imagine image generation and editing."
    FAMILY = xai.GROK_IMAGINE


class SeedreamGenerateImage(ModelFamilyNode):
    """Seedream node.

    ByteDance Seedream image generation and editing (5.0 Pro / Flash / Lite, 4.5, 4.0).
    """
    _NODE_NAME = "Seedream"
    DESCRIPTION = "ByteDance Seedream image generation and editing (5.0 Pro / Flash / Lite, 4.5, 4.0)."
    FAMILY = bytedance.SEEDREAM


class QwenImageGenerateImage(ModelFamilyNode):
    """Qwen Image node.

    Alibaba Qwen Image generation and editing.
    """
    _NODE_NAME = "Qwen Image"
    DESCRIPTION = "Alibaba Qwen Image generation and editing."
    FAMILY = alibaba.QWEN_IMAGE


class WanImageGenerateImage(ModelFamilyNode):
    """Wan Image node.

    Alibaba Wan image generation and editing.
    """
    _NODE_NAME = "Wan Image"
    DESCRIPTION = "Alibaba Wan image generation and editing."
    FAMILY = alibaba.WAN_IMAGE


class CustomModelGenerateImage(ModelFamilyNode):
    """Image Generation (Custom Model) node.

    Image generation with any model id on the connected provider.
    """
    _NODE_NAME = "Image Generation (Custom Model)"
    DESCRIPTION = "Image generation with any model id on the connected provider."
    FAMILY = generic.CUSTOM_IMAGE
