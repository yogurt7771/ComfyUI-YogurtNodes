"""文本 / 图像理解节点：每个模型系列一个节点，图片输入可选。"""

from . import protocols  # noqa: F401
from .catalog import anthropic, gemini, generic, openai, xai
from .framework import ModelFamilyNode


class GeminiChat(ModelFamilyNode):
    """Gemini Chat node.

    Gemini text generation with optional image understanding and chat history.
    """
    _NODE_NAME = "Gemini Chat"
    DESCRIPTION = "Gemini text generation with optional image understanding and chat history."
    _SUBCATEGORY = "Text"
    FAMILY = gemini.GEMINI


class GPTChat(ModelFamilyNode):
    """GPT Chat node.

    OpenAI GPT text generation with optional image understanding and chat history.
    """
    _NODE_NAME = "GPT Chat"
    DESCRIPTION = "OpenAI GPT text generation with optional image understanding and chat history."
    _SUBCATEGORY = "Text"
    FAMILY = openai.GPT


class GrokChat(ModelFamilyNode):
    """Grok Chat node.

    xAI Grok text generation with optional image understanding and chat history.
    """
    _NODE_NAME = "Grok Chat"
    DESCRIPTION = "xAI Grok text generation with optional image understanding and chat history."
    _SUBCATEGORY = "Text"
    FAMILY = xai.GROK


class ClaudeChat(ModelFamilyNode):
    """Claude Chat node.

    Anthropic Claude via OpenRouter or an OpenAI-compatible relay.
    """
    _NODE_NAME = "Claude Chat"
    DESCRIPTION = "Anthropic Claude via OpenRouter or an OpenAI-compatible relay."
    _SUBCATEGORY = "Text"
    FAMILY = anthropic.CLAUDE


class CustomModelChat(ModelFamilyNode):
    """Chat (Custom Model) node.

    Chat with any model id on the connected provider.
    """
    _NODE_NAME = "Chat (Custom Model)"
    DESCRIPTION = "Chat with any model id on the connected provider."
    _SUBCATEGORY = "Text"
    FAMILY = generic.CUSTOM_CHAT
