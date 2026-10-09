"""供应商节点：只负责接入配置，输出连到模型节点的 provider 端口。"""

from .catalog import providers as catalog
from .framework import ProviderNode


class GoogleAIStudioProvider(ProviderNode):
    """Provider: Google AI Studio node.

    Google AI Studio (Gemini API) provider; connect to Gemini / Nano Banana nodes.
    """
    _NODE_NAME = "Provider: Google AI Studio"
    DESCRIPTION = "Google AI Studio (Gemini API) provider; connect to Gemini / Nano Banana nodes."
    PROVIDER = catalog.GOOGLE_AI


class GoogleGenAICompatibleProvider(ProviderNode):
    """Provider: Google GenAI Compatible node.

    Any Gemini-API-compatible endpoint with a custom base URL.
    """
    _NODE_NAME = "Provider: Google GenAI Compatible"
    DESCRIPTION = "Any Gemini-API-compatible endpoint with a custom base URL."
    PROVIDER = catalog.GENAI_COMPATIBLE


class VertexAIProvider(ProviderNode):
    """Provider: Vertex AI node.

    Vertex AI provider (express-mode API key or service-account credentials).
    """
    _NODE_NAME = "Provider: Vertex AI"
    DESCRIPTION = "Vertex AI provider (express-mode API key or service-account credentials)."
    PROVIDER = catalog.VERTEX_AI


class OpenAIProvider(ProviderNode):
    """Provider: OpenAI node.

    Official OpenAI API provider.
    """
    _NODE_NAME = "Provider: OpenAI"
    DESCRIPTION = "Official OpenAI API provider."
    PROVIDER = catalog.OPENAI


class OpenAICompatibleProvider(ProviderNode):
    """Provider: OpenAI Compatible node.

    Any OpenAI-compatible endpoint (chat completions / images) with a custom base URL.
    """
    _NODE_NAME = "Provider: OpenAI Compatible"
    DESCRIPTION = "Any OpenAI-compatible endpoint (chat completions / images) with a custom base URL."
    PROVIDER = catalog.OPENAI_COMPATIBLE


class OpenRouterProvider(ProviderNode):
    """Provider: OpenRouter node.

    OpenRouter provider with optional upstream provider routing.
    """
    _NODE_NAME = "Provider: OpenRouter"
    DESCRIPTION = "OpenRouter provider with optional upstream provider routing."
    PROVIDER = catalog.OPENROUTER


class XAIProvider(ProviderNode):
    """Provider: xAI node.

    xAI (Grok) API provider.
    """
    _NODE_NAME = "Provider: xAI"
    DESCRIPTION = "xAI (Grok) API provider."
    PROVIDER = catalog.XAI


class GRSAIProvider(ProviderNode):
    """Provider: GRSAI node.

    GRSAI drawing relay provider (Nano Banana / GPT Image).
    """
    _NODE_NAME = "Provider: GRSAI"
    DESCRIPTION = "GRSAI drawing relay provider (Nano Banana / GPT Image)."
    PROVIDER = catalog.GRSAI


class FreedomGPTProvider(ProviderNode):
    """Provider: FreedomGPT node.

    FreedomGPT provider; use with the custom-model chat and image nodes.
    """
    _NODE_NAME = "Provider: FreedomGPT"
    DESCRIPTION = "FreedomGPT provider; use with the custom-model chat and image nodes."
    PROVIDER = catalog.FREEDOMGPT


class VolcengineArkProvider(ProviderNode):
    """Provider: Volcengine Ark node.

    Volcengine Ark (China) provider for Seedream and Doubao models.
    """
    _NODE_NAME = "Provider: Volcengine Ark"
    DESCRIPTION = "Volcengine Ark (China) provider for Seedream and Doubao models."
    PROVIDER = catalog.VOLCENGINE_ARK


class BytePlusArkProvider(ProviderNode):
    """Provider: BytePlus ModelArk node.

    BytePlus ModelArk (international) provider for Seedream models.
    """
    _NODE_NAME = "Provider: BytePlus ModelArk"
    DESCRIPTION = "BytePlus ModelArk (international) provider for Seedream models."
    PROVIDER = catalog.BYTEPLUS_ARK


class DashScopeProvider(ProviderNode):
    """Provider: Alibaba DashScope node.

    Alibaba Cloud DashScope (Bailian) provider for Qwen Image and Wan.
    """
    _NODE_NAME = "Provider: Alibaba DashScope"
    DESCRIPTION = "Alibaba Cloud DashScope (Bailian) provider for Qwen Image and Wan."
    PROVIDER = catalog.DASHSCOPE


class TopazProvider(ProviderNode):
    """Provider: Topaz Labs node.

    Topaz Labs Image API provider.
    """
    _NODE_NAME = "Provider: Topaz Labs"
    DESCRIPTION = "Topaz Labs Image API provider."
    PROVIDER = catalog.TOPAZ


class MagnificProvider(ProviderNode):
    """Provider: Magnific node.

    Magnific upscaler API provider.
    """
    _NODE_NAME = "Provider: Magnific"
    DESCRIPTION = "Magnific upscaler API provider."
    PROVIDER = catalog.MAGNIFIC
