import asyncio
import json
from typing import List

from ..utils import OpenRouterClient


class OpenRouterGenerateText:
    """OpenRouter Generate Text node.

    Generate text using OpenRouter API
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 获取基础设施提供商列表
        infrastructure_providers = OpenRouterClient.get_infrastructure_providers()

        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "API key for accessing OpenRouter API",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "anthropic/claude-3.5-sonnet",
                        "tooltip": "OpenRouter model name",
                    },
                ),
                "infrastructure_provider": (
                    infrastructure_providers,
                    {
                        "default": "auto",
                        "tooltip": "Infrastructure provider (auto, azure, aws, etc.) - Optional",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "System-level prompt that affects the overall conversation style",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Main prompt content input by the user",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Sampling temperature, higher values produce more random outputs",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Sampling probability threshold, controls output diversity",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 0,
                        "max": 32768,
                        "step": 1,
                        "tooltip": "Maximum number of tokens in the generated text",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of retries when request fails",
                    },
                ),
                "chat_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "<-system->\n"
                            "{{system_instruction}}\n"
                            "<-/system->\n"
                            "<-user->\n"
                            "{{prompt}}\n"
                            "<-/user->"
                        ),
                        "tooltip": "Content template for the generated text",
                    },
                ),
                "provider_list": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Provider list (comma separated, e.g: 'openai,azure,together'), empty means use infrastructure_provider",
                    },
                ),
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "代理URL，格式: protocol://user:pass@addr:port，支持http,https,socks5,socks5h",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Random seed for generation (-1 for random)",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Timeout for the request in seconds, 0 means no timeout",
                    },
                ),
            },
            "optional": {
                "history": ("HISTORY",),
                "extra": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Extra parameters for the request, in JSON format",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING", "HISTORY", "ANY")
    RETURN_NAMES = ("text", "history", "payload")

    FUNCTION = "generate_text"

    _NODE_NAME = "OpenRouter Generate Text"
    DESCRIPTION = "Generate text using OpenRouter API"
    CATEGORY = "YogurtNodes/LLM"

    async def generate_text(
        self,
        api_key: str = "",
        model_name: str = "",
        infrastructure_provider: str = "auto",
        system_prompt: str = "",
        prompt: str = "",
        temperature: float = 1.0,
        top_p: float = 0,
        max_tokens: int = 8192,
        retry_count: int = 1,
        chat_template: str = "",
        provider_list: str = "",
        proxy_url: str = "",
        seed: int = -1,
        timeout: int = 0,
        extra: str = "{}",
        history: List[tuple[str, str]] | None = None,
    ):
        client = OpenRouterClient(api_key, proxy_url, timeout)

        try:
            extra_dict = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra_dict = {}

        # Parse provider list or use single infrastructure_provider
        provider_param = None
        if provider_list.strip():
            # Convert comma-separated string to list
            providers = [p.strip() for p in provider_list.split(",") if p.strip()]
            if providers:
                provider_param = providers
        elif infrastructure_provider != "auto":
            provider_param = infrastructure_provider

        text, history, payload = await asyncio.to_thread(
            client.generate_text,
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            provider=provider_param,
            chat_template=chat_template,
            seed=seed,
            history=history,
            extra=extra_dict,
        )
        return (text, history, payload)
