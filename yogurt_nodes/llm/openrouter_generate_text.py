from typing import List
from .openrouter_client import OpenRouterClient


class OpenRouterGenerateText:
    """
    OpenRouter Generate Text
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 尝试获取动态模型列表，如果失败则使用缓存列表
        try:
            # 创建临时客户端获取列表（使用环境变量或缓存）
            temp_client = OpenRouterClient()
            models = temp_client.get_all_models()
        except Exception:
            # 如果API调用失败，使用缓存列表
            models = OpenRouterClient.get_cached_models()

        # 确保列表不为空
        if not models:
            models = OpenRouterClient.get_cached_models()

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
                    models,
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
                        "default": 3,
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
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "代理URL，格式: protocol://user:pass@addr:port，支持http,https,socks5,socks5h",
                    },
                ),
            },
            "optional": {
                "history": ("HISTORY",),
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

    def generate_text(
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
        proxy_url: str = "",
        history: List[tuple[str, str]] | None = None,
    ):
        client = OpenRouterClient(api_key, proxy_url)
        text, history, payload = client.generate_text(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            provider=(
                infrastructure_provider if infrastructure_provider != "auto" else None
            ),
            chat_template=chat_template,
            history=history,
        )
        return (text, history, payload)
