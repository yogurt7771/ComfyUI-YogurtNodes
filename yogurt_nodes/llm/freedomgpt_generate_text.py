from typing import List
from .freedomgpt_client import FreedomGPTClient


class FreedomGPTGenerateText:
    """
    FreedomGPT Generate Text
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 尝试获取动态模型列表，如果失败则使用缓存列表
        try:
            # 创建临时客户端获取列表（使用环境变量或缓存）
            temp_client = FreedomGPTClient()
            models = temp_client.get_all_models()
        except Exception:
            # 如果API调用失败，使用缓存列表
            models = FreedomGPTClient.get_cached_models()

        # 确保列表不为空
        if not models:
            models = FreedomGPTClient.get_cached_models()

        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "API key for accessing FreedomGPT API",
                    },
                ),
                "model_name": (
                    models,
                    {
                        "default": "liberty",
                        "tooltip": "FreedomGPT model name",
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Sampling probability threshold, controls output diversity",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 40,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Top-K sampling parameter, limits vocabulary to top K tokens",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 0,
                        "max": 32768,
                        "step": 1,
                        "tooltip": "Maximum number of tokens in the generated text",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Batch size for processing multiple requests",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of retry attempts if the request fails",
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

    RETURN_TYPES = ("STRING", "HISTORY", "*")
    RETURN_NAMES = ("text", "history", "payload")

    FUNCTION = "generate_text"

    _NODE_NAME = "FreedomGPT Generate Text"
    DESCRIPTION = "Generate text using FreedomGPT API"
    CATEGORY = "YogurtNodes/LLM"

    def generate_text(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        batch_size: int,
        retry_count: int,
        chat_template: str,
        proxy_url: str,
        history: List[tuple[str, str]] | None = None,
    ):
        client = FreedomGPTClient(api_key, proxy_url)

        text, history, payload = client.generate_text(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            chat_template=chat_template,
            top_k=top_k,
            batch_size=batch_size,
            history=history,
        )
        return (text, history, payload)