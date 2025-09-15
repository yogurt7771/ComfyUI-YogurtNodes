from typing import List

import torch
import torchvision

from .openrouter_client import OpenRouterClient


class OpenRouterImageUnderstand:
    """
    使用 OpenRouter API 理解图像内容
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
                        "tooltip": "OpenRouter vision model name",
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
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "history": ("HISTORY",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING", "HISTORY", "ANY")
    RETURN_NAMES = ("text", "history", "payload")

    FUNCTION = "understand_image"

    _NODE_NAME = "OpenRouter Image Understand"
    DESCRIPTION = "Understand image content using OpenRouter API"
    CATEGORY = "YogurtNodes/LLM"

    def understand_image(
        self,
        api_key: str = "",
        model_name: str = "",
        infrastructure_provider: str = "auto",
        system_prompt: str = "",
        prompt: str = "",
        temperature: float = 1.0,
        top_p: float = 0,
        max_tokens: int = 8192,
        retry_count: int = 3,
        chat_template: str = "",
        provider_list: str = "",
        proxy_url: str = "",
        image: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        image3: torch.Tensor | None = None,
        image4: torch.Tensor | None = None,
        history: List[tuple[str, str]] | None = None,
    ):
        # 收集所有非空图像
        images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if len(img.shape) == 4:
                    img = img[0]
                img = img.permute(2, 0, 1)
                images.append(torchvision.transforms.ToPILImage()(img))

        if not images:
            raise ValueError("At least one image must be provided")

        client = OpenRouterClient(api_key, proxy_url)
        
        # Parse provider list or use single infrastructure_provider
        provider_param = None
        if provider_list.strip():
            # Convert comma-separated string to list
            providers = [p.strip() for p in provider_list.split(",") if p.strip()]
            if providers:
                provider_param = providers
        elif infrastructure_provider != "auto":
            provider_param = infrastructure_provider
            
        text, history, payload = client.understand_image(
            model_name=model_name,
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            provider=provider_param,
            chat_template=chat_template,
            history=history,
        )
        return (text, history, payload)
