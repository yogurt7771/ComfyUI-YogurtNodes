from typing import List

import torch
import torchvision

from .freedomgpt_client import FreedomGPTClient


class FreedomGPTImageUnderstand:
    """
    FreedomGPT Image Understand
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 获取支持视觉的模型列表
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
                    FreedomGPTClient().get_all_text_models(),
                    {
                        "default": "liberty",
                        "tooltip": "FreedomGPT vision model name",
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
                        "tooltip": "Question or instruction about the image",
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
                        "default": 1,
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
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 99999999,
                        "step": 1,
                        "tooltip": "Random seed for reproducible results, -1 for random",
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

    _NODE_NAME = "FreedomGPT Image Understand"
    DESCRIPTION = "Understand image content using FreedomGPT vision models"
    CATEGORY = "YogurtNodes/LLM"

    def understand_image(
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
        seed: int,
        proxy_url: str,
        image: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        image3: torch.Tensor | None = None,
        image4: torch.Tensor | None = None,
        history: List[tuple[str, str]] | None = None,
    ):
        client = FreedomGPTClient(api_key, proxy_url)

        # 将tensor转换为PIL图像列表
        images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if len(img.shape) == 4:
                    img = img[0]
                img = img.permute(2, 0, 1)
                images.append(torchvision.transforms.ToPILImage()(img))

        text, history, payload = client.understand_image(
            model_name=model_name,
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            chat_template=chat_template,
            top_k=top_k,
            batch_size=batch_size,
            seed=seed,
            history=history,
        )
        return (text, history, payload)
