import asyncio
import json
from typing import List

import torch
import torchvision

from ..utils import OpenAIClient


class OpenAIImageUnderstand:
    """OpenAI Image Understand node.

    Understand image content using OpenAI vision models
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "API key for accessing OpenAI API",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Base URL for OpenAI API (leave blank for official API)",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "gpt-5.1",
                        "tooltip": "OpenAI vision model name",
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
                        "default": 4096,
                        "min": 0,
                        "max": 32768,
                        "step": 1,
                        "tooltip": "Maximum number of tokens in the generated text",
                    },
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Frequency penalty to reduce repetition",
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Presence penalty to encourage new topics",
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
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "代理URL，格式: protocol://user:pass@addr:port，支持http,https,socks5,socks5h",
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
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "history": ("HISTORY",),
                "extra": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Extra parameters for the request, in JSON format",
                    },
                ),
                "image_send_mode": (
                    ["openai", "base64"],
                    {
                        "default": "openai",
                        "tooltip": "图片发送方式: openai(标准image_url对象), base64(x.ai兼容url字段)",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING", "HISTORY", "ANY")
    RETURN_NAMES = ("text", "history", "payload")

    FUNCTION = "understand_image"

    _NODE_NAME = "OpenAI Image Understand"
    DESCRIPTION = "Understand image content using OpenAI vision models"

    async def understand_image(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        frequency_penalty: float,
        presence_penalty: float,
        retry_count: int,
        chat_template: str,
        proxy_url: str,
        timeout: int,
        extra: str = "{}",
        image_send_mode: str = "openai",
        image: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        image3: torch.Tensor | None = None,
        image4: torch.Tensor | None = None,
        history: List[tuple[str, str]] | None = None,
    ):
        client = OpenAIClient(api_key, base_url, proxy_url, timeout)

        try:
            extra_dict = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra_dict = {}

        # 将tensor转换为PIL图像列表
        images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if len(img.shape) == 4:
                    img = img[0]
                img = img.permute(2, 0, 1)
                images.append(torchvision.transforms.ToPILImage()(img))

        text, history, payload = await asyncio.to_thread(
            client.understand_image,
            model_name=model_name,
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            chat_template=chat_template,
            image_send_mode=image_send_mode,
            history=history,
            extra=extra_dict,
        )
        return (text, history, payload)
