import asyncio
import json
from typing import Optional
from typing_extensions import List

import torch
import torchvision

from ..utils import OpenRouterClient
from .image_output_utils import build_image_outputs


class OpenRouterGenerateImage:
    """OpenRouter Generate Image node.

    Generate image using OpenRouter API and return as torch.Tensor (h,w,c) and text
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
                        "tooltip": "API key for accessing OpenRouter API",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "google/gemini-2.5-flash-image-preview",
                        "tooltip": "OpenRouter model name for image generation",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "System-level prompt that affects the overall image generation style",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Main prompt content for image generation",
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
                        "step": 1,
                        "tooltip": "Number of retries when request fails",
                    },
                ),
                "provider": (
                    OpenRouterClient.get_infrastructure_providers(),
                    {
                        "default": "auto",
                        "tooltip": "Infrastructure provider preference",
                    },
                ),
                "provider_list": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Provider list (comma separated, e.g: 'openai,azure,together'), empty means use provider",
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
                        "tooltip": "Content template for the image generation prompt",
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
                "aspect_ratio": (
                    [
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    {
                        "default": "auto",
                        "tooltip": "Aspect ratio for generated images",
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
                "image_size": (
                    ["1k", "2k", "4k"],
                    {
                        "default": "1k",
                        "tooltip": "Image size for the generated image (gemini only for now)",
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
                "return_text": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to request/return text output (controls modalities: image+text vs image only)",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING", "HISTORY")
    RETURN_NAMES = ("image", "images", "num_images", "text", "history")
    OUTPUT_IS_LIST = (False, True, False, False, False)

    FUNCTION = "generate_image"

    _NODE_NAME = "OpenRouter Generate Image"
    DESCRIPTION = "Generate image using OpenRouter API and return as torch.Tensor (h,w,c) and text"

    async def generate_image(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        return_text: bool,
        retry_count: int,
        provider: str,
        provider_list: str,
        chat_template: str,
        proxy_url: str,
        seed: int,
        timeout: int,
        aspect_ratio: str,
        image_size: str,
        extra: str = "{}",
        image: Optional[torch.Tensor] = None,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
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

        try:
            extra_dict = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra_dict = {}

        client = OpenRouterClient(api_key, proxy_url, timeout)

        # Parse provider list or use single provider
        provider_param = None
        if provider_list.strip():
            # Convert comma-separated string to list
            providers = [p.strip() for p in provider_list.split(",") if p.strip()]
            if providers:
                provider_param = providers
        elif provider != "auto":
            provider_param = provider

        images, text, history = await asyncio.to_thread(
            client.generate_image,
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            provider=provider_param,
            chat_template=chat_template,
            seed=seed,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            return_text=return_text,
            history=history,
            extra=extra_dict,
        )

        image_output, image_list, num_images = build_image_outputs(images)
        return (image_output, image_list, num_images, text, history)
