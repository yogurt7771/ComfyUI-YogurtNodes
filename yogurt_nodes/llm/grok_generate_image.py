import asyncio
import json
from typing import Optional
from typing_extensions import List

import torch
import torchvision

from ..utils import GrokClient


class GrokGenerateImage:
    """
    Grok Generate Image
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
                        "tooltip": "API key for accessing xAI API",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Base URL for xAI API (leave blank for official API)",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "grok-imagine-image",
                        "tooltip": "xAI image generation model name",
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
                "resolution": (
                    ["auto", "1k", "2k"],
                    {
                        "default": "auto",
                        "tooltip": "Resolution for generated images",
                    },
                ),
                "n": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of images to generate",
                    },
                ),
                "response_format": (
                    ["url", "b64_json"],
                    {
                        "default": "url",
                        "tooltip": "Response format for generated images",
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
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("IMAGE", "STRING", "HISTORY")
    RETURN_NAMES = ("image", "text", "history")

    FUNCTION = "generate_image"

    _NODE_NAME = "Grok Generate Image"
    DESCRIPTION = "Generate image using xAI Grok API and return as torch.Tensor (h,w,c) and text"
    CATEGORY = "YogurtNodes/LLM"

    async def generate_image(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
        n: int,
        response_format: str,
        retry_count: int,
        chat_template: str,
        proxy_url: str,
        seed: int,
        timeout: int,
        extra: str = "{}",
        image: Optional[torch.Tensor] = None,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        history: List[tuple[str, str]] | None = None,
    ):
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

        client = GrokClient(api_key, base_url, proxy_url, timeout)
        images, text, history = await asyncio.to_thread(
            client.generate_image,
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            n=n,
            response_format=response_format,
            retry_count=retry_count,
            chat_template=chat_template,
            seed=seed,
            history=history,
            extra=extra_dict,
        )

        tensor_imgs = []
        for image in images:
            tensor_img = torchvision.transforms.ToTensor()(image)
            tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
            tensor_imgs.append(tensor_img)

        if len(tensor_imgs) == 1:
            return (tensor_imgs[0], text, history)
        elif len(tensor_imgs) > 1:
            return (torch.cat(tensor_imgs, dim=0), text, history)
        else:
            return (torch.zeros(1, 3, 1, 1, dtype=torch.float32), text, history)
