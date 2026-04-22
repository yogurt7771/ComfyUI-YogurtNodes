import asyncio
import json
from typing import Optional
from typing_extensions import List

import torch
import torchvision

from ..utils import FreedomGPTClient


class FreedomGPTGenerateImage:
    """FreedomGPT Generate Image node.

    Generate images using FreedomGPT API
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
                        "tooltip": "API key for accessing FreedomGPT API",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "liberty",
                        "tooltip": "FreedomGPT image generation model name",
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
                "number_of_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of images to generate",
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
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
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
    RETURN_NAMES = ("images", "text", "history")

    FUNCTION = "generate_image"

    _NODE_NAME = "FreedomGPT Generate Image"
    DESCRIPTION = "Generate images using FreedomGPT API"
    CATEGORY = "YogurtNodes/LLM"

    async def generate_image(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        number_of_images: int,
        retry_count: int,
        seed: int,
        proxy_url: str,
        timeout: int,
        extra: str = "{}",
        image: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        image3: torch.Tensor | None = None,
        image4: torch.Tensor | None = None,
        history: List[tuple[str, str]] | None = None,
    ):
        client = FreedomGPTClient(api_key, proxy_url, timeout)

        try:
            extra_dict = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra_dict = {}

        # 将tensor转换为PIL图像列表
        input_images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if len(img.shape) == 4:
                    img = img[0]
                img = img.permute(2, 0, 1)
                input_images.append(torchvision.transforms.ToPILImage()(img))

        pil_images, response_text, history = await asyncio.to_thread(
            client.generate_image,
            model_name=model_name,
            prompt=prompt,
            number_of_images=number_of_images,
            retry_count=retry_count,
            system_prompt=system_prompt,
            history=history,
            seed=seed,
            input_images=input_images,
            extra=extra_dict,
        )

        # 转换PIL图像为ComfyUI tensor格式
        if pil_images:
            tensors = []
            for pil_img in pil_images:
                # 转换为tensor格式 [1, H, W, C]
                tensor_img = torchvision.transforms.ToTensor()(pil_img)
                tensor_img = tensor_img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                tensor_img = tensor_img.unsqueeze(0)  # [H, W, C] -> [1, H, W, C]
                tensors.append(tensor_img)

            # 拼接所有图像 [N, H, W, C]
            if len(tensors) > 1:
                result_tensor = torch.cat(tensors, dim=0)
            else:
                result_tensor = tensors[0]
        else:
            # 如果没有图像，返回空白图像
            result_tensor = torch.zeros(1, 512, 512, 3)

        return (result_tensor, response_text, history)
