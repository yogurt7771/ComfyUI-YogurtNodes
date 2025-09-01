from typing import Optional
from typing_extensions import List

import torch
import torchvision

from .freedomgpt_client import FreedomGPTClient


class FreedomGPTGenerateImage:
    """
    FreedomGPT Generate Image
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
                    FreedomGPTClient.get_image_models(),
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
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of retry attempts if the request fails",
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

    RETURN_TYPES = ("IMAGE", "STRING", "HISTORY")
    RETURN_NAMES = ("images", "text", "history")

    FUNCTION = "generate_image"

    _NODE_NAME = "FreedomGPT Generate Image"
    DESCRIPTION = "Generate images using FreedomGPT API"
    CATEGORY = "YogurtNodes/LLM"

    def generate_image(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        number_of_images: int,
        retry_count: int,
        proxy_url: str,
        history: List[tuple[str, str]] | None = None,
    ):
        client = FreedomGPTClient(api_key, proxy_url)

        pil_images, response_text, history = client.generate_image(
            model_name=model_name,
            prompt=prompt,
            number_of_images=number_of_images,
            retry_count=retry_count,
            system_prompt=system_prompt,
            history=history,
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