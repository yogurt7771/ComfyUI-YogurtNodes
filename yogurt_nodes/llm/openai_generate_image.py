import asyncio
import json
from typing import Optional
from typing_extensions import List

import torch
import torchvision

from ..utils import OpenAIClient
from .image_output_utils import build_image_outputs


class OpenAIGenerateImage:
    """OpenAI Generate Image node.

    Generate image using OpenAI API and return as torch.Tensor (h,w,c) and text
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
                        "default": "gpt-image-1",
                        "tooltip": "OpenAI image generation model name",
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
                "size": (
                    [
                        "auto",
                        "1024x1024(gpt-image-1)",
                        "1536x1024(gpt-image-1)",
                        "1024x1536(gpt-image-1)",
                        "256x256(dall-e-2)",
                        "512x512(dall-e-2)",
                        "1024x1024(dall-e-2)",
                        "1024x1024(dall-e-3)",
                        "1792x1024(dall-e-3)",
                        "1024x1792(dall-e-3)",
                        "1k(nano banana)",
                        "2k(nano banana)",
                        "4k(nano banana)",
                        "8k(nano banana)",
                    ],
                    {
                        "default": "auto",
                        "tooltip": "Size of the generated image",
                    },
                ),
                "quality": (
                    ["auto", "high", "standard", "hd"],
                    {
                        "default": "auto",
                        "tooltip": "Quality of the generated image",
                    },
                ),
                "style": (
                    ["vivid", "natural"],
                    {
                        "default": "vivid",
                        "tooltip": "Style of the generated image (dall-e-3 only)",
                    },
                ),
                "n": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of images to generate (dall-e-2 only)",
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
                "api_type": (
                    ["auto", "response", "image"],
                    {
                        "default": "auto",
                        "tooltip": "选择使用的API类型: auto(自动根据模型选择), response(Responses API), image(Images API)",
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
                        "tooltip": "Aspect ratio for generated images, for nano banana only",
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
                    ["upload", "base64"],
                    {
                        "default": "upload",
                        "tooltip": "输入图像发送方式: upload(文件上传, OpenAI官方兼容), base64(JSON中data URL, x.ai兼容)",
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

    _NODE_NAME = "OpenAI Generate Image"
    DESCRIPTION = (
        "Generate image using OpenAI API and return as torch.Tensor (h,w,c) and text"
    )
    CATEGORY = "YogurtNodes/LLM"

    async def generate_image(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        size: str,
        quality: str,
        style: str,
        n: int,
        response_format: str,
        retry_count: int,
        chat_template: str,
        proxy_url: str,
        api_type: str,
        seed: int,
        aspect_ratio: str,
        timeout: int,
        extra: str = "{}",
        image_send_mode: str = "upload",
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
        if size != "auto":
            if "(" in size:
                size = size[:size.find("(")]

        try:
            extra_dict = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra_dict = {}

        client = OpenAIClient(api_key, base_url, proxy_url, timeout)
        images, text, history = await asyncio.to_thread(
            client.generate_image,
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            size=size,
            aspect_ratio=aspect_ratio,
            quality=quality,
            style=style,
            n=n,
            response_format=response_format,
            retry_count=retry_count,
            chat_template=chat_template,
            seed=seed,
            history=history,
            api_type=api_type,
            image_send_mode=image_send_mode,
            extra=extra_dict,
        )

        image_output, image_list, num_images = build_image_outputs(images)
        return (image_output, image_list, num_images, text, history)
