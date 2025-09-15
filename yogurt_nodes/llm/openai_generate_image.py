from typing import Optional
from typing_extensions import List

import torch
import torchvision

from .openai_client import OpenAIClient


class OpenAIGenerateImage:
    """
    OpenAI Generate Image
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
                    OpenAIClient.get_image_models(),
                    {
                        "default": "gpt-5",
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
                    ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"],
                    {
                        "default": "1024x1024",
                        "tooltip": "Size of the generated image",
                    },
                ),
                "quality": (
                    ["standard", "hd"],
                    {
                        "default": "standard",
                        "tooltip": "Quality of the generated image (dall-e-3 only)",
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
                        "max": 99999999,
                        "step": 1,
                        "tooltip": "Random seed for generation (-1 for random)",
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

    RETURN_TYPES = ("IMAGE", "STRING", "HISTORY")
    RETURN_NAMES = ("image", "text", "history")

    FUNCTION = "generate_image"

    _NODE_NAME = "OpenAI Generate Image"
    DESCRIPTION = "Generate image using OpenAI API and return as torch.Tensor (h,w,c) and text"
    CATEGORY = "YogurtNodes/LLM"

    def generate_image(
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
        
        client = OpenAIClient(api_key, base_url, proxy_url)
        images, text, history = client.generate_image(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            size=size,
            quality=quality,
            style=style,
            n=n,
            response_format=response_format,
            retry_count=retry_count,
            chat_template=chat_template,
            seed=seed,
            history=history,
            api_type=api_type,
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