import asyncio
import json
from typing import Optional
from typing_extensions import List

import torch
import torchvision

from ..utils import WanClient
from .image_output_utils import build_image_outputs


def collect_input_images(*images: Optional[torch.Tensor]):
    input_images = []
    for img in images:
        if img is None:
            continue
        if len(img.shape) == 4:
            img = img[0]
        pil_image = torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
        input_images.append(pil_image)
    return input_images


class WanGenerateImage:
    """
    Wan Generate/Edit Image
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
                        "tooltip": "百炼 API Key，留空时尝试读取 wan / dashscope 配置或 DASHSCOPE_API_KEY",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "DashScope API 根地址，留空时默认使用北京地域",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "wan2.7-image",
                        "tooltip": "Wan 图片模型名称，首版按统一 generation+editing 协议接入",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "本地拼接到主提示词前的系统提示词，接口仍以单轮 user 消息发送",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "图片生成或编辑提示词",
                    },
                ),
                "size": (
                    "STRING",
                    {
                        "default": "auto",
                        "multiline": False,
                        "tooltip": "输出分辨率，例如 2K、1K 或 1536*1024；auto 表示交给模型决定",
                    },
                ),
                "n": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 12,
                        "step": 1,
                        "tooltip": "输出图片数量，实际可用上限取决于模型和模式",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "反向提示词",
                    },
                ),
                "watermark": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "是否添加 AI 生成水印",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "step": 1,
                        "tooltip": "请求失败时的重试次数",
                    },
                ),
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "代理 URL，格式: protocol://user:pass@addr:port",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "随机种子，-1 表示让服务端自动生成",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "请求超时时间（秒），0 表示不限制",
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
                        "tooltip": "额外请求参数，支持补充 bbox_list、enable_sequential、thinking_mode 等高级字段",
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

    _NODE_NAME = "Wan Generate/Edit Image"
    DESCRIPTION = "使用阿里云百炼 Wan 图片模型进行文生图或图像编辑"
    CATEGORY = "YogurtNodes/LLM"

    async def generate_image(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        size: str,
        n: int,
        negative_prompt: str,
        watermark: bool,
        retry_count: int,
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
        try:
            extra_dict = json.loads(extra)
        except (json.JSONDecodeError, TypeError):
            extra_dict = {}

        input_images = collect_input_images(image, image1, image2, image3, image4)
        client = WanClient(
            api_key=api_key,
            base_url=base_url,
            proxy_url=proxy_url,
            timeout=timeout,
        )
        output_images, text, history = await asyncio.to_thread(
            client.generate_image,
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=input_images,
            size=size,
            n=n,
            negative_prompt=negative_prompt,
            watermark=watermark,
            retry_count=retry_count,
            seed=seed,
            history=history,
            extra=extra_dict,
        )
        image_output, image_list, num_images = build_image_outputs(output_images)
        return (image_output, image_list, num_images, text, history)
