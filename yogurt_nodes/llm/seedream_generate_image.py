from typing import Optional

import torch
import torchvision

from .seedream_client import SeeDreamClient


class SeeDreamGenerateImage:
    """
    SeeDream Generate Image - 豆包SeeDream图像生成节点
    支持文生图、图生图、多图生图和序列图像生成
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
                        "tooltip": "豆包SeeDream API密钥 (ARK_API_KEY)",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "doubao-seedream-4.0",
                        "tooltip": "豆包SeeDream模型名称",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "图像生成的提示词",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1024,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "生成图像的宽度",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1024,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "生成图像的高度",
                    },
                ),
                "sequential_image_generation": (
                    SeeDreamClient.get_sequential_modes(),
                    {
                        "default": "disabled",
                        "tooltip": "序列图像生成模式",
                    },
                ),
                "max_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "序列生成时的最大图像数量",
                    },
                ),
                "response_format": (
                    ["url", "b64_json"],
                    {
                        "default": "url",
                        "tooltip": "响应格式：URL或base64编码",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "请求失败时的重试次数",
                    },
                ),
                "region": (
                    "STRING",
                    {
                        "default": "cn-beijing",
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
                        "tooltip": "随机种子",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "输入图像（用于图生图）"}),
                "image1": ("IMAGE", {"tooltip": "输入图像1（用于多图生图）"}),
                "image2": ("IMAGE", {"tooltip": "输入图像2（用于多图生图）"}),
                "image3": ("IMAGE", {"tooltip": "输入图像3（用于多图生图）"}),
                "image4": ("IMAGE", {"tooltip": "输入图像4（用于多图生图）"}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "response_text")

    FUNCTION = "generate_image"

    _NODE_NAME = "SeeDream Generate Image"
    DESCRIPTION = (
        "使用豆包SeeDream API生成图像，支持文生图、图生图、多图生图和序列图像生成"
    )
    CATEGORY = "YogurtNodes/LLM"

    def generate_image(
        self,
        api_key: str,
        model: str,
        prompt: str,
        width: int,
        height: int,
        sequential_image_generation: str,
        max_images: int,
        response_format: str,
        retry_count: int,
        region: str,
        proxy_url: str,
        seed: int,
        image: Optional[torch.Tensor] = None,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
    ):
        """
        执行图像生成

        Args:
            api_key: API密钥
            model: 模型名称
            prompt: 提示词
            width: 图像宽度
            height: 图像高度
            sequential_image_generation: 序列生成模式
            max_images: 最大图像数量
            response_format: 响应格式
            retry_count: 重试次数
            proxy_url: 代理URL
            image: 主输入图像
            image1-4: 额外输入图像

        Returns:
            (生成的图像张量, 响应文本)
        """
        # 收集所有非空图像
        input_images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                # 转换ComfyUI图像格式 (batch, height, width, channels) 到PIL
                if len(img.shape) == 4:
                    img = img[0]  # 取第一个批次
                # 转换 (height, width, channels) 到 (channels, height, width)
                img = img.permute(2, 0, 1)
                # 转换到PIL图像
                pil_img = torchvision.transforms.ToPILImage()(img)
                input_images.append(pil_img)

        # 创建客户端
        client = SeeDreamClient(api_key, proxy_url)

        # 调用API
        generated_images, response_text = client.generate_image(
            model=model,
            prompt=prompt,
            images=input_images if input_images else None,
            size=f"{width}x{height}",
            sequential_image_generation=sequential_image_generation,
            max_images=max_images,
            stream=False,  # ComfyUI不支持流式
            response_format=response_format,
            watermark=False,
            region=region,
            retry_count=retry_count,
            seed=seed,
        )

        # 转换PIL图像回ComfyUI格式
        tensor_images = []
        for pil_img in generated_images:
            # PIL到tensor (channels, height, width)
            tensor_img = torchvision.transforms.ToTensor()(pil_img)
            # 转换到ComfyUI格式 (height, width, channels)
            tensor_img = tensor_img.permute(1, 2, 0)
            # 添加批次维度 (batch, height, width, channels)
            tensor_img = tensor_img.unsqueeze(0)
            tensor_images.append(tensor_img)

        # 合并所有图像
        if len(tensor_images) == 1:
            final_tensor = tensor_images[0]
        elif len(tensor_images) > 1:
            final_tensor = torch.cat(tensor_images, dim=0)
        else:
            # 如果没有图像，创建一个空白图像
            final_tensor = torch.zeros(1, 512, 512, 3, dtype=torch.float32)

        return (final_tensor, response_text)
