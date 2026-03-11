import asyncio
import json
from typing import Optional
from typing_extensions import List

import torch
import torchvision

from ..utils import GRSAIClient


def _parse_reference_urls(image_urls: str) -> List[str]:
    if not image_urls.strip():
        return []

    try:
        parsed = json.loads(image_urls)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    values: List[str] = []
    for line in image_urls.replace(",", "\n").splitlines():
        value = line.strip()
        if value:
            values.append(value)
    return values


class GRSAINanoBananaGenerateImage:
    """GRSAI Nano Banana image generation node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "API key for accessing the GRSAI API",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Base URL for the GRSAI API, leave blank to use config/env/default host",
                    },
                ),
                "model_name": (
                    GRSAIClient.get_models(),
                    {
                        "default": "nano-banana-pro",
                        "tooltip": "Nano Banana model name from the GRSAI documentation",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional system prompt prepended locally before the request prompt",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Prompt content for image generation or editing",
                    },
                ),
                "aspect_ratio": (
                    GRSAIClient.get_aspect_ratios(),
                    {
                        "default": "auto",
                        "tooltip": "Output image ratio documented by GRSAI",
                    },
                ),
                "image_size": (
                    ["auto", *GRSAIClient.get_image_sizes()],
                    {
                        "default": "1K",
                        "tooltip": "Output image size documented by GRSAI",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "step": 1,
                        "tooltip": "Number of retries when submit or poll fails",
                    },
                ),
                "poll_interval_ms": (
                    "INT",
                    {
                        "default": 2000,
                        "min": 100,
                        "step": 100,
                        "tooltip": "Polling interval in milliseconds for task result queries",
                    },
                ),
                "max_wait_seconds": (
                    "INT",
                    {
                        "default": 300,
                        "min": 1,
                        "step": 1,
                        "tooltip": "Maximum total time to wait for GRSAI task completion",
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
                        "tooltip": "Local prompt template used to combine the system prompt and prompt",
                    },
                ),
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Proxy URL, format: protocol://user:pass@addr:port",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Timeout for each request in seconds, 0 means no timeout",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Reference image URLs or data URLs, accepts JSON array or newline/comma separated values",
                    },
                ),
                "history": ("HISTORY",),
                "extra": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Extra request parameters in JSON format",
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

    _NODE_NAME = "GRSAI Nano Banana"
    DESCRIPTION = (
        "Generate or edit images with the GRSAI Nano Banana API and return torch tensors"
    )
    CATEGORY = "YogurtNodes/LLM"

    async def generate_image(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        aspect_ratio: str,
        image_size: str,
        retry_count: int,
        poll_interval_ms: int,
        max_wait_seconds: int,
        chat_template: str,
        proxy_url: str,
        timeout: int,
        image_urls: str = "",
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
        if not isinstance(extra_dict, dict):
            extra_dict = {}

        client = GRSAIClient(api_key, base_url, proxy_url, timeout)
        generated_images, text, history = await asyncio.to_thread(
            client.generate_image,
            model_name=model_name,
            prompt=prompt,
            images=images,
            reference_urls=_parse_reference_urls(image_urls),
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            retry_count=retry_count,
            poll_interval_ms=poll_interval_ms,
            max_wait_seconds=max_wait_seconds,
            system_prompt=system_prompt,
            chat_template=chat_template,
            history=history,
            extra=extra_dict,
        )

        tensor_imgs = []
        for generated_image in generated_images:
            tensor_img = torchvision.transforms.ToTensor()(generated_image)
            tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
            tensor_imgs.append(tensor_img)

        if len(tensor_imgs) == 1:
            return (tensor_imgs[0], text, history)
        if len(tensor_imgs) > 1:
            return (torch.cat(tensor_imgs, dim=0), text, history)
        return (torch.zeros(1, 3, 1, 1, dtype=torch.float32), text, history)
