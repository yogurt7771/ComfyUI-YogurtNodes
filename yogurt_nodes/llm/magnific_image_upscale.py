from __future__ import annotations

import torch

from ..utils import MagnificClient
from .image_upscale_utils import (
    parse_extra_json,
    pil_images_to_tensor_batch,
    tensor_batch_to_pil_images,
)


class MagnificImageUpscaleAPI:
    """Magnific Image Upscale API node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input IMAGE batch to upscale."}),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Magnific API key. Empty value still sends the request and lets the API return its own auth error.",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Magnific API base URL. Empty uses https://api.magnific.com.",
                    },
                ),
                "mode": (
                    ["creative", "precision"],
                    {
                        "default": "creative",
                        "tooltip": "creative uses /v1/ai/image-upscaler; precision uses /v1/ai/image-upscaler-precision.",
                    },
                ),
                "scale_factor": (
                    ["2x", "4x", "8x", "16x"],
                    {
                        "default": "2x",
                        "tooltip": "Official creative upscale scale_factor parameter.",
                    },
                ),
                "optimized_for": (
                    [
                        "standard",
                        "soft_portraits",
                        "hard_portraits",
                        "art_n_illustration",
                        "videogame_assets",
                        "nature_n_landscapes",
                        "films_n_photography",
                        "3d_renders",
                        "science_fiction_n_horror",
                    ],
                    {
                        "default": "standard",
                        "tooltip": "Official creative optimized_for parameter.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Official creative prompt parameter.",
                    },
                ),
                "creativity": (
                    "INT",
                    {
                        "default": 0,
                        "min": -10,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Official creative creativity parameter.",
                    },
                ),
                "hdr": (
                    "INT",
                    {
                        "default": 0,
                        "min": -10,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Official creative hdr parameter.",
                    },
                ),
                "resemblance": (
                    "INT",
                    {
                        "default": 0,
                        "min": -10,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Official creative resemblance parameter.",
                    },
                ),
                "fractality": (
                    "INT",
                    {
                        "default": 0,
                        "min": -10,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Official creative fractality parameter.",
                    },
                ),
                "engine": (
                    [
                        "automatic",
                        "magnific_illusio",
                        "magnific_sharpy",
                        "magnific_sparkle",
                    ],
                    {
                        "default": "automatic",
                        "tooltip": "Official creative engine parameter.",
                    },
                ),
                "sharpen": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Official precision sharpen parameter.",
                    },
                ),
                "smart_grain": (
                    "INT",
                    {
                        "default": 7,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Official precision smart_grain parameter.",
                    },
                ),
                "ultra_detail": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Official precision ultra_detail parameter.",
                    },
                ),
                "filter_nsfw": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Official filter_nsfw parameter.",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 60,
                        "min": 1,
                        "max": 3600,
                        "step": 1,
                        "tooltip": "Per HTTP request timeout in seconds.",
                    },
                ),
                "task_timeout": (
                    "INT",
                    {
                        "default": 900,
                        "min": 1,
                        "max": 86400,
                        "step": 1,
                        "tooltip": "Maximum time to wait for each API task in seconds.",
                    },
                ),
                "poll_interval": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Seconds between task status polls.",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Retry attempts for retryable HTTP failures, including 429.",
                    },
                ),
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional proxy URL.",
                    },
                ),
            },
            "optional": {
                "extra": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Extra official or newly released Magnific JSON fields, such as webhook_url.",
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_image"

    _NODE_NAME = "Magnific Image Upscale API"
    DESCRIPTION = "Call the Magnific image upscaler API, wait for completion, and return an IMAGE batch."

    def upscale_image(
        self,
        image: torch.Tensor,
        api_key: str,
        base_url: str,
        mode: str,
        scale_factor: str,
        optimized_for: str,
        prompt: str,
        creativity: int,
        hdr: int,
        resemblance: int,
        fractality: int,
        engine: str,
        sharpen: int,
        smart_grain: int,
        ultra_detail: int,
        filter_nsfw: bool,
        timeout: int,
        task_timeout: int,
        poll_interval: float,
        retry_count: int,
        proxy_url: str,
        extra: str = "{}",
    ):
        extra_dict = parse_extra_json(extra)
        client = MagnificClient(
            api_key=api_key,
            base_url=base_url,
            proxy_url=proxy_url,
            timeout=timeout,
            retry_count=retry_count,
        )

        output_images = []
        for input_image in tensor_batch_to_pil_images(image):
            output_images.append(
                client.upscale_image(
                    image=input_image,
                    mode=mode,
                    scale_factor=scale_factor,
                    optimized_for=optimized_for,
                    prompt=prompt,
                    creativity=creativity,
                    hdr=hdr,
                    resemblance=resemblance,
                    fractality=fractality,
                    engine=engine,
                    sharpen=sharpen,
                    smart_grain=smart_grain,
                    ultra_detail=ultra_detail,
                    filter_nsfw=filter_nsfw,
                    task_timeout=task_timeout,
                    poll_interval=poll_interval,
                    extra=extra_dict,
                )
            )
        return (pil_images_to_tensor_batch(output_images),)
