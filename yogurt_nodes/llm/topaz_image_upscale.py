from __future__ import annotations

import torch

from ..utils import TopazClient
from .image_upscale_utils import (
    parse_extra_json,
    pil_images_to_tensor_batch,
    tensor_batch_to_pil_images,
)


class TopazImageUpscaleAPI:
    """Topaz Image Upscale API node."""

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
                        "tooltip": "Topaz API key. Empty value still sends the request and lets the API return its own auth error.",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Topaz Image API base URL. Empty uses https://api.topazlabs.com/image/v1.",
                    },
                ),
                "model_type": (
                    ["standard", "generative"],
                    {
                        "default": "standard",
                        "tooltip": "standard uses /enhance/async; generative uses /enhance-gen/async.",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "Standard V2",
                        "multiline": False,
                        "tooltip": "Official Topaz model name, such as Standard V2, Low Resolution V2, High Fidelity V2, Redefine, Recovery V2, or Standard MAX.",
                    },
                ),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Used to compute official output_width/output_height when those fields are 0.",
                    },
                ),
                "output_width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32000,
                        "step": 1,
                        "tooltip": "Official output_width. 0 computes width from scale_factor.",
                    },
                ),
                "output_height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32000,
                        "step": 1,
                        "tooltip": "Official output_height. 0 computes height from scale_factor unless only output_width is set.",
                    },
                ),
                "crop_to_fill": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Official crop_to_fill parameter.",
                    },
                ),
                "output_format": (
                    ["png", "jpeg", "jpg", "tiff", "tif"],
                    {
                        "default": "png",
                        "tooltip": "Official output_format parameter.",
                    },
                ),
                "face_enhancement": (
                    ["auto", "true", "false"],
                    {
                        "default": "auto",
                        "tooltip": "Official face_enhancement parameter. auto omits the field.",
                    },
                ),
                "face_enhancement_strength": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Official face_enhancement_strength, -1 omits the field.",
                    },
                ),
                "face_enhancement_creativity": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Official face_enhancement_creativity, -1 omits the field.",
                    },
                ),
                "subject_detection": (
                    ["auto", "All", "Foreground", "Background"],
                    {
                        "default": "auto",
                        "tooltip": "Official subject_detection parameter. auto omits the field.",
                    },
                ),
                "sharpen": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Official sharpen parameter, -1 omits the field.",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Official denoise parameter, -1 omits the field.",
                    },
                ),
                "fix_compression": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Official fix_compression parameter, -1 omits the field.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Official strength parameter for applicable models, -1 omits the field.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Official prompt parameter for applicable generative models.",
                    },
                ),
                "autoprompt": (
                    ["auto", "true", "false"],
                    {
                        "default": "auto",
                        "tooltip": "Official autoprompt parameter. auto omits the field.",
                    },
                ),
                "creativity": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 6,
                        "step": 1,
                        "tooltip": "Official creativity parameter for applicable models, -1 omits the field.",
                    },
                ),
                "texture": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Official texture parameter for applicable models, -1 omits the field.",
                    },
                ),
                "detail": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Official detail parameter for applicable models, -1 omits the field.",
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
                        "tooltip": "Extra official or newly released Topaz form-data fields as JSON.",
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

    _NODE_NAME = "Topaz Image Upscale API"
    DESCRIPTION = "Call the Topaz Labs Image API, wait for completion, and return an IMAGE batch."

    def upscale_image(
        self,
        image: torch.Tensor,
        api_key: str,
        base_url: str,
        model_type: str,
        model_name: str,
        scale_factor: float,
        output_width: int,
        output_height: int,
        crop_to_fill: bool,
        output_format: str,
        face_enhancement: str,
        face_enhancement_strength: float,
        face_enhancement_creativity: float,
        subject_detection: str,
        sharpen: float,
        denoise: float,
        fix_compression: float,
        strength: float,
        prompt: str,
        autoprompt: str,
        creativity: int,
        texture: int,
        detail: float,
        timeout: int,
        task_timeout: int,
        poll_interval: float,
        retry_count: int,
        proxy_url: str,
        extra: str = "{}",
    ):
        extra_dict = parse_extra_json(extra)
        client = TopazClient(
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
                    model_type=model_type,
                    model=model_name,
                    scale_factor=scale_factor,
                    output_width=output_width,
                    output_height=output_height,
                    crop_to_fill=crop_to_fill,
                    output_format=output_format,
                    face_enhancement=face_enhancement,
                    face_enhancement_strength=face_enhancement_strength,
                    face_enhancement_creativity=face_enhancement_creativity,
                    subject_detection=subject_detection,
                    sharpen=sharpen,
                    denoise=denoise,
                    fix_compression=fix_compression,
                    strength=strength,
                    prompt=prompt,
                    autoprompt=autoprompt,
                    creativity=creativity,
                    texture=texture,
                    detail=detail,
                    task_timeout=task_timeout,
                    poll_interval=poll_interval,
                    extra=extra_dict,
                )
            )
        return (pil_images_to_tensor_batch(output_images),)
