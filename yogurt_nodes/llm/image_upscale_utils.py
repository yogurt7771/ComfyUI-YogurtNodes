from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image


def parse_extra_json(extra: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(extra or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tensor_batch_to_pil_images(image: torch.Tensor) -> List[Image.Image]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"IMAGE tensor must be [N,H,W,C], got shape={tuple(image.shape)}")

    pil_images: List[Image.Image] = []
    for sample in image:
        array = (
            sample.detach()
            .cpu()
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )
        channels = array.shape[-1]
        if channels == 1:
            pil_image = Image.fromarray(array[..., 0], mode="L").convert("RGB")
        elif channels >= 4:
            pil_image = Image.fromarray(array[..., :4], mode="RGBA")
        else:
            pil_image = Image.fromarray(array[..., :3], mode="RGB")
        pil_images.append(pil_image)
    return pil_images


def pil_images_to_tensor_batch(images: List[Image.Image]) -> torch.Tensor:
    if not images:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    tensors = []
    expected_shape = None
    for image in images:
        rgb_image = image.convert("RGB")
        array = np.asarray(rgb_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy())
        if expected_shape is None:
            expected_shape = tuple(tensor.shape)
        elif tuple(tensor.shape) != expected_shape:
            raise ValueError(
                "Upscale API returned images with different sizes; "
                "ComfyUI IMAGE batches require matching H/W/C shapes"
            )
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)
