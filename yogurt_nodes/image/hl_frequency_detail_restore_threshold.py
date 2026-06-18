import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    if t_image.dtype != torch.float32:
        t_image = t_image.float()
    return Image.fromarray(
        np.clip(
            255.0 * t_image.cpu().numpy().squeeze(),
            0,
            255,
        ).astype(np.uint8)
    )


def gaussian_blur(image: Image.Image, radius: int) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def image_to_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def np_to_image(image_np: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(image_np, 0.0, 1.0) * 255.0)).convert("RGB")


def threshold_delta(delta: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return delta
    ret = delta.copy()
    ret[np.abs(ret) < threshold] = 0
    return ret


class HLFrequencyDetailRestoreThreshold:
    """H/L frequency detail restore with independent high/low delta thresholds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail_image": ("IMAGE",),
                "keep_high_freq": ("INT", {"default": 64, "min": 0, "max": 1023}),
                "erase_low_freq": ("INT", {"default": 32, "min": 0, "max": 1023}),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 1023}),
                "high_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "low_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "H/L Frequency Detail Restore Threshold"
    DESCRIPTION = (
        "Restore high/low frequency details like LayerStyle's H/L Frequency "
        "Detail Restore, with thresholds applied to high and low delta maps."
    )

    def execute(
        self,
        image,
        detail_image,
        keep_high_freq,
        erase_low_freq,
        mask_blur,
        high_threshold,
        low_threshold,
        mask=None,
    ):
        b_images = []
        l_images = []
        l_masks = []
        ret_images = []

        for b in image:
            b_images.append(torch.unsqueeze(b, 0))

        for l in detail_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == "RGBA":
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new("L", m.size, "white"))

        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert("L"))

        max_batch = max(len(b_images), len(l_images), len(l_masks))

        for i in range(max_batch):
            background_tensor = b_images[i] if i < len(b_images) else b_images[-1]
            background_image = tensor2pil(background_tensor).convert("RGB")
            detail_tensor = l_images[i] if i < len(l_images) else l_images[-1]
            detail_pil = tensor2pil(detail_tensor).convert("RGB")
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            background_np = image_to_np(background_image)
            detail_np = image_to_np(detail_pil)
            blur_detail_np = image_to_np(gaussian_blur(detail_pil, keep_high_freq))

            high_delta = threshold_delta(
                detail_np - blur_detail_np,
                float(high_threshold),
            )

            if erase_low_freq:
                low_np = image_to_np(gaussian_blur(background_image, erase_low_freq))
                low_delta = threshold_delta(
                    low_np - background_np,
                    float(low_threshold),
                )
                low_np = background_np + low_delta
            else:
                low_np = background_np.copy()

            # Linear light with a 0.5-centered high-frequency map:
            # low + 2 * (0.5 + 0.5 * high_delta) - 1 == low + high_delta.
            ret_image = np_to_image(low_np + high_delta)

            _mask = ImageChops.invert(_mask)
            if mask_blur > 0:
                _mask = gaussian_blur(_mask, mask_blur)
            ret_image.paste(background_image, _mask)
            ret_images.append(pil2tensor(ret_image))

        return (torch.cat(ret_images, dim=0),)
