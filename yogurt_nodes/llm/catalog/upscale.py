"""放大模型系列：Topaz（精确 / 生成式）与 Magnific（Creative / Precision）。
模型版本在 models/topaz_standard.json、topaz_generative.json、magnific_creative.json、magnific_precision.json。"""

from __future__ import annotations

from ..framework import FamilySpec, boolean, combo, integer, number
from .common import EXTRA


def _unit(id: str, tooltip: str):
    return number(id, -1.0, min=-1.0, max=1.0, step=0.01, tooltip=f"{tooltip} -1 uses the model default.")


def _output_params(face: bool):
    params = [
        number("scale_factor", 2.0, min=1.0, max=16.0, step=0.1,
               tooltip="Upscale factor, used when output_width and output_height are 0."),
        integer("output_width", 0, min=0, max=32000, advanced=True, tooltip="Exact width; 0 uses scale_factor."),
        integer("output_height", 0, min=0, max=32000, advanced=True, tooltip="Exact height; 0 uses scale_factor."),
        boolean("crop_to_fill", False, advanced=True, tooltip="Crop instead of letterboxing on ratio change."),
        combo("output_format", ("png", "jpeg", "tiff", "webp"), advanced=True, tooltip="Returned file format."),
    ]
    if face:
        params += [
            combo("face_enhancement", ("auto", "true", "false"), advanced=True, tooltip="Enhance faces."),
            _unit("face_enhancement_strength", "Face sharpness relative to background."),
            _unit("face_enhancement_creativity", "Face enhancement creativity."),
            combo("subject_detection", ("auto", "All", "Foreground", "Background"), advanced=True,
                  tooltip="Which subjects to process."),
        ]
    return tuple(params) + (EXTRA,)


TOPAZ_STANDARD = FamilySpec(
    task="upscale",
    catalog="topaz_standard",
    default_provider="topaz",
    image_input="image",
    params=_output_params(face=True),
)

TOPAZ_GENERATIVE = FamilySpec(
    task="upscale",
    catalog="topaz_generative",
    default_provider="topaz",
    image_input="image",
    params=_output_params(face=False),
)

_NSFW = boolean("filter_nsfw", False, advanced=True, tooltip="Let the service filter NSFW content.")

MAGNIFIC_CREATIVE = FamilySpec(
    task="upscale",
    catalog="magnific_creative",
    default_provider="magnific",
    image_input="image",
    params=(_NSFW, EXTRA),
)

MAGNIFIC_PRECISION = FamilySpec(
    task="upscale",
    catalog="magnific_precision",
    default_provider="magnific",
    image_input="image",
    params=(_NSFW, EXTRA),
)
