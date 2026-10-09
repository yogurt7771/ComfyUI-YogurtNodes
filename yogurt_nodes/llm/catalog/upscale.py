"""放大模型系列：Topaz（精确 / 生成式）与 Magnific（Creative / Precision）。"""

from __future__ import annotations

from ..framework import FamilySpec, ModelSpec, Route, boolean, combo, integer, number, text
from .common import EXTRA


def _unit(id: str, tooltip: str):
    return number(id, -1.0, min=-1.0, max=1.0, step=0.01, tooltip=f"{tooltip} -1 uses the model default.")


_SHARPEN = _unit("sharpen", "Sharpening amount (0-1).")
_DENOISE = _unit("denoise", "Noise reduction (0-1).")
_FIX_COMPRESSION = _unit("fix_compression", "Compression artifact removal (0-1).")


def _topaz(model: str, endpoint: str, **options) -> dict[str, Route]:
    return {"topaz": Route("topaz", model, {"endpoint": endpoint, **options})}


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
    default_provider="topaz",
    image_input="image",
    params=_output_params(face=True),
    models=(
        ModelSpec("Standard V2", _topaz("Standard V2", "standard"), (_SHARPEN, _DENOISE, _FIX_COMPRESSION)),
        ModelSpec("Low Resolution V2", _topaz("Low Resolution V2", "standard"), (_SHARPEN, _DENOISE, _FIX_COMPRESSION)),
        ModelSpec("High Fidelity V2", _topaz("High Fidelity V2", "standard"), (_SHARPEN, _DENOISE, _FIX_COMPRESSION)),
        ModelSpec("CGI", _topaz("CGI", "standard"), (_SHARPEN, _DENOISE)),
        ModelSpec("Text Refine", _topaz("Text Refine", "standard"),
                  (number("strength", -1.0, min=-1.0, max=1.0, step=0.01,
                          tooltip="Enhancement strength (0.01-1). -1 uses the model default."),
                   _SHARPEN, _DENOISE, _FIX_COMPRESSION)),
    ),
    custom_params=(_SHARPEN, _DENOISE, _FIX_COMPRESSION),
    custom_options={"endpoint": "standard"},
)


def _prompt():
    return text("prompt", multiline=True, tooltip="Optional guidance prompt; empty lets the model describe the image.")


def _creativity(maximum: int = 9, default: int = 3):
    return integer("creativity", default, min=1, max=maximum, tooltip="Higher adds more newly generated detail.")


def _grain():
    return (
        boolean("grain", False, advanced=True, tooltip="Add film grain."),
        combo("grain_model", ("silver", "gaussian", "grey"), advanced=True, tooltip="Ignored when grain is off."),
        number("grain_strength", 0.5, min=0.0, max=1.0, step=0.01, advanced=True, tooltip="Grain strength."),
        number("grain_size", 1.0, min=1.0, max=5.0, step=0.1, advanced=True, tooltip="Grain particle size."),
        number("grain_density", 0.5, min=0.0, max=1.0, step=0.01, advanced=True, tooltip="Grain density."),
    )


_REIMAGINE = (
    _prompt(),
    _creativity(),
    combo("subject_detection", ("All", "Foreground", "Background"), advanced=True, tooltip="Subjects to process."),
    boolean("face_enhancement", True, advanced=True, tooltip="Enhance faces."),
    number("face_enhancement_creativity", 0.0, min=0.0, max=1.0, step=0.01, advanced=True,
           tooltip="Face enhancement creativity."),
    number("face_enhancement_strength", 1.0, min=0.0, max=1.0, step=0.01, advanced=True,
           tooltip="Face sharpness relative to background."),
    boolean("face_preservation", True, advanced=True, tooltip="Preserve facial identity."),
    boolean("color_preservation", True, advanced=True, tooltip="Preserve original colors."),
)
_BLOOM = (
    _prompt(),
    _creativity(),
    integer("seed", 2, min=1, max=2000, control_after_generate=True, tooltip="Seed for reproducible results."),
    boolean("color_preservation", True, advanced=True, tooltip="Preserve original colors."),
) + _grain()
_WONDER = (
    combo("enhancement_strength", ("high", "medium", "low"), tooltip="Enhancement level."),
) + _grain()
_REDEFINE = (
    _prompt(),
    _creativity(6, 2),
    integer("texture", 1, min=1, max=5, tooltip="Amount of added texture."),
    _SHARPEN,
    _DENOISE,
)

TOPAZ_GENERATIVE = FamilySpec(
    task="upscale",
    default_provider="topaz",
    image_input="image",
    params=_output_params(face=False),
    models=(
        ModelSpec("Wonder 3.5", _topaz("Wonder 3.5", "generative"), _WONDER),
        ModelSpec("Bloom 2", _topaz("Bloom 2", "generative", autoprompt=True), _BLOOM),
        ModelSpec("Reimagine", _topaz("Reimagine", "generative"), _REIMAGINE),
        ModelSpec("Redefine", _topaz("Redefine", "generative", autoprompt=True), _REDEFINE),
        ModelSpec("Recovery V2", _topaz("Recovery V2", "generative"),
                  (number("detail", -1.0, min=-1.0, max=1.0, step=0.01,
                          tooltip="Detail level (0-1). -1 uses the model default."),)),
    ),
    custom_params=(_prompt(), _creativity()),
    custom_options={"endpoint": "generative", "autoprompt": True},
)

_CREATIVE_FIELDS = ("scale_factor", "optimized_for", "prompt", "creativity", "hdr", "resemblance",
                    "fractality", "engine", "filter_nsfw")
_PRECISION_V1_FIELDS = ("sharpen", "smart_grain", "ultra_detail", "filter_nsfw")
_PRECISION_V2_FIELDS = ("scale_factor", "flavor", "sharpen", "smart_grain", "ultra_detail", "filter_nsfw")


def _slider(id: str, tooltip: str, default: int = 0, low: int = -10, high: int = 10):
    return integer(id, default, min=low, max=high, tooltip=tooltip)


_SCALE = combo("scale_factor", ("2x", "4x", "8x", "16x"), tooltip="Upscale factor.")
_NSFW = boolean("filter_nsfw", False, advanced=True, tooltip="Let the service filter NSFW content.")

MAGNIFIC_CREATIVE = FamilySpec(
    task="upscale",
    default_provider="magnific",
    image_input="image",
    params=(_NSFW, EXTRA),
    models=(
        ModelSpec(
            "Creative",
            {"magnific": Route("magnific", "v1/ai/image-upscaler", {"fields": _CREATIVE_FIELDS})},
            (
                _SCALE,
                combo("optimized_for", (
                    "standard", "soft_portraits", "hard_portraits", "art_n_illustration", "videogame_assets",
                    "nature_n_landscapes", "films_n_photography", "3d_renders", "science_fiction_n_horror",
                ), tooltip="Content type to optimize for."),
                text("prompt", multiline=True, tooltip="Optional guidance prompt."),
                _slider("creativity", "How much new detail to invent."),
                _slider("hdr", "Detail and contrast boost."),
                _slider("resemblance", "Closeness to the original image."),
                _slider("fractality", "Detail density per pixel."),
                combo("engine", ("automatic", "magnific_illusio", "magnific_sharpy", "magnific_sparkle"),
                      tooltip="Upscaling engine."),
            ),
        ),
    ),
)


def _precision_sliders(sharpen_default: int):
    return (
        _slider("sharpen", "Edge sharpness.", sharpen_default, 0, 100),
        _slider("smart_grain", "Grain to avoid an over-smooth look.", 7, 0, 100),
        _slider("ultra_detail", "Fine detail added during upscaling.", 30, 0, 100),
    )


MAGNIFIC_PRECISION = FamilySpec(
    task="upscale",
    default_provider="magnific",
    image_input="image",
    params=(_NSFW, EXTRA),
    models=(
        ModelSpec(
            "Precision V2",
            {"magnific": Route("magnific", "v1/ai/image-upscaler-precision-v2",
                               {"fields": _PRECISION_V2_FIELDS, "integer_scale": True})},
            (_SCALE, combo("flavor", ("sublime", "photo", "photo_denoiser"), tooltip="Processing style."))
            + _precision_sliders(7),
        ),
        ModelSpec(
            "Precision V1",
            {"magnific": Route("magnific", "v1/ai/image-upscaler-precision", {"fields": _PRECISION_V1_FIELDS})},
            _precision_sliders(50),
        ),
    ),
)
