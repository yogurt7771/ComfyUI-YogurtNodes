"""OpenAI 模型系列：GPT Image（生图）与 GPT（文本 / 图像理解）。"""

from __future__ import annotations

from ..framework import FamilySpec, ModelSpec, Route, combo, integer, seed
from .common import EXTRA, count, temperature, top_p

GPT_IMAGE_2_SIZES = (
    "auto", "1024x1024", "1024x1536", "1536x1024", "2048x2048",
    "2048x1152", "1152x2048", "3840x2160", "2160x3840", "Custom",
)
GPT_IMAGE_1_SIZES = ("auto", "1024x1024", "1024x1536", "1536x1024")


def _image_routes(model_id: str, openrouter: bool = True, grsai: bool = False) -> dict[str, Route]:
    routes = {"openai": Route("openai_images", model_id)}
    if openrouter:
        routes["openrouter"] = Route("openrouter_images", f"openai/{model_id}")
    if grsai:
        routes["grsai"] = Route("grsai_draw", model_id)
    return routes


def _image_params(sizes, qualities, backgrounds):
    params = [combo("size", sizes, tooltip="Output size; 'auto' lets the model choose.")]
    if "Custom" in sizes:
        params += [
            integer("custom_width", 1024, min=480, max=3840, step=16, tooltip="Used when size is Custom."),
            integer("custom_height", 1024, min=480, max=3840, step=16, tooltip="Used when size is Custom."),
        ]
    params += [
        combo("quality", qualities, tooltip="Higher quality is slower and costs more."),
        combo("background", backgrounds, tooltip="'transparent' requires png or webp output."),
    ]
    return tuple(params)


_QUALITY_25 = ("auto", "low", "medium", "high", "xhigh", "max")
_QUALITY = ("auto", "low", "medium", "high")
_BG_ALL = ("auto", "opaque", "transparent")

GPT_IMAGE = FamilySpec(
    task="image",
    default_provider="openai",
    max_images=16,
    system_prompt="",
    params=(
        count(10),
        seed(),
        combo("output_format", ("png", "jpeg", "webp"), advanced=True, tooltip="Encoded format returned by the API."),
        integer("output_compression", 100, min=0, max=100, advanced=True, tooltip="jpeg / webp compression level."),
        combo("moderation", ("auto", "low"), advanced=True, tooltip="Content moderation strictness."),
        EXTRA,
    ),
    models=(
        ModelSpec("GPT Image 2.5 Flare", _image_routes("gpt-image-2.5-flare"),
                  _image_params(GPT_IMAGE_2_SIZES, _QUALITY_25, _BG_ALL)),
        ModelSpec("GPT Image 2.5 Sunburst", _image_routes("gpt-image-2.5-sunburst"),
                  _image_params(GPT_IMAGE_2_SIZES, _QUALITY_25, _BG_ALL)),
        ModelSpec("GPT Image 2", _image_routes("gpt-image-2", grsai=True),
                  _image_params(GPT_IMAGE_2_SIZES, _QUALITY, ("auto", "opaque"))),
        ModelSpec("GPT Image 1.5", _image_routes("gpt-image-1.5", openrouter=False, grsai=True),
                  _image_params(GPT_IMAGE_1_SIZES, _QUALITY, _BG_ALL)),
        ModelSpec("GPT Image 1", _image_routes("gpt-image-1"), _image_params(GPT_IMAGE_1_SIZES, _QUALITY, _BG_ALL)),
        ModelSpec("GPT Image 1 Mini", _image_routes("gpt-image-1-mini"),
                  _image_params(GPT_IMAGE_1_SIZES, _QUALITY, _BG_ALL)),
    ),
    custom_params=_image_params(GPT_IMAGE_2_SIZES, _QUALITY_25, _BG_ALL),
)


def _text_routes(model_id: str) -> dict[str, Route]:
    return {
        "openai": Route("openai_chat", model_id),
        "openrouter": Route("openai_chat", f"openai/{model_id}"),
    }


def _effort(options, default="default"):
    return combo(
        "reasoning_effort",
        ("default",) + tuple(options),
        default=default,
        tooltip="Reasoning effort; 'default' leaves it to the model.",
    )


def _verbosity():
    return combo("verbosity", ("default", "low", "medium", "high"), advanced=True, tooltip="Answer length hint.")


def _max_tokens(default=16384, maximum=128000):
    return integer("max_output_tokens", default, min=16, max=maximum, advanced=True,
                   tooltip="Includes reasoning tokens; raise it if replies are empty or truncated.")


def _reasoning_model(options):
    return (_effort(options), _verbosity(), _max_tokens())


_GPT_56 = ("none", "low", "medium", "high", "xhigh", "max")
_GPT_5 = ("minimal", "low", "medium", "high")
_O_SERIES = ("low", "medium", "high")
_SAMPLING_MODEL = (temperature(), top_p(1.0), _max_tokens(16384, 32768))

GPT = FamilySpec(
    task="text",
    default_provider="openai",
    max_images=16,
    system_prompt="",
    history=True,
    params=(
        seed(),
        combo("image_detail", ("auto", "low", "high"), advanced=True, tooltip="Image detail level for vision."),
        EXTRA,
    ),
    models=(
        ModelSpec("GPT-6 Astra", _text_routes("gpt-6-astra"),
                  _reasoning_model(("low", "medium", "high", "xhigh", "max"))),
        ModelSpec("GPT-6 Sol", _text_routes("gpt-6-sol"), _reasoning_model(_GPT_56)),
        ModelSpec("GPT-6 Luna", _text_routes("gpt-6-luna"), _reasoning_model(_GPT_56)),
        ModelSpec("GPT-5.6 Sol", _text_routes("gpt-5.6-sol"), _reasoning_model(_GPT_56)),
        ModelSpec("GPT-5.6 Terra", _text_routes("gpt-5.6-terra"), _reasoning_model(_GPT_56)),
        ModelSpec("GPT-5.6 Luna", _text_routes("gpt-5.6-luna"), _reasoning_model(_GPT_56)),
        ModelSpec("GPT-5.5 Pro", _text_routes("gpt-5.5-pro"), _reasoning_model(("medium", "high", "xhigh"))),
        ModelSpec("GPT-5.5", _text_routes("gpt-5.5"), _reasoning_model(("none", "low", "medium", "high", "xhigh"))),
        ModelSpec("GPT-5", _text_routes("gpt-5"), _reasoning_model(_GPT_5)),
        ModelSpec("GPT-5 Mini", _text_routes("gpt-5-mini"), _reasoning_model(_GPT_5)),
        ModelSpec("GPT-5 Nano", _text_routes("gpt-5-nano"), _reasoning_model(_GPT_5)),
        ModelSpec("GPT-4.1", _text_routes("gpt-4.1"), _SAMPLING_MODEL),
        ModelSpec("GPT-4.1 Mini", _text_routes("gpt-4.1-mini"), _SAMPLING_MODEL),
        ModelSpec("GPT-4.1 Nano", _text_routes("gpt-4.1-nano"), _SAMPLING_MODEL),
        ModelSpec("o4-mini", _text_routes("o4-mini"), (_effort(_O_SERIES), _max_tokens())),
        ModelSpec("o3", _text_routes("o3"), (_effort(_O_SERIES), _max_tokens())),
    ),
    custom_params=(
        _effort(("none", "minimal", "low", "medium", "high", "xhigh", "max")),
        _verbosity(),
        _max_tokens(),
    ),
)
