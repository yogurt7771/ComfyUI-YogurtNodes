"""extra 设置节点：把某个 API 的专有参数做成控件，经 extra_params 端口连到生成节点。

数值控件用 -1（或 0）、下拉用 default 表示“不发送”，由服务端按已加载模型的默认值处理。
"""

from __future__ import annotations

import re
from typing import Any

from PIL import Image

from .catalog import sdcpp
from .framework import ExtraNode, Param, boolean, combo, integer, number, text
from .framework.runtime import images_from_input
from .protocols.http import data_url

_DEFAULT = "default"
_LORA_LINE = re.compile(r"(.+?)\s*:\s*(-?\d+(?:\.\d+)?)")


def _lora_entries(value: str) -> list[dict[str, Any]]:
    entries = []
    for line in (value or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LORA_LINE.fullmatch(line)
        path, multiplier = (match.group(1), float(match.group(2))) if match else (line, 1.0)
        entries.append({"path": path, "multiplier": multiplier})
    return entries


def _image_url(value: Any) -> str:
    return data_url(images_from_input(value)[0])


def _mask_url(mask: Any) -> str:
    plane = mask if mask.dim() == 2 else mask[0]
    return data_url(Image.fromarray((plane.clamp(0, 1) * 255).round().byte().cpu().numpy()))


class SDCPPExtra(ExtraNode):
    """stable-diffusion.cpp Extra node.

    stable-diffusion.cpp sampling, LoRA, img2img / inpaint, ControlNet, IP-Adapter and hires settings for extra_params.
    """
    _NODE_NAME = "stable-diffusion.cpp Extra"
    DESCRIPTION = (
        "stable-diffusion.cpp sampling, LoRA, img2img / inpaint, ControlNet, IP-Adapter and hires settings "
        "for extra_params."
    )
    PROTOCOL = sdcpp.PROTOCOL
    PARAMS = (
        text("negative_prompt", multiline=True, tooltip="Negative prompt."),
        integer("steps", 0, min=0, max=100,
                tooltip="Sampling steps; 0 uses the server default. sd-server clamps steps to 100."),
        number("cfg", -1.0, min=-1.0, max=100.0, step=0.1, tooltip="Text CFG scale; -1 uses the server default."),
        combo("sampler", (_DEFAULT,) + sdcpp.SAMPLERS, tooltip="Sampling method; 'default' lets the server choose."),
        combo("scheduler", (_DEFAULT,) + sdcpp.SCHEDULERS, tooltip="Scheduler; 'default' lets the server choose."),
        number("distilled_guidance", -1.0, min=-1.0, max=100.0, step=0.1,
               tooltip="Distilled guidance for FLUX-style models; -1 uses the server default."),
        text("loras", multiline=True,
             tooltip="One LoRA per line as path[:multiplier], path as listed by sd-server, "
                     "e.g. my_lora.safetensors:0.8."),
        Param("init_image", "image", optional=True,
              tooltip="img2img / inpaint source; output size follows it unless set on the generation node."),
        Param("mask", "mask", optional=True, tooltip="Inpaint mask (white = repaint); used with init_image."),
        number("strength", -1.0, min=-1.0, max=1.0, step=0.01,
               tooltip="img2img denoising strength; -1 uses the server default."),
        Param("control_image", "image", optional=True,
              tooltip="ControlNet condition image; needs a ControlNet loaded on sd-server."),
        number("control_strength", -1.0, min=-1.0, max=10.0, step=0.01,
               tooltip="ControlNet strength; -1 uses the server default."),
        Param("ip_adapter_image", "image", optional=True,
              tooltip="IP-Adapter reference image (SD 1.5 / SDXL); needs --ip-adapter and --clip_vision on sd-server."),
        number("ip_adapter_strength", -1.0, min=-1.0, max=2.0, step=0.01,
               tooltip="IP-Adapter strength; -1 uses the server default (1.0)."),
        integer("clip_skip", 0, min=0, max=12, advanced=True, tooltip="CLIP skip; 0 uses the server default."),
        number("eta", -1.0, min=-1.0, max=10.0, step=0.01, advanced=True,
               tooltip="Sampler eta; -1 uses the server default."),
        number("flow_shift", -1.0, min=-1.0, max=100.0, step=0.01, advanced=True,
               tooltip="Flow shift for flow-matching models; -1 uses the server default."),
        boolean("hires", False, advanced=True, tooltip="Run a highres-fix second pass."),
        text("hires_upscaler", default="Latent", advanced=True,
             tooltip="Latent, Lanczos, Nearest, Latent (bicubic)… or an upscaler name listed by sd-server."),
        number("hires_scale", 2.0, min=1.0, max=8.0, step=0.05, advanced=True, tooltip="Highres-fix scale."),
        integer("hires_steps", 0, min=0, max=150, advanced=True,
                tooltip="Highres-fix steps; 0 uses the server default."),
        number("hires_denoise", 0.7, min=0.0, max=1.0, step=0.01, advanced=True,
               tooltip="Highres-fix denoising strength."),
        boolean("vae_tiling", False, advanced=True, tooltip="Tile VAE encode / decode to reduce memory."),
        combo("cache_mode", (_DEFAULT,) + sdcpp.CACHE_MODES, advanced=True,
              tooltip="Step cache acceleration; 'default' keeps the server setting."),
    )

    @classmethod
    def build(cls, **kwargs) -> dict[str, Any]:
        data: dict[str, Any] = {}
        sample: dict[str, Any] = {}
        guidance: dict[str, Any] = {}
        if kwargs.get("negative_prompt"):
            data["negative_prompt"] = kwargs["negative_prompt"]
        if kwargs.get("steps", 0) > 0:
            sample["sample_steps"] = int(kwargs["steps"])
        if kwargs.get("sampler", _DEFAULT) != _DEFAULT:
            sample["sample_method"] = kwargs["sampler"]
        if kwargs.get("scheduler", _DEFAULT) != _DEFAULT:
            sample["scheduler"] = kwargs["scheduler"]
        for key, target, field in (
            ("eta", sample, "eta"),
            ("flow_shift", sample, "flow_shift"),
            ("cfg", guidance, "txt_cfg"),
            ("distilled_guidance", guidance, "distilled_guidance"),
        ):
            if kwargs.get(key, -1) >= 0:
                target[field] = float(kwargs[key])
        if guidance:
            sample["guidance"] = guidance
        if sample:
            data["sample_params"] = sample

        loras = _lora_entries(kwargs.get("loras", ""))
        if loras:
            data["lora"] = loras
        if kwargs.get("init_image") is not None:
            data["init_image"] = _image_url(kwargs["init_image"])
        if kwargs.get("mask") is not None:
            data["mask_image"] = _mask_url(kwargs["mask"])
        if kwargs.get("strength", -1) >= 0:
            data["strength"] = float(kwargs["strength"])
        if kwargs.get("control_image") is not None:
            data["control_image"] = _image_url(kwargs["control_image"])
        if kwargs.get("control_strength", -1) >= 0:
            data["control_strength"] = float(kwargs["control_strength"])
        if kwargs.get("ip_adapter_image") is not None:
            data["ip_adapter_image"] = _image_url(kwargs["ip_adapter_image"])
        if kwargs.get("ip_adapter_strength", -1) >= 0:
            data["ip_adapter_strength"] = float(kwargs["ip_adapter_strength"])
        if kwargs.get("clip_skip", 0) > 0:
            data["clip_skip"] = int(kwargs["clip_skip"])
        if kwargs.get("hires"):
            hires: dict[str, Any] = {
                "enabled": True,
                "upscaler": kwargs.get("hires_upscaler") or "Latent",
                "scale": float(kwargs.get("hires_scale", 2.0)),
                "denoising_strength": float(kwargs.get("hires_denoise", 0.7)),
            }
            if kwargs.get("hires_steps", 0) > 0:
                hires["steps"] = int(kwargs["hires_steps"])
            data["hires"] = hires
        if kwargs.get("vae_tiling"):
            data["vae_tiling_params"] = {"enabled": True}
        if kwargs.get("cache_mode", _DEFAULT) != _DEFAULT:
            data["cache_mode"] = kwargs["cache_mode"]
        return data
