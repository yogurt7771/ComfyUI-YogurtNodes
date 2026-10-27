"""stable-diffusion.cpp（sd-server 原生 /sdcpp/v1 API）：服务端当前加载的模型。

采样器 / 调度器名称与 sd-server 的 GET /sdcpp/v1/capabilities 一致，缓存模式取自 examples/server/routes_sdcpp.cpp
（均为 2026-10 的 master）；新名称可通过 extra JSON 直接传入。
"""

from __future__ import annotations

from ..framework import FamilySpec, ModelSpec, Route, integer, seed
from .common import EXTRA, count

SAMPLERS = (
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm",
    "ddim_trailing", "tcd", "res_multistep", "res_2s", "er_sde", "euler_cfg_pp", "euler_a_cfg_pp", "euler_ge",
    "dpm++2m_sde", "dpm++2m_sde_bt", "lms",
)
SCHEDULERS = (
    "discrete", "normal", "karras", "exponential", "ays", "gits", "sgm_uniform", "simple", "smoothstep",
    "kl_optimal", "lcm", "bong_tangent", "ltx2", "logit_normal", "flux2", "flux", "beta", "llada_image",
)
CACHE_MODES = ("disabled", "easycache", "ucache", "dbcache", "taylorseer", "cache-dit", "spectrum")

PROTOCOL = "sdcpp_img_gen"


def _dimension(id: str):
    return integer(id, 0, min=0, max=4096, step=8,
                   tooltip="0 uses the init image size, else the server default. sd-server allows 64-4096.")


SDCPP_IMAGE = FamilySpec(
    task="image",
    default_provider="sdcpp",
    max_images=16,
    params=(
        count(8),
        seed("0 picks a random seed on the server (sent as -1); changing it also forces a re-run."),
        EXTRA,
    ),
    models=(
        ModelSpec(
            "Loaded model",
            routes={"sdcpp": Route(PROTOCOL, "")},
            params=(_dimension("width"), _dimension("height")),
            description="Whatever model sd-server has loaded.",
        ),
    ),
    extra_input=True,
)
