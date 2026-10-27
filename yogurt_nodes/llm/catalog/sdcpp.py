"""stable-diffusion.cpp（sd-server 原生 /sdcpp/v1 API）的选项表，供 Extra 节点使用。

生成走通用的 Custom Model 节点 + stable-diffusion.cpp 供应商，sdcpp 专有参数经 extra_params 端口传入。
采样器 / 调度器名称与 sd-server 的 GET /sdcpp/v1/capabilities 一致，缓存模式取自 examples/server/routes_sdcpp.cpp
（均为 2026-10 的 master）；新名称可通过 extra JSON 直接传入。
"""

from __future__ import annotations

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
