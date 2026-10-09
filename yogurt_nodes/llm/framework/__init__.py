"""LLM / 生图 / 放大节点的公共框架。

新增模型：在 catalog 中追加 ModelSpec；新增模型系列：再加一个继承 ModelFamilyNode 的薄节点类。
新增供应商：在 catalog/providers.py 注册 ProviderSpec 并加一个继承 ProviderNode 的薄节点类；
只有遇到全新的 API 格式时才需要在 protocols 中注册新协议。
"""

from .nodes import CUSTOM_MODEL_LABEL, ModelFamilyNode, ProviderNode  # noqa: F401
from .params import Param, boolean, combo, images, integer, number, seed, text  # noqa: F401
from .runtime import cancellable_sleep, describe_exception  # noqa: F401
from .specs import (  # noqa: F401
    PROTOCOLS,
    PROVIDERS,
    FamilySpec,
    GenerationRequest,
    GenerationResult,
    ModelSpec,
    ProviderConfig,
    ProviderSpec,
    Route,
    register_protocol,
    register_provider,
)
