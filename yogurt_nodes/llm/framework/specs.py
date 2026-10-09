"""供应商 / 模型 / 协议的声明类型与注册表。

- ProviderSpec：一个接入渠道（官方或兼容中转），声明它的配置字段、
  它使用哪一组模型路由（route_key）以及自定义模型时默认走的协议。
- ModelSpec：一个模型版本，声明自己的参数和在各渠道下的路由。
- FamilySpec：一个模型系列，对应一个节点。
- 协议（protocol）：一种 API 格式的实现，按名字注册，供应商复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from PIL import Image

from .params import Param

PROVIDER_IO_TYPE = "YOGURT_LLM_PROVIDER"
HISTORY_IO_TYPE = "HISTORY"


@dataclass(frozen=True)
class ProviderSpec:
    kind: str
    display_name: str
    route_key: str
    fields: tuple[Param, ...] = ()
    custom_protocols: dict[str, str] = field(default_factory=dict)
    connection: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class ProviderConfig:
    """供应商节点的输出，也是模型节点未连线时的默认配置。"""

    kind: str
    route_key: str
    settings: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        value = self.settings.get(key, default)
        return default if value is None else value


@dataclass(frozen=True)
class Route:
    protocol: str
    model: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    routes: dict[str, Route]
    params: tuple[Param, ...] = ()
    max_images: int | None = None
    description: str = ""


@dataclass(frozen=True)
class FamilySpec:
    task: str
    models: tuple[ModelSpec, ...]
    default_provider: str
    params: tuple[Param, ...] = ()
    max_images: int = 0
    image_input: str = ""
    system_prompt: str | None = None
    history: bool = False
    custom_params: tuple[Param, ...] | None = None
    custom_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    task: str
    prompt: str = ""
    system_prompt: str = ""
    images: list[Image.Image] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    history: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class GenerationResult:
    images: list[Image.Image] = field(default_factory=list)
    text: str = ""
    thought: str = ""


ProtocolHandler = Callable[[ProviderConfig, Route, GenerationRequest], Awaitable[GenerationResult]]

PROVIDERS: dict[str, ProviderSpec] = {}
PROTOCOLS: dict[str, ProtocolHandler] = {}


def register_provider(spec: ProviderSpec) -> ProviderSpec:
    PROVIDERS[spec.kind] = spec
    return spec


def register_protocol(name: str):
    def decorator(handler: ProtocolHandler) -> ProtocolHandler:
        PROTOCOLS[name] = handler
        return handler

    return decorator
