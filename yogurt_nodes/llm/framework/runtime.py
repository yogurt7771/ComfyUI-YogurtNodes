"""运行时：解析供应商与路由、统一重试与报错、图像张量转换。"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable

import torch
from PIL import Image

import comfy.model_management as model_management

from ..image_output_utils import empty_image_tensor, pil_image_to_tensor
from ..image_upscale_utils import tensor_batch_to_pil_images
from .specs import (
    PROTOCOLS,
    PROVIDERS,
    GenerationRequest,
    GenerationResult,
    ModelSpec,
    ProviderConfig,
    Route,
)


def default_provider(kind: str) -> ProviderConfig:
    spec = PROVIDERS[kind]
    return ProviderConfig(kind=spec.kind, route_key=spec.route_key)


def supported_providers(model: ModelSpec) -> list[str]:
    return [
        spec.display_name
        for spec in PROVIDERS.values()
        if spec.kind in model.routes or spec.route_key in model.routes
    ]


def resolve_route(model: ModelSpec, provider: ProviderConfig) -> Route:
    route = model.routes.get(provider.kind) or model.routes.get(provider.route_key)
    if route is not None:
        return route
    provider_name = PROVIDERS[provider.kind].display_name if provider.kind in PROVIDERS else provider.kind
    raise ValueError(
        f"{model.label} is not available via provider '{provider_name}'. "
        f"Supported providers: {', '.join(supported_providers(model)) or 'none'}."
    )


def describe_exception(exception: BaseException | None, timeout: Any = 0) -> str:
    if exception is None:
        return "unknown error"
    message = str(exception)
    if not message and isinstance(exception, TimeoutError) and timeout:
        message = f"request exceeded timeout of {timeout}s"
    name = type(exception).__name__
    return f"{name}: {message}" if message else name


async def cancellable_sleep(seconds: float, interval: float = 0.25) -> None:
    deadline = time.monotonic() + max(seconds, 0.0)
    while True:
        model_management.throw_exception_if_processing_interrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(interval, remaining))


async def run_request(
    label: str,
    provider: ProviderConfig,
    route: Route,
    request: GenerationRequest,
    retry_count: int = 1,
) -> GenerationResult:
    handler = PROTOCOLS.get(route.protocol)
    if handler is None:
        raise ValueError(f"Protocol '{route.protocol}' is not registered")

    attempts = max(1, int(retry_count))
    last_exception: Exception | None = None
    for attempt in range(attempts):
        model_management.throw_exception_if_processing_interrupted()
        try:
            return await handler(provider, route, request)
        except Exception as exception:  # noqa: BLE001 - 统一重试后带原因抛出
            last_exception = exception
            if attempt + 1 < attempts:
                await cancellable_sleep(3)

    raise RuntimeError(
        f"{label} failed after {attempts} attempt(s): "
        f"{describe_exception(last_exception, provider.get('timeout', 0))}"
    ) from last_exception


def images_from_input(value: Any) -> list[Image.Image]:
    """Autogrow 字典、单个 IMAGE 张量或其列表 → PIL 列表（批次会展开）。"""
    if value is None:
        return []
    if isinstance(value, dict):
        tensors: Iterable[Any] = (value[key] for key in sorted(value, key=_slot_order))
    elif isinstance(value, (list, tuple)):
        tensors = value
    else:
        tensors = [value]

    result: list[Image.Image] = []
    for tensor in tensors:
        if tensor is None:
            continue
        result.extend(image.convert("RGB") for image in tensor_batch_to_pil_images(tensor))
    return result


def _slot_order(name: str) -> tuple[int, str]:
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix), name) if suffix.isdigit() else (0, name)


def image_outputs(images: list[Image.Image]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """返回 (批次张量, 单图张量列表)；尺寸不一致时批次只含第一张。"""
    if not images:
        return empty_image_tensor(), []
    tensors = [pil_image_to_tensor(image.convert("RGB")) for image in images]
    first_shape = tensors[0].shape
    if all(tensor.shape == first_shape for tensor in tensors):
        return torch.cat(tensors, dim=0), tensors
    return tensors[0], tensors
