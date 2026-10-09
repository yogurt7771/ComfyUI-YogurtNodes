"""节点基类：子类只需声明 _NODE_NAME 与 FAMILY / PROVIDER。"""

from __future__ import annotations

import inspect
from typing import Any

from comfy_api.latest import IO

from ...node_registry import category_from_module_name
from .params import Param, integer, text
from .runtime import default_provider, image_outputs, images_from_input, resolve_route, run_request
from .specs import (
    HISTORY_IO_TYPE,
    PROVIDER_IO_TYPE,
    PROVIDERS,
    FamilySpec,
    GenerationRequest,
    ModelSpec,
    ProviderConfig,
    ProviderSpec,
    Route,
)

CUSTOM_MODEL_LABEL = "Custom model"


def _node_id(cls: type) -> str:
    return f"Yogurt{cls.__name__}"


def _category(cls: type) -> str:
    # 与 node_registry 的规则一致：包路径分类 + _SUBCATEGORY
    return f"{category_from_module_name(cls.__module__)}/{cls._SUBCATEGORY}"


def _description(cls: type) -> str:
    # 直接查类字典：ComfyNode.DESCRIPTION 是会回调 define_schema 的 classproperty
    for klass in cls.__mro__:
        value = klass.__dict__.get("DESCRIPTION")
        if isinstance(value, str):
            return value
    return inspect.cleandoc(cls.__doc__ or "")


class ProviderNode(IO.ComfyNode):
    PROVIDER: ProviderSpec
    _SUBCATEGORY = "Providers"

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id=_node_id(cls),
            display_name=cls._NODE_NAME,
            category=_category(cls),
            description=_description(cls),
            inputs=[param.to_input() for param in cls.PROVIDER.fields],
            outputs=[IO.Custom(PROVIDER_IO_TYPE).Output(display_name="provider")],
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        spec = cls.PROVIDER
        return IO.NodeOutput(ProviderConfig(kind=spec.kind, route_key=spec.route_key, settings=dict(kwargs)))


class ModelFamilyNode(IO.ComfyNode):
    FAMILY: FamilySpec
    _SUBCATEGORY = "Image"

    @classmethod
    def _models(cls) -> tuple[ModelSpec, ...]:
        family = cls.FAMILY
        if family.custom_params is None:
            return family.models
        return family.models + (_custom_model_spec(family),)

    @classmethod
    def define_schema(cls):
        family = cls.FAMILY
        inputs: list[Any] = []
        if family.image_input:
            inputs.append(IO.Image.Input(family.image_input, tooltip="Image to process."))
        if family.task in ("image", "text"):
            inputs.append(
                IO.String.Input("prompt", multiline=True, default="", tooltip="Prompt sent to the model.")
            )
        inputs.append(
            IO.DynamicCombo.Input(
                "model",
                options=[
                    IO.DynamicCombo.Option(model.label, [param.to_input() for param in model.params])
                    for model in cls._models()
                ],
                tooltip="Model version; the inputs below change with the selected model.",
            )
        )
        inputs.extend(param.to_input() for param in family.params)
        if family.system_prompt is not None:
            inputs.append(
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=family.system_prompt,
                    optional=True,
                    advanced=True,
                    tooltip="System instruction sent with the request.",
                )
            )
        if family.max_images:
            inputs.append(
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, family.max_images + 1)],
                        min=0,
                    ),
                    tooltip=f"Optional reference image(s); batches are expanded. Up to {family.max_images}.",
                )
            )
        if family.history:
            inputs.append(IO.Custom(HISTORY_IO_TYPE).Input("history", optional=True, tooltip="Previous turns."))
        inputs.append(
            IO.Custom(PROVIDER_IO_TYPE).Input(
                "provider",
                optional=True,
                tooltip="Provider node. When unconnected, the official API is used with keys from "
                "api_key.json or environment variables.",
            )
        )
        inputs.append(
            integer("retry_count", 1, min=1, max=10, tooltip="Total attempts on failure.", advanced=True).to_input()
        )
        return IO.Schema(
            node_id=_node_id(cls),
            display_name=cls._NODE_NAME,
            category=_category(cls),
            description=_description(cls),
            inputs=inputs,
            outputs=_outputs(family.task),
        )

    @classmethod
    async def execute(cls, model: dict, **kwargs) -> IO.NodeOutput:
        family = cls.FAMILY
        values = dict(model)
        label = values.pop("model")
        spec = next((item for item in cls._models() if item.label == label), None)
        if spec is None:
            raise ValueError(f"Unknown model option: {label}")

        provider: ProviderConfig = kwargs.get("provider") or default_provider(family.default_provider)
        params = {param.id: values.get(param.id, param.default) for param in spec.params}
        params.update({param.id: kwargs.get(param.id, param.default) for param in family.params})
        route = _resolve(spec, provider, params, family)

        images = images_from_input(kwargs.get("images"))
        limit = spec.max_images if spec.max_images is not None else family.max_images
        if family.max_images and len(images) > limit:
            raise ValueError(f"{label} accepts at most {limit} reference image(s), got {len(images)}.")
        if family.image_input:
            images = images_from_input(kwargs.get(family.image_input)) + images

        history = list(kwargs.get("history") or [])
        request = GenerationRequest(
            task=family.task,
            prompt=kwargs.get("prompt", ""),
            system_prompt=kwargs.get("system_prompt") or "",
            images=images,
            params=params,
            seed=int(params.get("seed", 0) or 0),
            history=history,
        )
        result = await run_request(
            f"{cls._NODE_NAME} ({label})", provider, route, request, kwargs.get("retry_count", 1)
        )

        if family.task == "text":
            history.append(("user", request.prompt))
            history.append(("assistant", result.text))
            return IO.NodeOutput(result.text, result.thought, history)
        batch, image_list = image_outputs(result.images)
        if family.task == "upscale":
            return IO.NodeOutput(batch)
        return IO.NodeOutput(batch, image_list, result.text, result.thought)


def _outputs(task: str) -> list[Any]:
    if task == "text":
        return [
            IO.String.Output(display_name="text"),
            IO.String.Output(display_name="thought"),
            IO.Custom(HISTORY_IO_TYPE).Output(display_name="history"),
        ]
    if task == "upscale":
        return [IO.Image.Output(display_name="image")]
    return [
        IO.Image.Output(display_name="image", tooltip="All images as one batch (first image only if sizes differ)."),
        IO.Image.Output(display_name="images", is_output_list=True),
        IO.String.Output(display_name="text"),
        IO.String.Output(display_name="thought"),
    ]


def _custom_model_spec(family: FamilySpec) -> ModelSpec:
    params: tuple[Param, ...] = (
        text("model_id", tooltip="Model id as the provider expects it."),
    ) + tuple(family.custom_params or ())
    return ModelSpec(label=CUSTOM_MODEL_LABEL, routes={}, params=params)


def _resolve(spec: ModelSpec, provider: ProviderConfig, params: dict[str, Any], family: FamilySpec) -> Route:
    if spec.label != CUSTOM_MODEL_LABEL:
        return resolve_route(spec, provider)
    model_id = str(params.pop("model_id", "") or "").strip()
    if not model_id:
        raise ValueError("model_id is required for a custom model")
    provider_spec = PROVIDERS.get(provider.kind)
    protocol = provider_spec.custom_protocols.get(family.task) if provider_spec else None
    if not protocol:
        raise ValueError(f"Provider '{provider.kind}' does not support custom {family.task} models")
    return Route(protocol=protocol, model=model_id, options=dict(family.custom_options))
