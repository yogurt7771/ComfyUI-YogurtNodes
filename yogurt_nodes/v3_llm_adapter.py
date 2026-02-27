from __future__ import annotations

import inspect
from typing import Any

try:
    from comfy_api.latest import IO
    _IO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    IO = None  # type: ignore[assignment]
    _IO_IMPORT_ERROR = exc


_INPUT_TYPE_TO_IO = {
    "STRING": lambda: IO.String,
    "INT": lambda: IO.Int,
    "FLOAT": lambda: IO.Float,
    "BOOLEAN": lambda: IO.Boolean,
    "IMAGE": lambda: IO.Image,
    "MASK": lambda: IO.Mask,
}

_OUTPUT_TYPE_TO_IO = {
    "STRING": lambda: IO.String,
    "INT": lambda: IO.Int,
    "FLOAT": lambda: IO.Float,
    "BOOLEAN": lambda: IO.Boolean,
    "IMAGE": lambda: IO.Image,
    "MASK": lambda: IO.Mask,
}


def _build_common_input_kwargs(
    options: dict[str, Any], optional: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs: dict[str, Any] = {"optional": optional}
    for key in (
        "display_name",
        "tooltip",
        "lazy",
        "advanced",
        "socketless",
        "force_input",
        "raw_link",
    ):
        if key in options:
            kwargs[key] = options.pop(key)
    return kwargs, options


def _input_from_v1(
    input_id: str, io_type: Any, options_in: dict[str, Any], optional: bool
):
    options = dict(options_in or {})
    common_kwargs, options = _build_common_input_kwargs(options, optional)

    # v1 有些 STRING 带 values，本质是下拉框
    if io_type == "STRING" and "values" in options:
        combo_options = options.pop("values")
        kwargs: dict[str, Any] = {
            "options": combo_options,
            "default": options.pop(
                "default", combo_options[0] if combo_options else None
            ),
        }
        if "control_after_generate" in options:
            kwargs["control_after_generate"] = options.pop(
                "control_after_generate"
            )
        extra_dict = options or None
        return IO.Combo.Input(
            input_id, **kwargs, **common_kwargs, extra_dict=extra_dict
        )

    if isinstance(io_type, list):
        kwargs = {
            "options": io_type,
            "default": options.pop("default", io_type[0] if io_type else None),
        }
        if "control_after_generate" in options:
            kwargs["control_after_generate"] = options.pop(
                "control_after_generate"
            )
        extra_dict = options or None
        return IO.Combo.Input(
            input_id, **kwargs, **common_kwargs, extra_dict=extra_dict
        )

    if io_type == "STRING":
        kwargs: dict[str, Any] = {
            "default": options.pop("default", None),
            "multiline": options.pop("multiline", False),
        }
        if "placeholder" in options:
            kwargs["placeholder"] = options.pop("placeholder")
        if "dynamic_prompts" in options:
            kwargs["dynamic_prompts"] = options.pop("dynamic_prompts")
        extra_dict = options or None
        return IO.String.Input(
            input_id, **kwargs, **common_kwargs, extra_dict=extra_dict
        )

    if io_type == "INT":
        kwargs = {}
        for key in ("default", "min", "max", "step", "control_after_generate"):
            if key in options:
                kwargs[key] = options.pop(key)
        extra_dict = options or None
        return IO.Int.Input(
            input_id, **kwargs, **common_kwargs, extra_dict=extra_dict
        )

    if io_type == "FLOAT":
        kwargs = {}
        for key in ("default", "min", "max", "step", "round"):
            if key in options:
                kwargs[key] = options.pop(key)
        extra_dict = options or None
        return IO.Float.Input(
            input_id, **kwargs, **common_kwargs, extra_dict=extra_dict
        )

    if io_type == "BOOLEAN":
        kwargs = {}
        for key in ("default", "label_on", "label_off"):
            if key in options:
                kwargs[key] = options.pop(key)
        extra_dict = options or None
        return IO.Boolean.Input(
            input_id, **kwargs, **common_kwargs, extra_dict=extra_dict
        )

    if io_type in {"ANY", "*"}:
        extra_dict = options or None
        return IO.AnyType.Input(input_id, **common_kwargs, extra_dict=extra_dict)

    io_cls_factory = _INPUT_TYPE_TO_IO.get(io_type)
    if io_cls_factory is not None:
        io_cls = io_cls_factory()
        extra_dict = options or None
        return io_cls.Input(input_id, **common_kwargs, extra_dict=extra_dict)

    # 未知类型按自定义类型处理（如 HISTORY）
    extra_dict = options or None
    return IO.Custom(str(io_type)).Input(
        input_id, **common_kwargs, extra_dict=extra_dict
    )


def _output_from_v1(io_type: str, display_name: str | None, is_output_list: bool):
    kwargs = {"display_name": display_name, "is_output_list": is_output_list}
    if io_type in {"ANY", "*"}:
        return IO.AnyType.Output(**kwargs)

    io_cls_factory = _OUTPUT_TYPE_TO_IO.get(io_type)
    if io_cls_factory is not None:
        return io_cls_factory().Output(**kwargs)
    return IO.Custom(str(io_type)).Output(**kwargs)


def _schema_from_v1(node_cls: type, node_id: str):
    input_types = node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {}
    required = input_types.get("required", {}) or {}
    optional = input_types.get("optional", {}) or {}

    inputs = []
    for input_id, spec in required.items():
        io_type = spec[0]
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        inputs.append(_input_from_v1(input_id, io_type, options, optional=False))
    for input_id, spec in optional.items():
        io_type = spec[0]
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        inputs.append(_input_from_v1(input_id, io_type, options, optional=True))

    return_types = list(getattr(node_cls, "RETURN_TYPES", []) or [])
    return_names = list(getattr(node_cls, "RETURN_NAMES", []) or [])
    output_is_list = list(getattr(node_cls, "OUTPUT_IS_LIST", []) or [])
    if not output_is_list:
        output_is_list = [False] * len(return_types)

    outputs = []
    for idx, io_type in enumerate(return_types):
        display_name = return_names[idx] if idx < len(return_names) else None
        is_list = output_is_list[idx] if idx < len(output_is_list) else False
        outputs.append(_output_from_v1(io_type, display_name, is_list))

    return IO.Schema(
        node_id=node_id,
        display_name=getattr(node_cls, "_NODE_NAME", node_cls.__name__),
        category=getattr(node_cls, "CATEGORY", "YogurtNodes/LLM"),
        description=getattr(node_cls, "DESCRIPTION", "") or "",
        inputs=inputs,
        outputs=outputs,
        is_output_node=bool(getattr(node_cls, "OUTPUT_NODE", False)),
        is_input_list=bool(getattr(node_cls, "INPUT_IS_LIST", False)),
    )


def wrap_llm_node_to_v3(node_id: str, node_cls: type) -> type:
    if IO is None:
        return node_cls
    module_name = getattr(node_cls, "__module__", "")
    if (
        not module_name.startswith("yogurt_nodes.llm")
        and ".yogurt_nodes.llm." not in f".{module_name}."
    ):
        return node_cls
    if issubclass(node_cls, IO.ComfyNode):
        return node_cls

    function_name = getattr(node_cls, "FUNCTION", "execute")

    class WrappedLLMNode(IO.ComfyNode):
        _NODE_NAME = getattr(node_cls, "_NODE_NAME", node_cls.__name__)
        DESCRIPTION = getattr(node_cls, "DESCRIPTION", "")
        CATEGORY = getattr(node_cls, "CATEGORY", "YogurtNodes/LLM")
        _ORIGIN_NODE_CLASS = node_cls

        @classmethod
        def define_schema(cls):
            return _schema_from_v1(node_cls, node_id=node_id)

        @classmethod
        async def execute(cls, **kwargs):
            node = node_cls()
            fn = getattr(node, function_name)
            result = fn(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

    WrappedLLMNode.__name__ = f"{node_cls.__name__}V3"
    WrappedLLMNode.__qualname__ = WrappedLLMNode.__name__
    return WrappedLLMNode


def get_v3_adapter_status() -> tuple[bool, str]:
    if IO is not None:
        return True, "ok"
    if _IO_IMPORT_ERROR is not None:
        return False, str(_IO_IMPORT_ERROR)
    return False, "unknown error"
