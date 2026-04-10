from __future__ import annotations

import inspect
from dataclasses import asdict
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
    "AUDIO": lambda: IO.Audio,
    "VIDEO": lambda: IO.Video,
    "MODEL": lambda: IO.Model,
    "CLIP": lambda: IO.Clip,
}

_OUTPUT_TYPE_TO_IO = {
    "STRING": lambda: IO.String,
    "INT": lambda: IO.Int,
    "FLOAT": lambda: IO.Float,
    "BOOLEAN": lambda: IO.Boolean,
    "IMAGE": lambda: IO.Image,
    "MASK": lambda: IO.Mask,
    "AUDIO": lambda: IO.Audio,
    "VIDEO": lambda: IO.Video,
    "MODEL": lambda: IO.Model,
    "CLIP": lambda: IO.Clip,
}

_UPLOAD_KEY_TO_TYPE = {
    "image_upload": lambda: IO.UploadType.image,
    "audio_upload": lambda: IO.UploadType.audio,
    "video_upload": lambda: IO.UploadType.video,
    "file_upload": lambda: IO.UploadType.model,
}

_HIDDEN_KEY_TO_ENUM = {
    "UNIQUE_ID": lambda: IO.Hidden.unique_id,
    "PROMPT": lambda: IO.Hidden.prompt,
    "EXTRA_PNGINFO": lambda: IO.Hidden.extra_pnginfo,
    "DYNPROMPT": lambda: IO.Hidden.dynprompt,
    "AUTH_TOKEN_COMFY_ORG": lambda: IO.Hidden.auth_token_comfy_org,
    "API_KEY_COMFY_ORG": lambda: IO.Hidden.api_key_comfy_org,
}

_FOLDER_NAME_TO_ENUM = {
    "input": lambda: IO.FolderType.input,
    "output": lambda: IO.FolderType.output,
    "temp": lambda: IO.FolderType.temp,
}

_INTERNAL_WIDGET_INPUTS = {"audioUI", "upload"}


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


def _extract_upload_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    upload_kwargs: dict[str, Any] = {}
    for option_key, upload_factory in _UPLOAD_KEY_TO_TYPE.items():
        if options.pop(option_key, False):
            upload_kwargs["upload"] = upload_factory()
            break

    image_folder = options.pop("image_folder", None)
    if image_folder is not None and image_folder in _FOLDER_NAME_TO_ENUM:
        upload_kwargs["image_folder"] = _FOLDER_NAME_TO_ENUM[image_folder]()

    return upload_kwargs


def _has_audio_upload(spec: Any) -> bool:
    return (
        isinstance(spec, tuple)
        and len(spec) > 1
        and isinstance(spec[1], dict)
        and spec[1].get("audio_upload") is True
    )


def _input_types_need_audio_ui(input_types: dict[str, dict[str, Any]]) -> bool:
    for section_name in ("required", "optional"):
        section = input_types.get(section_name, {}) or {}
        if any(_has_audio_upload(spec) for spec in section.values()):
            return True
    return False


def _inject_audio_ui(node_info: dict[str, Any]) -> dict[str, Any]:
    input_info = node_info.setdefault("input", {})
    required = input_info.setdefault("required", {})
    if "audioUI" in required:
        return node_info

    reordered_required = {}
    inserted = False
    for input_id, input_spec in required.items():
        reordered_required[input_id] = input_spec
        if input_id == "audio":
            reordered_required["audioUI"] = ("AUDIO_UI", {})
            inserted = True
    if not inserted:
        reordered_required["audioUI"] = ("AUDIO_UI", {})

    input_info["required"] = reordered_required
    node_info["input_order"] = {
        key: list(value.keys()) for key, value in input_info.items()
    }
    return node_info


def _call_with_supported_kwargs(fn, kwargs: dict[str, Any]):
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in _INTERNAL_WIDGET_INPUTS
    }
    signature = inspect.signature(fn)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return fn(**filtered_kwargs)

    supported_kwargs = {
        key: value
        for key, value in filtered_kwargs.items()
        if key in signature.parameters
    }
    return fn(**supported_kwargs)


def _input_from_v1(
    input_id: str, io_type: Any, options_in: dict[str, Any], optional: bool
):
    options = dict(options_in or {})
    common_kwargs, options = _build_common_input_kwargs(options, optional)
    upload_kwargs = _extract_upload_kwargs(options)

    if io_type == "STRING" and "values" in options:
        combo_options = options.pop("values")
        kwargs: dict[str, Any] = {"options": combo_options}
        if "default" in options:
            kwargs["default"] = options.pop("default")
        if "control_after_generate" in options:
            kwargs["control_after_generate"] = options.pop(
                "control_after_generate"
            )
        extra_dict = options or None
        return IO.Combo.Input(
            input_id,
            **kwargs,
            **common_kwargs,
            **upload_kwargs,
            extra_dict=extra_dict,
        )

    if isinstance(io_type, list):
        kwargs = {"options": io_type}
        if "default" in options:
            kwargs["default"] = options.pop("default")
        if "control_after_generate" in options:
            kwargs["control_after_generate"] = options.pop(
                "control_after_generate"
            )
        extra_dict = options or None
        return IO.Combo.Input(
            input_id,
            **kwargs,
            **common_kwargs,
            **upload_kwargs,
            extra_dict=extra_dict,
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


def _hidden_from_v1(input_types: dict[str, dict[str, Any]]) -> list:
    hidden_specs = input_types.get("hidden", {}) or {}
    hidden_values = []
    for hidden_type in hidden_specs.values():
        hidden_factory = _HIDDEN_KEY_TO_ENUM.get(hidden_type)
        if hidden_factory is None:
            continue
        hidden_values.append(hidden_factory())
    return hidden_values


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
        category=getattr(node_cls, "CATEGORY", "YogurtNodes"),
        description=getattr(node_cls, "DESCRIPTION", "") or "",
        inputs=inputs,
        outputs=outputs,
        hidden=_hidden_from_v1(input_types),
        is_output_node=bool(getattr(node_cls, "OUTPUT_NODE", False)),
        is_input_list=bool(getattr(node_cls, "INPUT_IS_LIST", False)),
    )


def wrap_node_to_v3(node_id: str, node_cls: type) -> type:
    if IO is None:
        return node_cls
    if issubclass(node_cls, IO.ComfyNode):
        return node_cls

    function_name = getattr(node_cls, "FUNCTION", "execute")

    class WrappedNode(IO.ComfyNode):
        _NODE_NAME = getattr(node_cls, "_NODE_NAME", node_cls.__name__)
        DESCRIPTION = getattr(node_cls, "DESCRIPTION", "")
        CATEGORY = getattr(node_cls, "CATEGORY", "YogurtNodes")
        _ORIGIN_NODE_CLASS = node_cls

        @classmethod
        def define_schema(cls):
            return _schema_from_v1(node_cls, node_id=node_id)

        @classmethod
        def GET_NODE_INFO_V1(cls) -> dict[str, Any]:  # noqa: N802
            node_info = asdict(cls.GET_SCHEMA().get_v1_info(cls))
            input_types = node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {}
            if _input_types_need_audio_ui(input_types):
                return _inject_audio_ui(node_info)
            return node_info

        @classmethod
        async def execute(cls, **kwargs):
            node = node_cls()
            fn = getattr(node, function_name)
            result = _call_with_supported_kwargs(fn, kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        @classmethod
        def validate_inputs(cls, **kwargs):
            validator = getattr(node_cls, "VALIDATE_INPUTS", None)
            if validator is None:
                raise NotImplementedError
            return _call_with_supported_kwargs(validator, kwargs)

        @classmethod
        def fingerprint_inputs(cls, **kwargs):
            fingerprint = getattr(node_cls, "IS_CHANGED", None)
            if fingerprint is None:
                raise NotImplementedError
            return _call_with_supported_kwargs(fingerprint, kwargs)

        @classmethod
        def check_lazy_status(cls, **kwargs) -> list[str]:
            node = node_cls()
            checker = getattr(node, "check_lazy_status", None)
            if checker is None:
                return [
                    name
                    for name, value in kwargs.items()
                    if name not in _INTERNAL_WIDGET_INPUTS and value is None
                ]
            return _call_with_supported_kwargs(checker, kwargs)

    WrappedNode.__name__ = f"{node_cls.__name__}WrappedV3"
    WrappedNode.__qualname__ = WrappedNode.__name__
    return WrappedNode


def get_v3_node_adapter_status() -> tuple[bool, str]:
    if IO is not None:
        return True, "ok"
    if _IO_IMPORT_ERROR is not None:
        return False, str(_IO_IMPORT_ERROR)
    return False, "unknown error"
