"""声明式参数：一份声明同时用于生成 V3 输入和读取运行时取值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comfy_api.latest import IO


@dataclass(frozen=True)
class Param:
    id: str
    kind: str
    default: Any = None
    options: tuple = ()
    min: float | None = None
    max: float | None = None
    step: float | None = None
    multiline: bool = False
    tooltip: str = ""
    advanced: bool = False
    optional: bool = False
    max_items: int = 0
    control_after_generate: bool = False
    io_type: str = ""

    def to_input(self):
        common = {"tooltip": self.tooltip or None, "optional": self.optional}
        if self.advanced:
            common["advanced"] = True
        if self.kind == "combo":
            return IO.Combo.Input(
                self.id,
                options=list(self.options),
                default=self.default if self.default is not None else self.options[0],
                **common,
            )
        if self.kind == "int":
            kwargs = _numeric_kwargs(self)
            if self.control_after_generate:
                kwargs["control_after_generate"] = True
            return IO.Int.Input(self.id, **kwargs, **common)
        if self.kind == "float":
            return IO.Float.Input(self.id, **_numeric_kwargs(self), **common)
        if self.kind == "bool":
            return IO.Boolean.Input(self.id, default=bool(self.default), **common)
        if self.kind == "string":
            return IO.String.Input(
                self.id,
                default=self.default or "",
                multiline=self.multiline,
                **common,
            )
        if self.kind == "images":
            return IO.Autogrow.Input(
                self.id,
                template=IO.Autogrow.TemplateNames(
                    IO.Image.Input("image"),
                    names=[f"image_{i}" for i in range(1, self.max_items + 1)],
                    min=0,
                ),
                tooltip=self.tooltip or None,
            )
        if self.kind == "image":
            return IO.Image.Input(self.id, **common)
        if self.kind == "custom":
            return IO.Custom(self.io_type).Input(self.id, **common)
        raise ValueError(f"Unknown param kind: {self.kind}")


def _numeric_kwargs(param: Param) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"default": param.default}
    for key in ("min", "max", "step"):
        value = getattr(param, key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def combo(id: str, options, default=None, tooltip: str = "", advanced: bool = False) -> Param:
    options = tuple(options)
    return Param(id, "combo", default=options[0] if default is None else default, options=options,
                 tooltip=tooltip, advanced=advanced)


def integer(id: str, default: int, min: int | None = None, max: int | None = None,
            step: int | None = None, tooltip: str = "", advanced: bool = False,
            control_after_generate: bool = False) -> Param:
    return Param(id, "int", default=default, min=min, max=max, step=step, tooltip=tooltip,
                 advanced=advanced, control_after_generate=control_after_generate)


def number(id: str, default: float, min: float | None = None, max: float | None = None,
           step: float | None = None, tooltip: str = "", advanced: bool = False) -> Param:
    return Param(id, "float", default=default, min=min, max=max, step=step, tooltip=tooltip, advanced=advanced)


def boolean(id: str, default: bool = False, tooltip: str = "", advanced: bool = False) -> Param:
    return Param(id, "bool", default=default, tooltip=tooltip, advanced=advanced)


def text(id: str, default: str = "", multiline: bool = False, tooltip: str = "",
         advanced: bool = False, optional: bool = False) -> Param:
    return Param(id, "string", default=default, multiline=multiline, tooltip=tooltip,
                 advanced=advanced, optional=optional)


def images(id: str = "images", max_items: int = 14, tooltip: str = "") -> Param:
    return Param(id, "images", max_items=max_items, tooltip=tooltip or f"Optional reference image(s), up to {max_items}.")


def seed(tooltip: str = "") -> Param:
    return integer(
        "seed",
        default=0,
        min=0,
        max=2**31 - 1,
        control_after_generate=True,
        tooltip=tooltip or "Sent to the API when the model supports it; changing it also forces a re-run. "
        "0 lets the API pick a random seed.",
    )
