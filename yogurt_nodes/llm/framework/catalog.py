"""模型目录：每个模型系列一份 JSON（catalog/models/<系列>.json），列出它的模型版本、各供应商下的路由和参数。

节点每次取定义或执行时都重读，改 JSON 后刷新页面即生效，不用重启 ComfyUI。格式见 EXTENDING.md。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .params import Param
from .specs import ModelSpec, Route

MODELS_DIR = Path(__file__).resolve().parents[1] / "catalog" / "models"

# 省略 default 时的取值，与 params.py 里各构造函数一致（combo 取第一个选项）。
_DEFAULTS = {"bool": False, "string": ""}


@dataclass(frozen=True)
class FamilyModels:
    models: tuple[ModelSpec, ...]
    # None 表示该系列没有 Custom model 选项。
    custom_params: tuple[Param, ...] | None = None
    custom_options: dict[str, Any] = field(default_factory=dict)


def load_family(name: str) -> FamilyModels:
    config = json.loads((MODELS_DIR / f"{name}.json").read_text(encoding="utf-8-sig"))
    profiles = {key: tuple(_param(item) for item in items) for key, items in config.get("profiles", {}).items()}

    def params(value) -> tuple[Param, ...]:
        # 字符串引用 profiles 里的一组参数，列表则是直接写出的参数。
        return profiles[value] if isinstance(value, str) else tuple(_param(item) for item in value)

    models = tuple(
        ModelSpec(
            label=entry["label"],
            routes={key: Route(route["protocol"], route["model"], dict(route.get("options", {})))
                    for key, route in entry["routes"].items()},
            params=params(entry.get("params", [])),
            max_images=entry.get("max_images"),
            description=entry.get("description", ""),
        )
        for entry in config.get("models", [])
    )
    custom = config.get("custom_model")
    if custom is None:
        return FamilyModels(models)
    return FamilyModels(models, params(custom.get("params", [])), dict(custom.get("route_options", {})))


def _param(entry: dict[str, Any]) -> Param:
    data = dict(entry)
    if "options" in data:
        data["options"] = tuple(data["options"])
    if "default" not in data:
        data["default"] = data["options"][0] if data["kind"] == "combo" else _DEFAULTS.get(data["kind"])
    return Param(**data)
