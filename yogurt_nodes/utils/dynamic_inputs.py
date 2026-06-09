from __future__ import annotations

from typing import Any, Iterable


DYNAMIC_INPUT_COUNT = 32


def make_dynamic_inputs(
    prefix: str,
    input_type: Any,
    *,
    count: int = DYNAMIC_INPUT_COUNT,
    start_index: int = 1,
    tooltip: str = "",
) -> dict[str, tuple[Any, dict[str, str]]]:
    inputs = {}
    for index in range(start_index, start_index + count):
        options = {}
        if tooltip:
            options["tooltip"] = tooltip.format(index=index)
        inputs[f"{prefix}{index}"] = (input_type, options)
    return inputs


def ordered_dynamic_values(
    values: dict[str, Any],
    prefix: str,
    *,
    count: int = DYNAMIC_INPUT_COUNT,
    start_index: int = 1,
) -> Iterable[Any]:
    for index in range(start_index, start_index + count):
        value = values.get(f"{prefix}{index}", None)
        if value is not None:
            yield value
