"""Anthropic 模型系列：Claude（文本 / 图像理解），经 OpenRouter 或 OpenAI 兼容中转调用。"""

from __future__ import annotations

from ..framework import FamilySpec, ModelSpec, Route, combo, integer, number, seed
from .common import EXTRA

_NO_SAMPLING_WITH_REASONING = {"drop_sampling_with_reasoning": True}


def _routes(openrouter_id: str, anthropic_id: str) -> dict[str, Route]:
    return {
        "openrouter": Route("openai_chat", f"anthropic/{openrouter_id}", _NO_SAMPLING_WITH_REASONING),
        "openai_compatible": Route("openai_chat", anthropic_id, _NO_SAMPLING_WITH_REASONING),
    }


def _params(efforts, default="off", temperature=True):
    params = [
        combo("reasoning_effort", efforts, default=default,
              tooltip="Extended thinking effort; 'off' disables it where the model allows."),
        integer("max_output_tokens", 32768, min=1024, max=64000, advanced=True,
                tooltip="Includes thinking tokens."),
    ]
    if temperature:
        params.append(number("temperature", 1.0, min=0.0, max=1.0, step=0.01, advanced=True,
                             tooltip="Only used when reasoning is off."))
    return tuple(params)


_ALWAYS = ("low", "medium", "high", "xhigh", "max")
_XHIGH = ("off", "low", "medium", "high", "xhigh", "max")

CLAUDE = FamilySpec(
    task="text",
    default_provider="openrouter",
    max_images=20,
    system_prompt="",
    history=True,
    params=(seed(), EXTRA),
    models=(
        ModelSpec("Claude Opus 5.5", _routes("claude-opus-5.5", "claude-opus-5-5"),
                  _params(_ALWAYS, "high", temperature=False)),
        ModelSpec("Claude Sonnet 5.5", _routes("claude-sonnet-5.5", "claude-sonnet-5-5"),
                  _params(_XHIGH, temperature=False)),
        ModelSpec("Claude Fable 5.1", _routes("claude-fable-5.1", "claude-fable-5-1"),
                  _params(_ALWAYS, "high", temperature=False)),
        ModelSpec("Claude Opus 5", _routes("claude-opus-5", "claude-opus-5"), _params(_ALWAYS, "high", temperature=False)),
        ModelSpec("Claude Sonnet 5", _routes("claude-sonnet-5", "claude-sonnet-5"), _params(_XHIGH, temperature=False)),
        ModelSpec("Claude Fable 5", _routes("claude-fable-5", "claude-fable-5"), _params(_ALWAYS, "high", temperature=False)),
        ModelSpec("Claude Opus 4.8", _routes("claude-opus-4.8", "claude-opus-4-8"), _params(_XHIGH, temperature=False)),
        ModelSpec("Claude Opus 4.7", _routes("claude-opus-4.7", "claude-opus-4-7"), _params(_XHIGH, temperature=False)),
        ModelSpec("Claude Opus 4.6", _routes("claude-opus-4.6", "claude-opus-4-6"),
                  _params(("off", "low", "medium", "high", "max"))),
        ModelSpec("Claude Sonnet 4.6", _routes("claude-sonnet-4.6", "claude-sonnet-4-6"),
                  _params(("off", "low", "medium", "high", "max"))),
        ModelSpec("Claude Sonnet 4.5", _routes("claude-sonnet-4.5", "claude-sonnet-4-5-20250929"),
                  _params(("off", "low", "medium", "high"))),
        ModelSpec("Claude Haiku 4.5", _routes("claude-haiku-4.5", "claude-haiku-4-5-20251001"),
                  (integer("max_output_tokens", 32768, min=1024, max=64000, advanced=True),
                   number("temperature", 1.0, min=0.0, max=1.0, step=0.01, advanced=True))),
    ),
    custom_params=_params(_XHIGH),
)
