"""各声明文件共用的参数片段。"""

from __future__ import annotations

from ..framework import combo, integer, number, text

ASPECT_RATIOS = ("auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
ASPECT_RATIOS_EXTENDED = ASPECT_RATIOS + ("1:4", "4:1", "1:8", "8:1")


def api_key_field(json_key: str, env: str):
    return text(
        "api_key",
        tooltip=f"Leave empty to use '{json_key}' from api_key.json, then the {env} environment variable.",
    )


def base_url_field(default: str = "", tooltip: str = ""):
    return text("base_url", default=default, tooltip=tooltip or "API base URL; leave empty for the default endpoint.")


def proxy_field():
    return text("proxy_url", tooltip="Proxy URL, e.g. http://127.0.0.1:7890 or socks5://127.0.0.1:1080.")


def timeout_field():
    return integer("timeout", 0, min=0, max=3600, tooltip="Request timeout in seconds; 0 means no limit.")


def aspect_ratio(options=ASPECT_RATIOS, default: str = "auto"):
    return combo(
        "aspect_ratio",
        options,
        default=default,
        tooltip="'auto' follows the reference image, or the model default when there is none.",
    )


def temperature(default: float = 1.0, advanced: bool = True):
    return number("temperature", default, min=0.0, max=2.0, step=0.01, advanced=advanced,
                  tooltip="Sampling randomness.")


def top_p(default: float = 0.95, advanced: bool = True):
    return number("top_p", default, min=0.0, max=1.0, step=0.01, advanced=advanced, tooltip="Nucleus sampling.")


EXTRA = text(
    "extra",
    multiline=True,
    advanced=True,
    tooltip="Optional JSON object merged into the request (for parameters not exposed as inputs).",
)


def count(max_value: int, default: int = 1):
    return integer("n", default, min=1, max=max_value, tooltip="Number of images to generate.")
