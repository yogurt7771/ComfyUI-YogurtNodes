"""各声明文件共用的参数片段（供应商字段和系列级参数；模型版本的参数在 models/*.json）。"""

from __future__ import annotations

from ..framework import integer, text


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


EXTRA = text(
    "extra",
    multiline=True,
    advanced=True,
    tooltip="Optional JSON object merged into the request (for parameters not exposed as inputs).",
)


def count(max_value: int, default: int = 1):
    return integer("n", default, min=1, max=max_value, tooltip="Number of images to generate.")
