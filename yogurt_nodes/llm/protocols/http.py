"""HTTP 协议共用工具：连接解析（key / base_url）、可中断请求、图片编解码。

ProviderSpec.connection 支持的键：
- base_url：默认地址；base_url_json / base_url_env：api_key.json 键名与环境变量
- key_json / key_env：api_key.json 键名与环境变量（元组，按顺序取第一个非空值）
兼容渠道不声明 key_json / key_env，因此必须在节点里显式填写，避免把官方 key 发给第三方。
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ...utils.api_keys import load_api_keys
from ...utils.cancellable_http import CancellableHttpClient, CancellableResponse
from ..framework import PROVIDERS, ProviderConfig


@dataclass
class Connection:
    name: str
    base_url: str
    api_key: str
    http: CancellableHttpClient

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @property
    def bearer(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


def _first(values) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def connect(provider: ProviderConfig) -> Connection:
    spec = PROVIDERS[provider.kind]
    conn = spec.connection
    stored = load_api_keys() if (conn.get("key_json") or conn.get("base_url_json")) else {}

    api_key = _first(
        [provider.get("api_key", "")]
        + [stored.get(key, "") for key in conn.get("key_json", ())]
        + [os.getenv(name, "") for name in conn.get("key_env", ())]
    )
    if not api_key:
        sources = ", ".join([f"api_key.json '{k}'" for k in conn.get("key_json", ())] + list(conn.get("key_env", ())))
        hint = f" or set {sources}" if sources else ""
        raise ValueError(f"{spec.display_name}: API key is not set. Fill it in the provider node{hint}.")

    base_url = _first(
        [provider.get("base_url", "")]
        + [stored.get(key, "") for key in conn.get("base_url_json", ())]
        + [os.getenv(name, "") for name in conn.get("base_url_env", ())]
        + [conn.get("base_url", "")]
    )
    if not base_url:
        raise ValueError(f"{spec.display_name}: base_url is required.")

    http = CancellableHttpClient(
        proxy_url=provider.get("proxy_url", "") or None,
        timeout=float(provider.get("timeout", 0) or 0),
    )
    return Connection(spec.display_name, base_url.rstrip("/"), api_key, http)


async def request(conn: Connection, method: str, url: str, **kwargs) -> CancellableResponse:
    return await asyncio.to_thread(conn.http.request, method, url, **kwargs)


def ensure_ok(response: CancellableResponse, label: str) -> Any:
    if response.status_code >= 400:
        raise RuntimeError(f"{label} HTTP {response.status_code}: {response.text[:2000]}")
    try:
        return response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} returned non-JSON response: {response.text[:500]}") from exc


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.convert("RGB" if fmt == "JPEG" else image.mode).save(buffer, format=fmt)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def decode_image(encoded: str) -> Image.Image:
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.load()
    return image


async def download_image(conn: Connection, url: str) -> Image.Image:
    if url.startswith("data:"):
        return decode_image(url)
    response = await request(conn, "GET", url)
    if response.status_code >= 400:
        raise RuntimeError(f"Downloading image failed: HTTP {response.status_code}")
    image = Image.open(io.BytesIO(response.content))
    image.load()
    return image


def parse_extra(value: Any) -> dict[str, Any]:
    """节点上的 extra（JSON 字符串）→ dict，用于透传新参数。"""
    if isinstance(value, dict):
        return value
    if not value or not str(value).strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("extra must be a JSON object")
    return parsed


def is_set(value: Any) -> bool:
    return value is not None and value != "" and value != "auto" and value != "default"
