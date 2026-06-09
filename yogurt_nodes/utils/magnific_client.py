from __future__ import annotations

import base64
import io
import os
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import requests
from PIL import Image

import comfy.model_management as model_management
from .api_keys import load_api_keys
from .cancellable_http import CancellableHttpClient


DEFAULT_MAGNIFIC_BASE_URL = "https://api.magnific.com"
DEFAULT_HTTP_TIMEOUT = 60
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_RETRY_MAX_DELAY = 30.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAGNIFIC_FAILED_STATUSES = {"FAILED", "ERROR", "CANCELED", "CANCELLED"}


def _timeout_seconds(timeout: int | float) -> float:
    timeout_value = float(timeout or 0)
    return timeout_value if timeout_value > 0 else DEFAULT_HTTP_TIMEOUT


def encode_image_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _add_text_if_set(payload: Dict[str, Any], key: str, value: str):
    value = (value or "").strip()
    if value:
        payload[key] = value


def build_magnific_payload(
    image: Image.Image,
    mode: str = "creative",
    scale_factor: str = "2x",
    optimized_for: str = "standard",
    prompt: str = "",
    creativity: int = 0,
    hdr: int = 0,
    resemblance: int = 0,
    fractality: int = 0,
    engine: str = "automatic",
    sharpen: int = 50,
    smart_grain: int = 7,
    ultra_detail: int = 30,
    filter_nsfw: bool = False,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "image": encode_image_base64_png(image),
        "filter_nsfw": bool(filter_nsfw),
    }
    if mode == "precision":
        payload.update(
            {
                "sharpen": int(sharpen),
                "smart_grain": int(smart_grain),
                "ultra_detail": int(ultra_detail),
            }
        )
    else:
        payload["scale_factor"] = scale_factor
        payload["optimized_for"] = optimized_for
        _add_text_if_set(payload, "prompt", prompt)
        payload["creativity"] = int(creativity)
        payload["hdr"] = int(hdr)
        payload["resemblance"] = int(resemblance)
        payload["fractality"] = int(fractality)
        payload["engine"] = engine

    if extra:
        payload.update(extra)
    return payload


class MagnificClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        proxy_url: str = "",
        timeout: int = DEFAULT_HTTP_TIMEOUT,
        retry_count: int = 3,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    ):
        self._api_keys_cache: Optional[Dict[str, Any]] = None
        self.api_key = self._resolve_api_key(api_key)
        self.base_url = self._resolve_base_url(base_url)
        self.proxy_url = proxy_url
        self.timeout = _timeout_seconds(timeout)
        self.retry_count = max(int(retry_count), 1)
        self.retry_base_delay = max(float(retry_base_delay), 0.0)
        self.retry_max_delay = max(float(retry_max_delay), self.retry_base_delay)
        self.headers = {
            "x-magnific-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        self.http = CancellableHttpClient(proxy_url=self.proxy_url, timeout=self.timeout)

    def _load_api_keys(self) -> Dict[str, Any]:
        if self._api_keys_cache is None:
            try:
                self._api_keys_cache = load_api_keys()
            except Exception:
                self._api_keys_cache = {}
        return self._api_keys_cache

    def _resolve_api_key(self, api_key: str) -> str:
        if api_key:
            return api_key
        api_keys = self._load_api_keys()
        return api_keys.get("magnific", "") or os.getenv("MAGNIFIC_API_KEY", "")

    def _resolve_base_url(self, base_url: str) -> str:
        if base_url:
            return base_url.rstrip("/")
        api_keys = self._load_api_keys()
        base_url = (
            api_keys.get("magnific_base_url")
            or os.getenv("MAGNIFIC_BASE_URL", "")
            or DEFAULT_MAGNIFIC_BASE_URL
        )
        return base_url.rstrip("/")

    @property
    def proxies(self) -> Optional[Dict[str, str]]:
        if not self.proxy_url:
            return None
        return {"http": self.proxy_url, "https": self.proxy_url}

    def _retry_delay(self, attempt: int, response: requests.Response | None = None) -> float:
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after).timestamp()
                    return max(retry_at - time.time(), 0.0)
                except Exception:
                    pass
        return min(self.retry_base_delay * (2**attempt), self.retry_max_delay)

    def _build_error_message(self, response: requests.Response) -> str:
        body = response.text[:1000] if response.text else ""
        return f"Magnific API request failed: HTTP {response.status_code} {body}".strip()

    def _request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] | None = None,
        **kwargs,
    ) -> requests.Response:
        last_exception: Exception | None = None
        for attempt in range(self.retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                response = self.http.request(
                    method,
                    url,
                    headers=headers if headers is not None else self.headers,
                    timeout=self.timeout,
                    proxies=self.proxies,
                    **kwargs,
                )
            except requests.RequestException as exception:
                last_exception = exception
                if attempt < self.retry_count - 1:
                    self.http.sleep(self._retry_delay(attempt))
                    continue
                raise

            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt < self.retry_count - 1
            ):
                self.http.sleep(self._retry_delay(attempt, response=response))
                continue

            if response.status_code >= 400:
                raise RuntimeError(self._build_error_message(response))

            return response

        raise last_exception or RuntimeError("Magnific API request failed")

    def _response_json(self, response: requests.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _task_endpoint(self, mode: str) -> str:
        if mode == "precision":
            return f"{self.base_url}/v1/ai/image-upscaler-precision"
        return f"{self.base_url}/v1/ai/image-upscaler"

    def _task_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = payload.get("data")
        return data if isinstance(data, dict) else payload

    def _wait_for_completion(
        self,
        endpoint: str,
        task_id: str,
        task_timeout: int,
        poll_interval: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max(float(task_timeout), 1.0)
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Magnific task timed out: task_id={task_id}")
            response = self._request("GET", f"{endpoint}/{task_id}")
            payload = self._response_json(response)
            data = self._task_data(payload)
            status = str(data.get("status", "")).upper()
            if status == "COMPLETED":
                return data
            if status in MAGNIFIC_FAILED_STATUSES:
                raise RuntimeError(
                    f"Magnific task ended with status={status}: task_id={task_id}"
                )
            self.http.sleep(max(float(poll_interval), 0.1))

    def _download_image(self, url: str) -> Image.Image:
        response = self._request("GET", url, headers={})
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image.convert("RGB")

    def upscale_image(
        self,
        image: Image.Image,
        mode: str = "creative",
        scale_factor: str = "2x",
        optimized_for: str = "standard",
        prompt: str = "",
        creativity: int = 0,
        hdr: int = 0,
        resemblance: int = 0,
        fractality: int = 0,
        engine: str = "automatic",
        sharpen: int = 50,
        smart_grain: int = 7,
        ultra_detail: int = 30,
        filter_nsfw: bool = False,
        task_timeout: int = 900,
        poll_interval: float = 3.0,
        extra: Dict[str, Any] | None = None,
    ) -> Image.Image:
        mode = "precision" if mode == "precision" else "creative"
        endpoint = self._task_endpoint(mode)
        payload = build_magnific_payload(
            image=image,
            mode=mode,
            scale_factor=scale_factor,
            optimized_for=optimized_for,
            prompt=prompt,
            creativity=creativity,
            hdr=hdr,
            resemblance=resemblance,
            fractality=fractality,
            engine=engine,
            sharpen=sharpen,
            smart_grain=smart_grain,
            ultra_detail=ultra_detail,
            filter_nsfw=filter_nsfw,
            extra=extra,
        )

        response = self._request("POST", endpoint, json=payload)
        response_data = self._task_data(self._response_json(response))
        task_id = str(response_data.get("task_id", "") or "")
        if not task_id:
            raise RuntimeError("Magnific did not return a task_id")

        data = response_data
        if str(data.get("status", "")).upper() != "COMPLETED":
            data = self._wait_for_completion(
                endpoint=endpoint,
                task_id=task_id,
                task_timeout=task_timeout,
                poll_interval=poll_interval,
            )

        generated = data.get("generated") or []
        if not generated:
            raise RuntimeError(f"Magnific completed without generated image: task_id={task_id}")
        return self._download_image(str(generated[0]))
