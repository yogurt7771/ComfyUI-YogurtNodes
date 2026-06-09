from __future__ import annotations

import io
import json
import os
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

import requests
from PIL import Image

import comfy.model_management as model_management
from .api_keys import load_api_keys
from .cancellable_http import CancellableHttpClient


DEFAULT_TOPAZ_BASE_URL = "https://api.topazlabs.com/image/v1"
DEFAULT_HTTP_TIMEOUT = 60
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_RETRY_MAX_DELAY = 30.0
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
TOPAZ_FAILED_STATUSES = {"Failed", "Cancelled"}


def _timeout_seconds(timeout: int | float) -> float:
    timeout_value = float(timeout or 0)
    return timeout_value if timeout_value > 0 else DEFAULT_HTTP_TIMEOUT


def _bool_to_form(value: bool) -> str:
    return "true" if value else "false"


def _tri_state_to_form(value: str) -> Optional[str]:
    value = (value or "").strip().lower()
    if value in {"true", "false"}:
        return value
    return None


def _form_value(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_to_form(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _add_numeric_if_set(data: Dict[str, str], key: str, value: int | float):
    if value >= 0:
        data[key] = _form_value(value)


def _add_text_if_set(data: Dict[str, str], key: str, value: str):
    value = (value or "").strip()
    if value:
        data[key] = value


def encode_image_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_topaz_form_data(
    image: Image.Image,
    model: str,
    scale_factor: float = 2.0,
    output_width: int = 0,
    output_height: int = 0,
    crop_to_fill: bool = False,
    output_format: str = "png",
    face_enhancement: str = "auto",
    face_enhancement_strength: float = -1,
    face_enhancement_creativity: float = -1,
    subject_detection: str = "auto",
    sharpen: float = -1,
    denoise: float = -1,
    fix_compression: float = -1,
    strength: float = -1,
    prompt: str = "",
    autoprompt: str = "auto",
    creativity: int = -1,
    texture: int = -1,
    detail: float = -1,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    data: Dict[str, str] = {}
    _add_text_if_set(data, "model", model)

    scale = float(scale_factor or 0)
    explicit_width = output_width > 0
    explicit_height = output_height > 0
    if explicit_width:
        data["output_width"] = str(int(output_width))
    elif scale > 0 and not explicit_height:
        data["output_width"] = str(max(1, int(round(image.width * scale))))

    if explicit_height:
        data["output_height"] = str(int(output_height))
    elif scale > 0 and not explicit_width:
        data["output_height"] = str(max(1, int(round(image.height * scale))))

    data["crop_to_fill"] = _bool_to_form(crop_to_fill)
    _add_text_if_set(data, "output_format", output_format)

    face_value = _tri_state_to_form(face_enhancement)
    if face_value is not None:
        data["face_enhancement"] = face_value
    _add_numeric_if_set(data, "face_enhancement_strength", face_enhancement_strength)
    _add_numeric_if_set(
        data,
        "face_enhancement_creativity",
        face_enhancement_creativity,
    )

    if subject_detection and subject_detection != "auto":
        data["subject_detection"] = subject_detection

    _add_numeric_if_set(data, "sharpen", sharpen)
    _add_numeric_if_set(data, "denoise", denoise)
    _add_numeric_if_set(data, "fix_compression", fix_compression)
    _add_numeric_if_set(data, "strength", strength)
    _add_text_if_set(data, "prompt", prompt)

    autoprompt_value = _tri_state_to_form(autoprompt)
    if autoprompt_value is not None:
        data["autoprompt"] = autoprompt_value
    _add_numeric_if_set(data, "creativity", creativity)
    _add_numeric_if_set(data, "texture", texture)
    _add_numeric_if_set(data, "detail", detail)

    if extra:
        for key, value in extra.items():
            data[key] = _form_value(value)

    return data


class TopazClient:
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
        self.headers = {"X-API-Key": self.api_key}
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
        return api_keys.get("topaz", "") or os.getenv("TOPAZ_API_KEY", "")

    def _resolve_base_url(self, base_url: str) -> str:
        if base_url:
            return base_url.rstrip("/")
        api_keys = self._load_api_keys()
        base_url = (
            api_keys.get("topaz_base_url")
            or os.getenv("TOPAZ_BASE_URL", "")
            or DEFAULT_TOPAZ_BASE_URL
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
        return f"Topaz API request failed: HTTP {response.status_code} {body}".strip()

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

        raise last_exception or RuntimeError("Topaz API request failed")

    def _response_json(self, response: requests.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _extract_process_id(self, response: requests.Response) -> str:
        process_id = response.headers.get("X-Process-ID", "")
        if process_id:
            return process_id
        payload = self._response_json(response)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        return str(data.get("process_id", "") or "")

    def _wait_for_completion(
        self,
        process_id: str,
        task_timeout: int,
        poll_interval: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max(float(task_timeout), 1.0)
        status_url = f"{self.base_url}/status/{process_id}"

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Topaz task timed out: process_id={process_id}")
            response = self._request("GET", status_url)
            payload = self._response_json(response)
            status = str(payload.get("status", ""))
            if status == "Completed":
                return payload
            if status in TOPAZ_FAILED_STATUSES:
                raise RuntimeError(
                    f"Topaz task ended with status={status}: process_id={process_id}"
                )
            self.http.sleep(max(float(poll_interval), 0.1))

    def _download_result(self, process_id: str) -> Image.Image:
        response = self._request("GET", f"{self.base_url}/download/{process_id}")
        payload = self._response_json(response)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        download_url = str(data.get("download_url", "") or "")
        if not download_url:
            raise RuntimeError(f"Topaz download URL missing: process_id={process_id}")

        image_response = self._request("GET", download_url, headers={})
        image = Image.open(io.BytesIO(image_response.content))
        image.load()
        return image.convert("RGB")

    def upscale_image(
        self,
        image: Image.Image,
        model_type: str = "standard",
        model: str = "Standard V2",
        scale_factor: float = 2.0,
        output_width: int = 0,
        output_height: int = 0,
        crop_to_fill: bool = False,
        output_format: str = "png",
        face_enhancement: str = "auto",
        face_enhancement_strength: float = -1,
        face_enhancement_creativity: float = -1,
        subject_detection: str = "auto",
        sharpen: float = -1,
        denoise: float = -1,
        fix_compression: float = -1,
        strength: float = -1,
        prompt: str = "",
        autoprompt: str = "auto",
        creativity: int = -1,
        texture: int = -1,
        detail: float = -1,
        task_timeout: int = 900,
        poll_interval: float = 3.0,
        extra: Dict[str, Any] | None = None,
    ) -> Image.Image:
        data = build_topaz_form_data(
            image=image,
            model=model,
            scale_factor=scale_factor,
            output_width=output_width,
            output_height=output_height,
            crop_to_fill=crop_to_fill,
            output_format=output_format,
            face_enhancement=face_enhancement,
            face_enhancement_strength=face_enhancement_strength,
            face_enhancement_creativity=face_enhancement_creativity,
            subject_detection=subject_detection,
            sharpen=sharpen,
            denoise=denoise,
            fix_compression=fix_compression,
            strength=strength,
            prompt=prompt,
            autoprompt=autoprompt,
            creativity=creativity,
            texture=texture,
            detail=detail,
            extra=extra,
        )
        endpoint = "/enhance-gen/async" if model_type == "generative" else "/enhance/async"
        files = {"image": ("image.png", encode_image_png(image), "image/png")}
        response = self._request(
            "POST",
            f"{self.base_url}{endpoint}",
            data=data,
            files=files,
        )
        process_id = self._extract_process_id(response)
        if not process_id:
            raise RuntimeError("Topaz did not return a process_id")

        self._wait_for_completion(
            process_id=process_id,
            task_timeout=task_timeout,
            poll_interval=poll_interval,
        )
        return self._download_result(process_id)
