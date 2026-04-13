import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

import comfy.model_management as model_management
from .api_keys import load_api_keys
from .qwen_client import (
    DEFAULT_DASHSCOPE_BASE_URL,
    build_prompt,
    extract_request_prompt,
    image_to_data_url,
    is_retryable_exception,
    is_retryable_status,
    merge_extra_fields,
    RETRY_DELAY_SECONDS,
)


WAN_GENERATION_ENDPOINT = "/services/aigc/multimodal-generation/generation"
WAN_PARAMETER_KEYS = {
    "bbox_list",
    "color_palette",
    "enable_interleave",
    "enable_sequential",
    "n",
    "negative_prompt",
    "seed",
    "size",
    "thinking_mode",
    "watermark",
}


def build_wan_request(
    model_name: str,
    prompt: str,
    system_prompt: str = "",
    images: Optional[List[Image.Image]] = None,
    size: str = "auto",
    n: int = 1,
    negative_prompt: str = "",
    watermark: bool = False,
    seed: int = -1,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    images = images or []
    final_prompt = build_prompt(prompt=prompt, system_prompt=system_prompt)
    if not final_prompt:
        raise ValueError("Wan prompt 不能为空")

    content: List[Dict[str, Any]] = []
    for image in images:
        content.append({"image": image_to_data_url(image)})
    content.append({"text": final_prompt})

    request_body: Dict[str, Any] = {
        "model": model_name,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": {
            "n": n,
            "watermark": watermark,
        },
    }
    if negative_prompt.strip():
        request_body["parameters"]["negative_prompt"] = negative_prompt.strip()
    if size and size != "auto":
        request_body["parameters"]["size"] = size
    if seed >= 0:
        request_body["parameters"]["seed"] = seed

    return merge_extra_fields(request_body, extra, WAN_PARAMETER_KEYS)


def extract_wan_image_urls(result: Dict[str, Any]) -> List[str]:
    output = result.get("output") or {}
    choices = output.get("choices") or []
    if not choices:
        return []

    message = choices[0].get("message") or {}
    content = message.get("content") or []
    urls: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        image_url = item.get("image") or item.get("url")
        if image_url:
            urls.append(image_url)
    return urls


class WanClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        proxy_url: str = "",
        timeout: int = 0,
    ):
        self._api_keys_cache: Optional[Dict[str, Any]] = None
        self.api_key = self._resolve_api_key(api_key)
        self.base_url = self._resolve_base_url(base_url)
        self.proxy_url = proxy_url
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
        api_key = api_keys.get("wan") or api_keys.get("dashscope") or ""
        if not api_key:
            api_key = os.getenv("WAN_API_KEY", "") or os.getenv(
                "DASHSCOPE_API_KEY", ""
            )
        if not api_key:
            raise ValueError("Wan API key 未设置，请填写 api_key 或配置 DASHSCOPE_API_KEY")
        return api_key

    def _resolve_base_url(self, base_url: str) -> str:
        if base_url:
            return base_url.rstrip("/")

        api_keys = self._load_api_keys()
        base_url = (
            api_keys.get("wan_base_url")
            or api_keys.get("dashscope_base_url")
            or os.getenv("WAN_BASE_URL", "")
            or os.getenv("DASHSCOPE_BASE_URL", "")
            or DEFAULT_DASHSCOPE_BASE_URL
        )
        return base_url.rstrip("/")

    @property
    def proxies(self) -> Optional[Dict[str, str]]:
        if not self.proxy_url:
            return None
        return {"http": self.proxy_url, "https": self.proxy_url}

    def _download_image(self, url: str) -> Image.Image:
        response = requests.get(
            url,
            proxies=self.proxies,
            timeout=self.timeout if self.timeout > 0 else None,
        )
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image.convert("RGB")

    def _build_error_message(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            code = payload.get("code", response.status_code)
            message = payload.get("message", response.text)
            request_id = payload.get("request_id", "")
            suffix = f" request_id={request_id}" if request_id else ""
            return f"Wan API 请求失败: {code} - {message}{suffix}"
        except Exception:
            return f"Wan API 请求失败: HTTP {response.status_code} - {response.text}"

    def generate_image(
        self,
        model_name: str,
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        size: str = "auto",
        n: int = 1,
        negative_prompt: str = "",
        watermark: bool = False,
        retry_count: int = 1,
        seed: int = -1,
        history: List[tuple[str, str]] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        if history is None:
            history = []

        fallback_prompt = build_prompt(prompt=prompt, system_prompt=system_prompt)
        request_body = build_wan_request(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            size=size,
            n=n,
            negative_prompt=negative_prompt,
            watermark=watermark,
            seed=seed,
            extra=extra,
        )

        endpoint = f"{self.base_url}{WAN_GENERATION_ENDPOINT}"
        last_exception: Optional[Exception] = None
        attempt_count = max(retry_count, 1)
        for _attempt in range(attempt_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                response = requests.post(
                    endpoint,
                    headers=self.headers,
                    json=request_body,
                    proxies=self.proxies,
                    timeout=self.timeout if self.timeout > 0 else None,
                )
                if response.status_code != 200:
                    error = RuntimeError(self._build_error_message(response))
                    if (
                        _attempt < attempt_count - 1
                        and is_retryable_status(response.status_code)
                    ):
                        last_exception = error
                        time.sleep(RETRY_DELAY_SECONDS)
                        continue
                    raise error

                result = response.json()
                image_urls = extract_wan_image_urls(result)
                if not image_urls:
                    raise ValueError("Wan 未返回任何图像 URL")

                output_images = [self._download_image(url) for url in image_urls]
                response_text = json.dumps(result, ensure_ascii=False, indent=2)
                final_prompt = extract_request_prompt(
                    request_body,
                    fallback_prompt=fallback_prompt,
                )
                request_id = result.get("request_id", "")
                usage = result.get("usage") or {}
                image_count = usage.get("image_count", len(output_images))
                history.append(("user", final_prompt))
                history.append(
                    (
                        "assistant",
                        f"Generated {image_count} image(s) with {model_name}. request_id={request_id}",
                    )
                )
                return output_images, response_text, history
            except Exception as exception:  # noqa: BLE001
                last_exception = exception
                if _attempt < attempt_count - 1 and is_retryable_exception(exception):
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                raise

        raise last_exception or RuntimeError("Wan image generation failed")
