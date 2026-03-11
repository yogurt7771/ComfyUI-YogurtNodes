import base64
import io
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

import comfy.model_management as model_management
from .api_keys import load_api_keys


class GRSAIClient:
    """GRSAI Nano Banana image generation client."""

    DEFAULT_BASE_URL = "https://grsaiapi.com"

    _MODELS = [
        "nano-banana-2",
        "nano-banana-fast",
        "nano-banana",
        "nano-banana-pro",
        "nano-banana-pro-vt",
        "nano-banana-pro-cl",
        "nano-banana-pro-vip",
        "nano-banana-pro-4k-vip",
    ]

    _ASPECT_RATIOS = [
        "auto",
        "1:1",
        "16:9",
        "9:16",
        "4:3",
        "3:4",
        "3:2",
        "2:3",
        "5:4",
        "4:5",
        "21:9",
    ]

    _IMAGE_SIZES = ["1K", "2K", "4K"]

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        proxy_url: str = "",
        timeout: int = 0,
    ):
        api_keys = None
        if not api_key:
            if api_keys is None:
                api_keys = load_api_keys()
            api_key = api_keys.get("grsai", api_keys.get("grsai_api_key", ""))

        if not api_key:
            api_key = os.getenv("GRSAI_API_KEY", "")

        if not api_key:
            raise ValueError("GRSAI API key is not set")

        if not base_url:
            if api_keys is None:
                api_keys = load_api_keys()
            base_url = api_keys.get("grsai_base_url", "")

        if not base_url:
            base_url = os.getenv("GRSAI_BASE_URL", "")

        if not base_url:
            base_url = self.DEFAULT_BASE_URL

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.proxies = (
            {
                "http": proxy_url,
                "https": proxy_url,
            }
            if proxy_url
            else None
        )

    @classmethod
    def get_models(cls) -> List[str]:
        return list(cls._MODELS)

    @classmethod
    def get_aspect_ratios(cls) -> List[str]:
        return list(cls._ASPECT_RATIOS)

    @classmethod
    def get_image_sizes(cls) -> List[str]:
        return list(cls._IMAGE_SIZES)

    @staticmethod
    def _strip_template_tags(text: str) -> str:
        text = re.sub(r"<-\w+->", "", text)
        text = re.sub(r"<-/\w+->", "", text)
        return text.strip()

    def _download_image(self, url: str) -> Image.Image:
        if url.startswith("data:image"):
            encoded = url.split(",", 1)[-1]
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            return image

        response = requests.get(
            url,
            timeout=self.timeout if self.timeout > 0 else None,
            proxies=self.proxies,
        )
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _build_prompt(
        self,
        prompt: str,
        system_prompt: str = "",
        chat_template: str = "",
    ) -> str:
        if chat_template and system_prompt:
            content = chat_template.replace("{{system_instruction}}", system_prompt)
            content = content.replace("{{prompt}}", prompt)
            return self._strip_template_tags(content)
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}".strip()
        return prompt.strip()

    @staticmethod
    def _extract_error_message(body: Any) -> str:
        if isinstance(body, dict):
            msg = body.get("msg") or body.get("message") or body.get("error")
            if isinstance(msg, str) and msg:
                return msg
        return str(body)

    def _submit_task(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/v1/draw/nano-banana",
            headers=self.headers,
            json=payload,
            timeout=self.timeout if self.timeout > 0 else None,
            proxies=self.proxies,
        )
        if response.status_code != 200:
            raise ValueError(
                f"GRSAI submit failed with status {response.status_code}: {response.text}"
            )

        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"Unexpected GRSAI submit response: {body}")
        if body.get("code", 0) != 0:
            raise ValueError(
                f"GRSAI submit failed with code {body.get('code')}: "
                f"{self._extract_error_message(body)}"
            )
        return body

    def _poll_result(
        self,
        task_id: str,
        poll_interval_ms: int,
        max_wait_seconds: int,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max_wait_seconds
        last_status = ""
        last_error = ""

        while time.monotonic() <= deadline:
            model_management.throw_exception_if_processing_interrupted()
            response = requests.post(
                f"{self.base_url}/v1/draw/result",
                headers=self.headers,
                json={"id": task_id},
                timeout=self.timeout if self.timeout > 0 else None,
                proxies=self.proxies,
            )

            if response.status_code != 200:
                raise ValueError(
                    "GRSAI result query failed with status "
                    f"{response.status_code}: {response.text}"
                )

            body = response.json()
            code = body.get("code", 0) if isinstance(body, dict) else 0
            if code not in (0, -22):
                raise ValueError(
                    f"GRSAI result query failed with code {code}: "
                    f"{self._extract_error_message(body)}"
                )

            data = body.get("data", {}) if isinstance(body, dict) else {}
            if not isinstance(data, dict):
                data = {}

            status = str(data.get("status", "")).strip().lower()
            if not status and data.get("results"):
                return data

            if status == "succeeded":
                return data

            if status == "failed":
                failure_reason = data.get("failure_reason") or "unknown error"
                error_detail = data.get("error") or ""
                raise ValueError(
                    f"GRSAI generation failed: {failure_reason}. {error_detail}".strip()
                )

            last_status = status or last_status
            last_error = str(data.get("error", "") or last_error)
            time.sleep(max(poll_interval_ms, 100) / 1000.0)

        raise TimeoutError(
            "GRSAI generation timed out"
            f" after {max_wait_seconds}s. Last status: {last_status or 'unknown'}."
            f" Last error: {last_error or 'none'}."
        )

    def _resolve_result_payload(
        self,
        submit_body: Dict[str, Any],
        poll_interval_ms: int,
        max_wait_seconds: int,
    ) -> Dict[str, Any]:
        if submit_body.get("results"):
            return submit_body

        data = submit_body.get("data", {}) if isinstance(submit_body, dict) else {}
        if not isinstance(data, dict):
            data = {}

        if data.get("results"):
            return data

        task_id = data.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(
                f"GRSAI submit response did not include a task id: {submit_body}"
            )

        return self._poll_result(task_id, poll_interval_ms, max_wait_seconds)

    def _parse_results(
        self,
        result_payload: Dict[str, Any],
    ) -> tuple[List[Image.Image], str]:
        results = result_payload.get("results", [])
        if not isinstance(results, list):
            results = []

        images: List[Image.Image] = []
        texts: List[str] = []

        for item in results:
            if not isinstance(item, dict):
                continue

            content = item.get("content")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())

            url = item.get("url")
            if isinstance(url, str) and url:
                images.append(self._download_image(url))

        if not images:
            raise ValueError(f"GRSAI did not return any images: {result_payload}")

        response_text = "\n".join(texts).strip()
        if not response_text:
            response_text = f"Generated {len(images)} image(s)."

        return images, response_text

    def generate_image(
        self,
        model_name: str = "nano-banana-pro",
        prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        reference_urls: Optional[List[str]] = None,
        aspect_ratio: str = "auto",
        image_size: str = "1K",
        retry_count: int = 1,
        poll_interval_ms: int = 2000,
        max_wait_seconds: int = 300,
        system_prompt: str = "",
        chat_template: str = "",
        history: List[tuple[str, str]] | None = None,
        extra: dict | None = None,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        if history is None:
            history = []
        if images is None:
            images = []
        if reference_urls is None:
            reference_urls = []

        final_prompt = self._build_prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            chat_template=chat_template,
        )

        extra_payload = dict(extra) if isinstance(extra, dict) else {}
        extra_urls = extra_payload.pop("urls", [])
        merged_urls: List[str] = []
        for image in images:
            merged_urls.append(self._image_to_data_url(image))
        for value in reference_urls:
            if isinstance(value, str) and value.strip():
                merged_urls.append(value.strip())
        if isinstance(extra_urls, list):
            for value in extra_urls:
                if isinstance(value, str) and value.strip():
                    merged_urls.append(value.strip())

        payload: Dict[str, Any] = dict(extra_payload)
        payload["model"] = model_name
        payload["prompt"] = final_prompt
        payload["webHook"] = "-1"

        if aspect_ratio != "auto":
            payload["aspectRatio"] = aspect_ratio
        if image_size != "auto":
            payload["imageSize"] = image_size
        if merged_urls:
            payload["urls"] = merged_urls

        last_exception = None
        for attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                submit_body = self._submit_task(payload)
                result_payload = self._resolve_result_payload(
                    submit_body=submit_body,
                    poll_interval_ms=poll_interval_ms,
                    max_wait_seconds=max_wait_seconds,
                )
                output_images, response_text = self._parse_results(result_payload)
                history.append(("user", prompt))
                history.append(("assistant", response_text))
                return output_images, response_text, history
            except Exception as exception:
                last_exception = exception
                if attempt + 1 < retry_count:
                    time.sleep(3)

        raise last_exception or Exception("All retry attempts failed")
