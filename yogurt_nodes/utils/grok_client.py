import base64
import io
import os
import random
import re
from typing import Any, Dict, List, Optional

from PIL import Image

import comfy.model_management as model_management
from .api_keys import load_api_keys
from .cancellable_http import CancellableHttpClient
from .openai_client import build_messages, image_to_base64


class GrokClient:
    """
    xAI Grok API client for image generation and editing.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        proxy_url: str = "",
        timeout: int = 0,
    ):
        api_keys = None
        if len(api_key) == 0:
            if api_keys is None:
                api_keys = load_api_keys()
            api_key = api_keys.get("grok", api_keys.get("xai", ""))

        if len(api_key) == 0:
            api_key = os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", ""))

        if len(api_key) == 0:
            raise ValueError("xAI API key is not set")

        if len(base_url) == 0:
            if api_keys is None:
                api_keys = load_api_keys()
            base_url = api_keys.get(
                "grok_base_url",
                api_keys.get("xai_base_url", ""),
            )

        if len(base_url) == 0:
            base_url = os.getenv(
                "XAI_BASE_URL",
                os.getenv("GROK_BASE_URL", ""),
            )

        if len(base_url) == 0:
            base_url = "https://api.x.ai/v1"

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout
        self.proxy_url = proxy_url
        if proxy_url:
            self.proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
        else:
            self.proxies = None
        self.http = CancellableHttpClient(proxy_url=self.proxy_url, timeout=self.timeout)

    @staticmethod
    def _normalize_image_payload(image: Image.Image) -> Dict[str, str]:
        base64_image = image_to_base64(image)
        return {
            "type": "image_url",
            "url": f"data:image/jpeg;base64,{base64_image}",
        }

    @staticmethod
    def _strip_template_tags(text: str) -> str:
        text = re.sub(r"<-\w+->", "", text)
        text = re.sub(r"<-/\w+->", "", text)
        return text.strip()

    def _build_prompt(
        self,
        system_prompt: str,
        prompt: str,
        chat_template: str,
    ) -> str:
        if chat_template and system_prompt:
            content = chat_template.replace("{{system_instruction}}", system_prompt)
            content = content.replace("{{prompt}}", prompt)
            return self._strip_template_tags(content)
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    def _download_image_from_url(self, url: str) -> Image.Image:
        if url.startswith("data:image"):
            encoded = url.split(",", 1)[-1]
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            return image

        response = self.http.get(
            url,
            timeout=self.timeout if self.timeout > 0 else None,
            proxies=self.proxies,
        )
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.load()
        return image

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"text", "output_text"}:
                    part_text = part.get("text") or part.get("content", "")
                    if isinstance(part_text, str) and part_text:
                        text_parts.append(part_text)
            return "\n".join(text_parts).strip()

        return ""

    @staticmethod
    def _with_image_detail(
        messages: List[Dict[str, Any]],
        image_detail: str,
    ) -> List[Dict[str, Any]]:
        if image_detail == "auto":
            return messages

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "image_url":
                    continue
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    image_url["detail"] = image_detail
                elif "url" in item:
                    item["detail"] = image_detail
        return messages

    def generate_text(
        self,
        model_name: str = "grok-4",
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        history: List[tuple[str, str]] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        retry_count: int = 3,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        chat_template: str = "",
        seed: int = -1,
        image_send_mode: str = "openai",
        image_detail: str = "auto",
        extra: dict | None = None,
    ) -> tuple[str, List[tuple[str, str]], Any]:
        messages, history = build_messages(
            system_prompt=system_prompt,
            prompt=prompt,
            images=images,
            history=history,
            chat_template=chat_template,
            image_send_mode=image_send_mode,
        )
        messages = self._with_image_detail(messages, image_detail)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if top_p > 0:
            payload["top_p"] = top_p
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if frequency_penalty > 0:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty > 0:
            payload["presence_penalty"] = presence_penalty

        if extra:
            payload.update(extra)

        current_seed = seed
        last_exception = None
        response = None

        for attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                if seed >= 0:
                    payload["seed"] = current_seed + attempt
                response = self.http.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout if self.timeout > 0 else None,
                    proxies=self.proxies,
                )
                if response.status_code != 200:
                    raise ValueError(
                        f"xAI request failed with status {response.status_code}: {response.text}"
                    )

                result = response.json()
                choice0 = (result.get("choices") or [{}])[0]
                message = choice0.get("message") or {}
                content = self._extract_text_content(message.get("content", ""))
                if not content:
                    raise ValueError("Empty response content from xAI API")
                history.append(("assistant", content))
                return content, history, payload
            except Exception as exception:
                last_exception = exception
                self.http.sleep(3)

        raise RuntimeError(
            f"Failed to generate text after {retry_count} retries. "
            f"Last error: {last_exception}. Response: {response}"
        )

    def understand_image(
        self,
        model_name: str = "grok-4",
        prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        system_prompt: str = "",
        history: List[tuple[str, str]] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        retry_count: int = 3,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        chat_template: str = "",
        seed: int = -1,
        image_send_mode: str = "openai",
        image_detail: str = "high",
        extra: dict | None = None,
    ) -> tuple[str, List[tuple[str, str]], Any]:
        return self.generate_text(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            chat_template=chat_template,
            seed=seed,
            image_send_mode=image_send_mode,
            image_detail=image_detail,
            extra=extra,
        )

    def generate_image(
        self,
        model_name: str = "grok-imagine-image",
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "auto",
        resolution: str = "auto",
        n: int = 1,
        response_format: str = "url",
        retry_count: int = 3,
        chat_template: str = "",
        seed: int = -1,
        history: List[tuple[str, str]] | None = None,
        extra: dict | None = None,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        if history is None:
            history = []
        if images is None:
            images = []

        if len(images) > 3:
            raise ValueError("xAI image editing supports up to 3 input images")

        final_prompt = self._build_prompt(system_prompt, prompt, chat_template)
        has_input_images = len(images) > 0
        endpoint = (
            f"{self.base_url}/images/edits"
            if has_input_images
            else f"{self.base_url}/images/generations"
        )

        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": final_prompt,
            "response_format": response_format,
        }

        if n > 0:
            payload["n"] = n
        if aspect_ratio != "auto":
            payload["aspect_ratio"] = aspect_ratio
        if resolution != "auto":
            payload["resolution"] = resolution
        if seed >= 0:
            payload["seed"] = seed
        if has_input_images:
            image_payloads = [
                self._normalize_image_payload(image) for image in images
            ]
            if len(image_payloads) == 1:
                payload["image"] = image_payloads[0]
            else:
                payload["images"] = image_payloads

        if extra:
            payload.update(extra)

        current_seed = random.randint(0, 2**31 - 1) if seed < 0 else seed
        last_exception = None
        response = None

        for attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                if seed >= 0:
                    payload["seed"] = current_seed + attempt

                response = self.http.post(
                    endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout if self.timeout > 0 else None,
                    proxies=self.proxies,
                )

                if response.status_code != 200:
                    raise ValueError(
                        f"xAI request failed with status {response.status_code}: {response.text}"
                    )

                result = response.json()
                result_items = result.get("data", [])
                output_images = []

                for item in result_items:
                    if not isinstance(item, dict):
                        continue
                    image = None
                    if item.get("b64_json"):
                        img_bytes = base64.b64decode(item["b64_json"])
                        image = Image.open(io.BytesIO(img_bytes))
                        image.load()
                    elif item.get("url"):
                        image = self._download_image_from_url(item["url"])

                    if image is not None:
                        output_images.append(image)

                if len(output_images) == 0:
                    raise ValueError("No images were returned by xAI API")

                action = "Edited" if has_input_images else "Generated"
                response_text = f"{action} {len(output_images)} image(s)."
                history.append(("user", prompt))
                history.append(("assistant", response_text))
                return output_images, response_text, history

            except Exception as exception:
                last_exception = exception
                self.http.sleep(3)

        raise RuntimeError(
            f"Failed to generate image after {retry_count} retries. "
            f"Last error: {last_exception}. Response: {response}"
        )

    @staticmethod
    def get_text_models() -> List[str]:
        return [
            "grok-4",
            "grok-4-fast-reasoning",
        ]

    @staticmethod
    def get_vision_models() -> List[str]:
        return [
            "grok-4",
            "grok-4-fast-reasoning",
        ]

    @staticmethod
    def get_image_models() -> List[str]:
        return [
            "grok-imagine-image",
        ]
