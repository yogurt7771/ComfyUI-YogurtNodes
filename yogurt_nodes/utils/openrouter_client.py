import base64
import io
import json
import os
import random
import re
from pprint import pprint
from typing import Any, Dict, List, Optional

from PIL import Image

import comfy.model_management as model_management
from .api_keys import load_api_keys
from .cancellable_http import CancellableHttpClient
from .openai_client import build_messages


def normalize_image_size(image_size: str) -> str | None:
    if not image_size:
        return None
    raw_size = str(image_size).strip()
    if not raw_size or raw_size.lower() == "auto":
        return None
    size = raw_size.upper()
    if size in {"0.5K", "1K", "2K", "4K"}:
        return size
    return raw_size


def uses_openrouter_images_api(model_name: str) -> bool:
    """Return whether a model must use OpenRouter's dedicated Images API."""
    return str(model_name).strip().lower().startswith("openai/gpt-image-")


def image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a lossless data URL for ``input_references``."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_image_prompt(
    prompt: str,
    system_prompt: str = "",
    chat_template: str = "",
) -> str:
    """Flatten the node prompt fields for the prompt-only Images API."""
    if chat_template and system_prompt:
        rendered = chat_template.replace("{{system_instruction}}", system_prompt)
        rendered = rendered.replace("{{prompt}}", prompt)
        rendered = re.sub(r"<-/?\w+->", "", rendered)
        return rendered.strip()
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}".strip()
    return prompt


class OpenRouterClient:
    """
    OpenRouter API 客户端封装类
    """

    def __init__(self, api_key: str = "", proxy_url: str = "", timeout: int = 0):
        """
        初始化 OpenRouter 客户端

        Args:
            api_key (str): OpenRouter API 密钥
            proxy_url (str): 代理URL，格式为 protocol://user:pass@addr:port，支持http,https,socks5,socks5h

        API Key 支持三种获取方式，优先级如下：
        1. 直接通过参数 api_key 传入（推荐用于编程调用）
        2. llm目录下 api_key.json 文件，格式为 {"openrouter": "你的API密钥"}
        3. 环境变量 OPENROUTER_API_KEY

        Proxy 支持三种获取方式，优先级如下：
        1. 直接通过参数 proxy_url 传入
        2. llm目录下 api_key.json 文件，格式为 {"proxy": "代理URL"}
        3. 环境变量 HTTP_PROXY, HTTPS_PROXY, ALL_PROXY

        如三者均未设置，将抛出异常。
        """
        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从 api_key.json 文件中读取
            api_keys = load_api_keys()
            if "openrouter" in api_keys:
                api_key = api_keys["openrouter"]

        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从环境变量中读取
            api_key = os.getenv("OPENROUTER_API_KEY", "")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.proxy_url = proxy_url
        if proxy_url:
            self.proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
        else:
            self.proxies = None
        self.timeout = timeout
        self.http = CancellableHttpClient(proxy_url=self.proxy_url, timeout=self.timeout)

    def generate_text(
        self,
        model_name: str,
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        history: List[tuple[str, str]] | None = None,
        temperature: float = 1.0,
        top_p: float = 0,
        max_tokens: int = 8192,
        retry_count: int = 3,
        provider: Optional[str | List[str]] = None,
        chat_template: str = "",
        seed: int = -1,
        extra: dict | None = None,
    ) -> tuple[str, List[tuple[str, str]], Any]:
        """
        生成文本

        Args:
            model_name (str): 模型名称
            prompt (str): 用户提示词
            system_prompt (str): 系统提示词
            images (List[Image.Image]): 图像列表（用于多模态模型）
            temperature (float): 采样温度
            top_p (float): 采样概率阈值
            max_tokens (int): 生成文本的最大标记数
            retry_count (int): 重试次数
            provider (str): 基础设施提供商（可选，如azure, aws等）
            chat_template (str): 聊天模板
            seed (int): 随机种子，-1为随机值

        Returns:
            str: 生成的文本
        """
        messages, history = build_messages(
            system_prompt=system_prompt,
            prompt=prompt,
            images=images,
            history=history,
            chat_template=chat_template,
        )

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if top_p > 0:
            payload["top_p"] = top_p
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        # 添加provider参数（基础设施提供商）
        if provider and provider != "auto":
            if isinstance(provider, list):
                payload["provider"] = {"order": provider}
            else:
                payload["provider"] = {"allow_fallbacks": False, "order": [provider]}

        # 合并额外参数
        if extra:
            payload.update(extra)

        current_seed = random.randint(0, 2**31 - 1) if seed < 0 else seed

        # pprint(self.headers)
        # pprint(payload)
        last_exception = None
        response = None
        for _attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                payload["seed"] = current_seed + _attempt
                response = self.http.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout if self.timeout > 0 else None,
                    proxies=self.proxies,
                )
                # print(response)

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    if content:
                        history.append(("assistant", content))
                        return content, history, payload
                    raise ValueError("Empty response content from API")
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    # print(f"Attempt {attempt + 1} failed: {error_msg}")
                    last_exception = Exception(error_msg)

            except Exception as e:
                # print(f"Attempt {attempt + 1} failed: {str(e)}")
                last_exception = e
                self.http.sleep(3)
        raise last_exception or Exception("All retry attempts failed")

    def generate_image(
        self,
        model_name: str,
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        history: List[tuple[str, str]] | None = None,
        temperature: float = 1.0,
        top_p: float = 0,
        max_tokens: int = 8192,
        retry_count: int = 3,
        provider: Optional[str | List[str]] = None,
        chat_template: str = "",
        seed: int = -1,
        aspect_ratio: str = "auto",
        image_size: str = "auto",
        return_text: bool = True,
        extra: dict | None = None,
        size: str = "auto",
        quality: str = "auto",
        background: str = "auto",
        output_format: str = "auto",
        output_compression: int = -1,
        n: int = 1,
        moderation: str = "auto",
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """
        使用OpenRouter API生成图像
        
        Args:
            model_name (str): 模型名称
            prompt (str): 图像生成提示词
            system_prompt (str): 系统提示词
            images (List[Image.Image]): 输入图像列表（用于参考或编辑）
            history (List[tuple[str, str]]): 对话历史
            temperature (float): 采样温度
            top_p (float): 采样概率阈值
            max_tokens (int): 生成文本的最大标记数
            retry_count (int): 重试次数
            provider (str): 基础设施提供商（可选）
            chat_template (str): 聊天模板
            seed (int): 随机种子，-1为随机值
            return_text (bool): 是否请求并返回文本

        Returns:
            tuple: (图像列表, 响应文本, 对话历史)
        """
        if history is None:
            history = []
        if images is None:
            images = []

        if uses_openrouter_images_api(model_name):
            return self._generate_image_with_images_api(
                model_name=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                images=images,
                history=history,
                retry_count=retry_count,
                provider=provider,
                chat_template=chat_template,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
                output_compression=output_compression,
                n=n,
                moderation=moderation,
                extra=extra,
            )

        messages, history = build_messages(
            system_prompt=system_prompt,
            prompt=prompt,
            images=images,
            history=history,
            chat_template=chat_template,
        )

        modalities = ["image", "text"] if return_text else ["image"]
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "modalities": modalities,  # 关键：启用图像生成，可选文本输出
        }

        if top_p > 0:
            payload["top_p"] = top_p
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        # 添加provider参数（基础设施提供商）
        if provider and provider != "auto":
            if isinstance(provider, list):
                payload["provider"] = {"order": provider}
            else:
                payload["provider"] = {"allow_fallbacks": False, "order": [provider]}

        # 如果指定了aspect_ratio参数，添加到payload
        if aspect_ratio != "auto":
            payload.setdefault("image_config", {})["aspect_ratio"] = aspect_ratio

        normalized_image_size = normalize_image_size(image_size)
        if normalized_image_size:
            payload.setdefault("image_config", {})["image_size"] = normalized_image_size

        if quality != "auto":
            payload.setdefault("image_config", {})["quality"] = quality
        if background != "auto":
            payload.setdefault("image_config", {})["background"] = background
        if output_format != "auto":
            payload.setdefault("image_config", {})["output_format"] = output_format
        if output_compression >= 0:
            payload.setdefault("image_config", {})[
                "output_compression"
            ] = output_compression

        # 合并额外参数
        if extra:
            payload.update(extra)

        # 处理seed参数
        current_seed = random.randint(0, 2**31 - 1) if seed < 0 else seed

        last_exception = None
        response = None

        for attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                payload["seed"] = current_seed + attempt
                response = self.http.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout if self.timeout > 0 else None,
                    proxies=self.proxies,
                )

                if response.status_code == 200:
                    result = response.json()
                    choice = result["choices"][0]
                    message = choice["message"]

                    # 提取文本内容
                    raw_text_content = (
                        message.get("content", "").strip()
                        if message.get("content")
                        else ""
                    )
                    text_content = raw_text_content if return_text else ""

                    # 提取图像数据（base64格式）
                    generated_images = []

                    # 检查message中的images字段（按照OpenRouter文档格式）
                    if "images" in message and isinstance(message["images"], list):
                        for img_item in message["images"]:
                            if isinstance(img_item, dict) and "image_url" in img_item:
                                img_url = img_item["image_url"]["url"]
                                if isinstance(img_url, str) and img_url.startswith("data:image"):
                                    # 解析data URL格式: data:image/png;base64,xxxxx
                                    try:
                                        _, img_base64 = img_url.split(",", 1)
                                        img_bytes = base64.b64decode(img_base64)
                                        img = Image.open(io.BytesIO(img_bytes))
                                        generated_images.append(img)
                                    except Exception as e:
                                        print(f"Failed to decode image: {e}")
                                        continue

                    # 如果没有找到图像，但有内容，生成响应文本
                    if not generated_images and not text_content:
                        raise ValueError("No images or text content generated")

                    # 更新对话历史
                    if text_content or generated_images:
                        history.append(("assistant", text_content if text_content else f"Generated {len(generated_images)} image(s)"))

                    return generated_images, text_content, history

                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    last_exception = Exception(error_msg)

            except Exception as e:
                last_exception = e
                self.http.sleep(3)

        raise last_exception or Exception("All retry attempts failed")

    def _generate_image_with_images_api(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str,
        images: List[Image.Image],
        history: List[tuple[str, str]],
        retry_count: int,
        provider: Optional[str | List[str]],
        chat_template: str,
        aspect_ratio: str,
        image_size: str,
        size: str,
        quality: str,
        background: str,
        output_format: str,
        output_compression: int,
        n: int,
        moderation: str,
        extra: dict | None,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """Generate images through OpenRouter's dedicated ``/images`` API."""
        if not 1 <= n <= 10:
            raise ValueError("OpenRouter Images API supports n from 1 to 10")
        if not -1 <= output_compression <= 100:
            raise ValueError("output_compression must be -1 or between 0 and 100")
        if len(images) > 16:
            raise ValueError("GPT Image models support at most 16 reference images")

        final_prompt = render_image_prompt(prompt, system_prompt, chat_template)
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": final_prompt,
            "n": n,
        }

        if provider and provider != "auto":
            if isinstance(provider, list):
                payload["provider"] = {"order": provider}
            else:
                payload["provider"] = {
                    "allow_fallbacks": False,
                    "order": [provider],
                }

        effective_size = size if size and size != "auto" else image_size
        normalized_size = normalize_image_size(effective_size)
        if normalized_size:
            payload["size"] = normalized_size
        if aspect_ratio != "auto":
            payload["aspect_ratio"] = aspect_ratio
        if quality != "auto":
            payload["quality"] = quality
        if background != "auto":
            payload["background"] = background
        if output_format != "auto":
            payload["output_format"] = output_format
        if output_compression >= 0:
            payload["output_compression"] = output_compression

        if images:
            payload["input_references"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image)},
                }
                for image in images
            ]

        if moderation != "auto":
            provider_config = payload.setdefault("provider", {})
            provider_options = provider_config.setdefault("options", {})
            provider_options.setdefault("openai", {})[
                "moderation"
            ] = moderation

        if extra:
            payload.update(extra)

        history.append(("user", prompt))
        last_exception = None

        for attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                response = self.http.post(
                    f"{self.base_url}/images",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout if self.timeout > 0 else None,
                    proxies=self.proxies,
                )

                if response.status_code not in (200, 201):
                    raise Exception(
                        "API request failed with status "
                        f"{response.status_code}: {response.text}"
                    )

                result = response.json()
                generated_images: List[Image.Image] = []
                for image_data in result.get("data", []) or []:
                    if not isinstance(image_data, dict):
                        continue
                    encoded = image_data.get("b64_json")
                    if not isinstance(encoded, str) or not encoded:
                        continue
                    if encoded.startswith("data:"):
                        encoded = encoded.split(",", 1)[-1]
                    image_bytes = base64.b64decode(encoded)
                    generated_image = Image.open(io.BytesIO(image_bytes))
                    generated_image.load()
                    generated_images.append(generated_image)

                if not generated_images:
                    raise ValueError("No image data generated by OpenRouter Images API")

                history.append(
                    ("assistant", f"Generated {len(generated_images)} image(s)")
                )
                return generated_images, "", history

            except Exception as exception:
                last_exception = exception
                if attempt + 1 < retry_count:
                    self.http.sleep(3)

        raise last_exception or Exception("All retry attempts failed")

    def understand_image(
        self,
        model_name: str,
        prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        system_prompt: str = "",
        history: List[tuple[str, str]] | None = None,
        temperature: float = 1.0,
        top_p: float = 0,
        max_tokens: int = 8192,
        retry_count: int = 3,
        provider: Optional[str | List[str]] = None,
        chat_template: str = "",
        seed: int = -1,
        extra: dict | None = None,
    ) -> tuple[str, List[tuple[str, str]], Any]:
        """
        理解图像内容

        Args:
            model_name (str): 多模态模型名称
            prompt (str): 用户提示词
            images (List[Image.Image]): 图像列表
            system_prompt (str): 系统提示词
            temperature (float): 采样温度
            top_p (float): 采样概率阈值
            max_tokens (int): 生成文本的最大标记数
            retry_count (int): 重试次数
            provider (str): 基础设施提供商（可选，如azure, aws等）
            chat_template (str): 聊天模板
            extra (dict): 额外参数

        Returns:
            str: 图像理解结果
        """
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
            provider=provider,
            chat_template=chat_template,
            seed=seed,
            extra=extra,
        )

    def get_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        try:
            response = self.http.get(
                f"{self.base_url}/models", headers=self.headers, timeout=self.timeout if self.timeout > 0 else None, proxies=self.proxies
            )

            if response.status_code == 200:
                return response.json()["data"]
            else:
                # print(
                #     f"Error getting models: HTTP {response.status_code}: {response.text}"
                # )
                return []

        except Exception as e:
            # print(f"Error getting models: {e}")
            return []

    def get_all_models(self) -> List[str]:
        """获取所有可用模型列表"""
        try:
            models = self.get_models()
            model_ids = []

            for model in models:
                model_id = model.get("id", "")
                if model_id:
                    model_ids.append(model_id)

            return sorted(model_ids)

        except Exception as e:
            # print(f"Error getting models: {e}")
            return []

    @staticmethod
    def get_infrastructure_providers() -> List[str]:
        """获取基础设施提供商列表（用于provider参数）"""
        return [
            "auto",  # 自动选择
            "azure",
            "aws",
            "google-ai-studio",
            "google-vertex",
            "openai",
            "anthropic",
        ]

    def get_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """获取生成信息（费用、tokens等）"""
        try:
            response = self.http.get(
                f"{self.base_url}/generation/{generation_id}",
                headers=self.headers,
                timeout=self.timeout if self.timeout > 0 else None,
                proxies=self.proxies,
            )

            if response.status_code == 200:
                return response.json()
            else:
                # print(
                #     f"Error getting generation info: HTTP {response.status_code}: {response.text}"
                # )
                return {}

        except Exception as e:
            # print(f"Error getting generation info: {e}")
            return {}
