import base64
import io
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import requests
from PIL import Image

import comfy.model_management as model_management
from .api_keys import load_api_keys


def image_to_base64(image: Image.Image) -> str:
    """将PIL Image转换为base64编码"""
    img_bytes_io = io.BytesIO()
    image.save(img_bytes_io, format="JPEG", quality=95)
    img_bytes = img_bytes_io.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def add_image_contents(
    images: List[Image.Image], contents: List[Dict[str, Any]]
):
    for image in images:
        base64_image = image_to_base64(image)
        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )


def build_messages(
    system_prompt: str = "",
    prompt: str = "",
    images: Optional[List[Image.Image]] = None,
    history: List[tuple[str, str]] | None = None,
    chat_template: str = "",
    system_role: str = "system",
    user_role: str = "user",
    model_role: str = "assistant",
) -> tuple[List[Dict[str, Any]], List[tuple[str, str]]]:
    """构建消息列表，防止上下文长度超过限制"""
    if images is None:
        images = []
    if history is None:
        history = []
    if len(chat_template) == 0:
        template_path = Path(__file__).parent / "template.txt"
        if template_path.exists():
            template_content = template_path.read_text(
                encoding="utf-8"
            ).strip()
        else:
            template_content = (
                "<-system->\n"
                "{{system_instruction}}\n"
                "<-/system->\n"
                "<-history->\n"
                "{{history}}\n"
                "<-/history->\n"
                "<-user->\n"
                "{{prompt}}\n"
                "<-/user->"
            )
    else:
        template_content = chat_template
    role = None
    content_lines = []
    added_image = False
    messages = []
    with_user_prompt = False
    for line in template_content.splitlines():
        stripped_line = line.strip()
        if role is not None:
            if re.match(rf"^<-/{role}->$", stripped_line):
                # 遇到结束标记，添加当前内容到消息列表
                message_content = "\n".join(content_lines).strip()
                if message_content:
                    message_content = message_content.replace(
                        "{{system_instruction}}", system_prompt
                    )
                    if "{{prompt}}" in message_content:
                        message_content = message_content.replace(
                            "{{prompt}}", prompt
                        )
                        with_user_prompt = True
                    if role == "history":
                        for role, content in history:
                            contents = [
                                {
                                    "type": "text",
                                    "text": content,
                                }
                            ]
                            if role == "user" and not added_image:
                                add_image_contents(images, contents)
                                added_image = True
                            if role == "system":
                                message_role = system_role
                            elif role == "user":
                                message_role = user_role
                            elif role == "assistant":
                                message_role = model_role
                            messages.append(
                                {"role": message_role, "content": contents}
                            )
                    else:
                        if role == "system":
                            message_role = system_role
                        elif role == "user":
                            message_role = user_role
                        elif role == "assistant":
                            message_role = model_role
                        contents = []
                        if len(message_content) > 0:
                            contents.append(
                                {
                                    "type": "text",
                                    "text": message_content,
                                }
                            )
                        if (
                            role == "user"
                            and with_user_prompt
                            and not added_image
                        ):
                            add_image_contents(images, contents)
                            added_image = True
                        if len(contents) > 0:
                            messages.append(
                                {"role": message_role, "content": contents}
                            )
                content_lines = []
                role = None  # 重置角色
            else:
                content_lines.append(line)
        else:
            if re.match(r"^<-\w+->$", stripped_line):
                role = stripped_line[2:-2]
    history.append(("user", prompt))
    return messages, history


class OpenAIClient:
    """
    OpenAI API 客户端封装类
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        proxy_url: str = "",
        timeout: int = 0,
    ):
        """
        初始化 OpenAI 客户端

        Args:
            api_key (str): OpenAI API 密钥
            base_url (str): API 基础 URL，默认为官方 API
            proxy_url (str): 代理URL，格式为 protocol://user:pass@addr:port，
                支持http,https,socks5,socks5h

        API Key 支持三种获取方式，优先级如下：
        1. 直接通过参数 api_key 传入（推荐用于编程调用）
        2. llm目录下 api_key.json 文件，格式为 {"openai": "你的API密钥"}
        3. 环境变量 OPENAI_API_KEY

        Proxy 支持三种获取方式，优先级如下：
        1. 直接通过参数 proxy_url 传入
        2. llm目录下 api_key.json 文件，格式为 {"proxy": "代理URL"}
        3. 环境变量 HTTP_PROXY, HTTPS_PROXY, ALL_PROXY

        如三者均未设置，将抛出异常。
        """
        api_keys = None
        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从 api_key.json 文件中读取
            if api_keys is None:
                api_keys = load_api_keys()
            if "openai" in api_keys:
                api_key = api_keys["openai"]

        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从环境变量中读取
            api_key = os.getenv("OPENAI_API_KEY", "")

        if len(api_key) == 0:
            raise ValueError("OpenAI API key is not set")

        self.api_key = api_key

        # 设置基础 URL
        if len(base_url) == 0:
            # 尝试从 api_key.json 读取
            if api_keys is None:
                api_keys = load_api_keys()
            if "openai_base_url" in api_keys:
                base_url = api_keys["openai_base_url"]

        if len(base_url) == 0:
            # 尝试从环境变量读取
            base_url = os.getenv("OPENAI_BASE_URL", "")

        if len(base_url) == 0:
            # 默认使用官方 API
            base_url = "https://api.openai.com/v1"

        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.proxy_url = proxy_url if proxy_url else None
        self.timeout = timeout

    def generate_text(
        self,
        model_name: str,
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
            frequency_penalty (float): 频率惩罚
            presence_penalty (float): 存在惩罚
            chat_template (str): 聊天模板

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
        if frequency_penalty > 0:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty > 0:
            payload["presence_penalty"] = presence_penalty

        # 合并额外参数
        if extra:
            payload.update(extra)

        # pprint(f"Generating text with model {model_name}...")
        # pprint(f"Base URL: {self.base_url}")
        # pprint(payload)

        last_exception = None
        response = None
        for _attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                payload["seed"] = random.randint(0, 2**31 - 1)
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout if self.timeout > 0 else None,
                    proxies=self.proxies,
                )
                # pprint(response.text)

                if response.status_code == 200:
                    result = response.json()
                    choice0 = result["choices"][0]
                    content = choice0["message"]["content"].strip()
                    if content:
                        history.append(("assistant", content))
                        return content, history, payload
                    raise ValueError("Empty response content from API")
                else:
                    if response is None:
                        status = "None"
                    else:
                        status = response.status_code
                    body = response.text if response is not None else "None"
                    error_msg = f"API request failed with status {status}: {body}"
                    last_exception = Exception(error_msg)

            except (
                ConnectionError,
                TimeoutError,
                requests.RequestException,
            ) as exception:
                last_exception = exception
                time.sleep(3)
            except Exception as exception:
                last_exception = exception
                time.sleep(3)

        raise last_exception or Exception("All retry attempts failed")

    def understand_image(
        self,
        model_name: str,
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
            frequency_penalty (float): 频率惩罚
            presence_penalty (float): 存在惩罚
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
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            chat_template=chat_template,
            extra=extra,
        )

    def get_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=self.timeout if self.timeout > 0 else None,
                proxies=self.proxies,
            )

            if response.status_code == 200:
                return response.json()["data"]
            else:
                return []

        except (ConnectionError, TimeoutError, requests.RequestException):
            # print(f"Network error getting models: {exception}")
            return []
        except Exception:
            # print(f"Unexpected error getting models: {exception}")
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

        except (ConnectionError, TimeoutError, requests.RequestException):
            # print(f"Network error getting models: {exception}")
            return []
        except Exception:
            # print(f"Unexpected error getting models: {exception}")
            return []

    @staticmethod
    def get_cached_models() -> List[str]:
        """获取缓存的模型列表（离线备用）"""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-vision-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "text-davinci-003",
            "text-davinci-002",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "o1-preview",
            "o1-mini",
        ]

    @staticmethod
    def get_vision_models() -> List[str]:
        """获取支持视觉的模型列表"""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-vision-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
        ]

    @staticmethod
    def get_image_models() -> List[str]:
        """获取支持图像生成的模型列表"""
        return [
            "gpt-5",
            "gpt-image-1",
            "dall-e-2",
            "dall-e-3",
        ]

    def generate_image(
        self,
        model_name: str = "gpt-5",
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        size: str = "auto",
        aspect_ratio: str = "auto",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
        response_format: str = "url",
        retry_count: int = 3,
        chat_template: str = "",
        seed: int = -1,
        history: List[tuple[str, str]] | None = None,
        api_type: str = "auto",
        extra: dict | None = None,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """
        使用OpenAI API生成图像

        Args:
            model_name (str): 模型名称，默认为 gpt-5
            prompt (str): 图像生成提示词
            system_prompt (str): 系统提示词（用于修饰prompt）
            images (List[Image.Image]): 输入图像列表（用于参考或编辑）
            size (str): 图像尺寸
            quality (str): 图像质量
            style (str): 图像风格
            n (int): 生成图像数量
            response_format (str): 响应格式
            retry_count (int): 重试次数
            chat_template (str): 聊天模板
            seed (int): 随机种子，-1为随机值
            history (List[tuple[str, str]]): 对话历史
            api_type (str): API类型选择: auto/response/image

        Returns:
            tuple: (图像列表, 响应文本, 对话历史)
        """
        if history is None:
            history = []
        if images is None:
            images = []

        # 处理提示词模板
        final_prompt = prompt
        if chat_template and system_prompt:
            template_content = chat_template.replace(
                "{{system_instruction}}", system_prompt
            )
            template_content = template_content.replace("{{prompt}}", prompt)
            # 移除模板标签，只保留内容
            template_content = re.sub(r"<-\w+->", "", template_content)
            template_content = re.sub(r"<-/\w+->", "", template_content)
            final_prompt = template_content.strip()
        elif system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"

        # 根据api_type参数选择API
        if api_type == "image":
            use_images_api = True
        elif api_type == "response":
            use_images_api = False
        else:  # auto
            # OpenAI Images API: dall-e-* 与 gpt-image-1
            images_api_models = ["dall-e-2", "dall-e-3", "gpt-image-1"]
            use_images_api = model_name in images_api_models

        if use_images_api:
            return self._generate_image_with_images_api(
                model_name=model_name,
                prompt=final_prompt,
                images=images,
                size=size,
                aspect_ratio=aspect_ratio,
                quality=quality,
                style=style,
                n=n,
                response_format=response_format,
                retry_count=retry_count,
                seed=seed,
                history=history,
                extra=extra,
            )
        else:
            return self._generate_image_with_responses_api(
                model_name=model_name,
                prompt=final_prompt,
                images=images,
                size=size,
                aspect_ratio=aspect_ratio,
                retry_count=retry_count,
                seed=seed,
                history=history,
                extra=extra,
            )

    def get_usage_info(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """解析使用信息（token消耗等）"""
        try:
            if "usage" in usage_data:
                usage = usage_data["usage"]
                return {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            return {}

        except Exception:
            # print(f"Error parsing usage info: {exception}")
            return {}

    def _generate_image_with_images_api(
        self,
        model_name: str,
        prompt: str,
        images: List[Image.Image],
        size: str,
        aspect_ratio: str,
        quality: str,
        style: str,
        n: int,
        response_format: str,
        retry_count: int,
        seed: int,
        history: List[tuple[str, str]],
        extra: dict | None = None,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """使用 OpenAI Images API 生成/编辑图像（dall-e-*、gpt-image-1）"""

        def _pil_to_upload_file(img: Image.Image, filename: str) -> io.BytesIO:
            """
            将 PIL Image 转为可用于 multipart 上传的类文件对象。
            关键点：需要 name 属性（用于文件名），且 seek(0)。
            """
            buf = io.BytesIO()
            # PNG 更通用，且支持透明背景
            img.save(buf, format="PNG")
            buf.seek(0)
            buf.name = filename  # type: ignore[attr-defined]
            return buf

        # 构建参数字典（不同模型支持的参数不同，避免传入无效字段）
        image_kwargs: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "n": n,
            "response_format": response_format,
        }
        if size != "auto":
            image_kwargs["size"] = size
        if aspect_ratio != "auto":
            image_kwargs["aspect_ratio"] = aspect_ratio
        if quality and quality != "standard" and quality != "auto":
            image_kwargs["quality"] = quality
        if style != "vivid":
            image_kwargs["style"] = style

        upload_files = []
        if images:
            for i, img in enumerate(images):
                upload_files.append(_pil_to_upload_file(img, f"image_{i}.png"))
        if len(upload_files) > 0:
            image_kwargs["image"] = upload_files

        # 合并额外参数
        if extra:
            image_kwargs.update(extra)

        last_exception = None

        json_headers = dict(self.headers)
        if "Authorization" not in json_headers and self.api_key:
            json_headers["Authorization"] = f"Bearer {self.api_key}"

        # multipart 由 httpx 自动设置 boundary，这里去掉固定的 Content-Type
        multipart_headers = {
            k: v for k, v in json_headers.items() if k.lower() != "content-type"
        }

        url = (
            f"{self.base_url}/images/edits"
            if len(upload_files) > 0
            else f"{self.base_url}/images/generations"
        )

        with httpx.Client(
            # proxy=self.proxies,
            timeout=self.timeout if self.timeout > 0 else None,
        ) as client:
            for _attempt in range(retry_count):
                model_management.throw_exception_if_processing_interrupted()
                try:
                    if len(upload_files) > 0:
                        # 编辑 / 变体：采用 multipart/form-data
                        files_to_send = []
                        for buf in upload_files:
                            buf.seek(0)
                            files_to_send.append(
                                (
                                    "image",
                                    (
                                        getattr(buf, "name", "image.png"),
                                        buf,
                                        "image/png",
                                    ),
                                )
                            )

                        data_fields = dict(image_kwargs)
                        data_fields.pop("image", None)
                        # multipart 只接受字符串，转换一下
                        for key, value in list(data_fields.items()):
                            if isinstance(value, (int, float)):
                                data_fields[key] = str(value)
                        response = client.post(
                            url,
                            headers=multipart_headers,
                            data=data_fields,
                            files=files_to_send,
                        )
                        action = "Edited"
                    else:
                        # 纯生成：使用 application/json
                        response = client.post(
                            url, headers=json_headers, json=image_kwargs
                        )
                        action = "Generated"

                    if response.status_code not in (200, 201):
                        raise Exception(
                            f"Images API failed: {response.status_code} {response.text}"
                        )

                    result = response.json()
                    out_images: List[Image.Image] = []
                    revised_prompt = prompt

                    for image_data in result.get("data", []):
                        img = None
                        if (
                            response_format == "url"
                            and isinstance(image_data, dict)
                            and image_data.get("url")
                        ):
                            img_response = client.get(image_data["url"])
                            if img_response.status_code == 200:
                                img = Image.open(io.BytesIO(img_response.content))
                        elif isinstance(image_data, dict) and image_data.get(
                            "b64_json"
                        ):
                            img_bytes = base64.b64decode(image_data["b64_json"])
                            img = Image.open(io.BytesIO(img_bytes))

                        if img is not None:
                            img.load()  # 防止关闭前数据丢失
                            out_images.append(img)

                        if isinstance(image_data, dict) and image_data.get(
                            "revised_prompt"
                        ):
                            revised_prompt = image_data["revised_prompt"]

                    history.append(("user", prompt.split("\n\n")[-1]))
                    history.append(
                        (
                            "assistant",
                            (
                                f"{action} {len(out_images)} image(s). "
                                f"Revised: {revised_prompt}"
                            ),
                        )
                    )

                    return out_images, revised_prompt, history

                except Exception as exception:
                    last_exception = exception
                    time.sleep(3)

        raise last_exception or Exception("All retry attempts failed")

    def _generate_image_with_responses_api(
        self,
        model_name: str,
        prompt: str,
        images: List[Image.Image],
        retry_count: int,
        seed: int,
        history: List[tuple[str, str]],
        extra: dict | None = None,
        **kwargs,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """使用Responses API生成图像（适用于GPT模型）"""
        last_exception = None

        headers = dict(self.headers)
        if "Authorization" not in headers and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/responses"

        with httpx.Client(
            proxy=self.proxy_url,
            timeout=self.timeout if self.timeout > 0 else None,
        ) as client:
            for _attempt in range(retry_count):
                model_management.throw_exception_if_processing_interrupted()
                try:
                    # 构建输入内容
                    if images:
                        content = [{"type": "input_text", "text": prompt}]
                        for img in images:
                            img_base64 = image_to_base64(img)
                            image_url = f"data:image/jpeg;base64,{img_base64}"
                            content.append(
                                {"type": "input_image", "image_url": image_url}
                            )
                        input_data = [{"role": "user", "content": content}]
                    else:
                        input_data = prompt

                    payload = {
                        "model": model_name,
                        "input": input_data,
                        "tools": [{"type": "image_generation"}],
                        "stream": False,
                    }

                    # 合并额外参数
                    if extra:
                        payload.update(extra)

                    response = client.post(url, headers=headers, json=payload)

                    if response.status_code not in (200, 201):
                        raise Exception(
                            f"Responses API failed: {response.status_code} {response.text}"
                        )

                    result = response.json()
                    output = result.get("output", []) or []

                    out_images: List[Image.Image] = []
                    response_text = ""

                    def _try_add_image(data_str: str, out_list=out_images):
                        if not data_str:
                            return
                        try:
                            if data_str.startswith("data:image"):
                                base64_part = data_str.split(",", 1)[-1]
                            else:
                                base64_part = data_str
                            img_bytes = base64.b64decode(base64_part)
                            img = Image.open(io.BytesIO(img_bytes))
                            img.load()
                            out_list.append(img)
                        except Exception as exc:  # noqa: BLE001
                            print(f"Failed to decode image: {exc}")

                    for item in output:
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get("type", "")
                        if item_type in {"image_generation_call", "image"}:
                            _try_add_image(item.get("result", ""))
                        elif item_type == "text":
                            response_text = item.get("result", "") or response_text
                        elif "content" in item and isinstance(
                            item.get("content"), list
                        ):
                            for piece in item["content"]:
                                if not isinstance(piece, dict):
                                    continue
                                p_type = piece.get("type", "")
                                if p_type in {
                                    "output_image",
                                    "image",
                                    "image_generation_call",
                                }:
                                    _try_add_image(
                                        piece.get("image_base64")
                                        or piece.get("result")
                                        or piece.get("image_url", "")
                                    )
                                elif p_type in {"output_text", "text"}:
                                    response_text = (
                                        piece.get("text", "")
                                        or piece.get("result", "")
                                        or response_text
                                    )

                    if not response_text:
                        response_text = (
                            result.get("output_text", "")
                            or f"Generated {len(out_images)} image(s)"
                        )

                    history.append(("user", prompt))
                    history.append(("assistant", response_text))

                    return out_images, response_text, history

                except Exception as exception:
                    last_exception = exception
                    time.sleep(3)

        raise last_exception or Exception("All retry attempts failed")
