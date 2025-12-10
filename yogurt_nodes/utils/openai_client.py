import base64
import io
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import openai
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


def add_image_contents(images: List[Image.Image], contents: List[Dict[str, Any]]):
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
            template_content = template_path.read_text(encoding="utf-8").strip()
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
                        message_content = message_content.replace("{{prompt}}", prompt)
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
                            messages.append({"role": message_role, "content": contents})
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
                        if role == "user" and with_user_prompt and not added_image:
                            add_image_contents(images, contents)
                            added_image = True
                        if len(contents) > 0:
                            messages.append({"role": message_role, "content": contents})
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

    def __init__(self, api_key: str = "", base_url: str = "", proxy_url: str = ""):
        """
        初始化 OpenAI 客户端

        Args:
            api_key (str): OpenAI API 密钥
            base_url (str): API 基础 URL，默认为官方 API
            proxy_url (str): 代理URL，格式为 protocol://user:pass@addr:port，支持http,https,socks5,socks5h

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
            current_dir = os.path.dirname(os.path.abspath(__file__))
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
        self.proxy_url = proxy_url
        if proxy_url:
            self.proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
        else:
            self.proxies = None

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
                    timeout=120,
                    proxies=self.proxies,
                )
                # pprint(response.text)

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    if content:
                        history.append(("assistant", content))
                        return content, history, payload
                    raise ValueError("Empty response content from API")
                else:
                    error_msg = f"API request failed with status {response.status_code if response is not None else 'None'}: {response.text if response is not None else 'None'}"
                    # pprint(f"Attempt {attempt + 1} failed: {error_msg}")
                    last_exception = Exception(error_msg)

            except (
                ConnectionError,
                TimeoutError,
                requests.RequestException,
            ) as exception:
                # pprint(f"Network error in attempt {attempt + 1}/{retry_count}: {exception}")
                last_exception = exception
                time.sleep(3)
            except Exception as exception:
                # pprint(f"Unexpected error in attempt {attempt + 1}/{retry_count}: {exception} {response.text if response is not None else 'None'}")
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
        )

    def get_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30,
                proxies=self.proxies,
            )

            if response.status_code == 200:
                return response.json()["data"]
            else:
                # pprint(
                #     f"Error getting models: HTTP {response.status_code}: {response.text}"
                # )
                return []

        except (ConnectionError, TimeoutError, requests.RequestException) as exception:
            # print(f"Network error getting models: {exception}")
            return []
        except Exception as exception:
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

        except (ConnectionError, TimeoutError, requests.RequestException) as exception:
            # print(f"Network error getting models: {exception}")
            return []
        except Exception as exception:
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
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
        response_format: str = "url",
        retry_count: int = 3,
        chat_template: str = "",
        seed: int = -1,
        history: List[tuple[str, str]] | None = None,
        api_type: str = "auto",
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
            dalle_models = ["dall-e-2", "dall-e-3"]
            use_images_api = model_name in dalle_models

        if use_images_api:
            return self._generate_image_with_images_api(
                model_name=model_name,
                prompt=final_prompt,
                size=size,
                quality=quality,
                style=style,
                n=n,
                response_format=response_format,
                retry_count=retry_count,
                seed=seed,
                history=history,
            )
        else:
            return self._generate_image_with_responses_api(
                model_name=model_name,
                prompt=final_prompt,
                images=images,
                retry_count=retry_count,
                seed=seed,
                history=history,
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

        except Exception as exception:
            # print(f"Error parsing usage info: {exception}")
            return {}

    def _generate_image_with_images_api(
        self,
        model_name: str,
        prompt: str,
        size: str,
        quality: str,
        style: str,
        n: int,
        response_format: str,
        retry_count: int,
        seed: int,
        history: List[tuple[str, str]],
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """使用传统Images API生成图像（适用于DALL-E模型）"""

        # 构建参数字典
        image_kwargs = {
            "model": model_name,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "n": n,
            "response_format": response_format,
        }

        # dall-e-3 特有参数
        if model_name == "dall-e-3":
            image_kwargs["style"] = style
            image_kwargs["n"] = 1  # dall-e-3 只支持生成1张图

        # 处理seed参数
        current_seed = random.randint(0, 2**31 - 1) if seed == -1 else seed

        last_exception = None

        http_client = httpx.Client(proxy=(self.proxy_url if self.proxy_url else None), timeout=120.0)
        client = openai.Client(
            api_key=self.api_key, base_url=self.base_url, http_client=http_client
        )

        for _attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                response = client.images.generate(**image_kwargs)

                images = []
                revised_prompt = prompt

                for image_data in response.data:
                    if response_format == "url":
                        # 从URL下载图像
                        img_response = requests.get(
                            image_data.url, timeout=60, proxies=self.proxies
                        )
                        if img_response.status_code == 200:
                            img = Image.open(io.BytesIO(img_response.content))
                            images.append(img)
                    else:
                        # 从base64解码图像
                        img_data = base64.b64decode(image_data.b64_json)
                        img = Image.open(io.BytesIO(img_data))
                        images.append(img)

                    # 获取修订后的提示词（dall-e-3特有）
                    if (
                        hasattr(image_data, "revised_prompt")
                        and image_data.revised_prompt
                    ):
                        revised_prompt = image_data.revised_prompt

                history.append(("user", prompt.split("\n\n")[-1]))  # 使用原始prompt
                history.append(
                    (
                        "assistant",
                        f"Generated {len(images)} image(s). Revised prompt: {revised_prompt}",
                    )
                )

                return images, revised_prompt, history

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
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """使用Responses API生成图像（适用于GPT模型）"""

        # 处理seed参数
        current_seed = random.randint(0, 2**31 - 1) if seed == -1 else seed

        last_exception = None

        http_client = httpx.Client(proxy=(self.proxy_url if self.proxy_url else None), timeout=120.0)
        client = openai.Client(
            api_key=self.api_key, base_url=self.base_url, http_client=http_client
        )

        for _attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                # 构建输入内容
                if images:
                    # 有图像输入时使用结构化输入
                    content = [{"type": "input_text", "text": prompt}]

                    # 添加图像输入
                    for img in images:
                        img_base64 = image_to_base64(img)
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{img_base64}",
                            }
                        )

                    input_data = [{"role": "user", "content": content}]
                else:
                    # 仅文本输入时使用简单字符串
                    input_data = prompt

                # 使用Responses API生成图像
                response = client.responses.create(
                    model=model_name,
                    input=input_data,
                    tools=[{"type": "image_generation"}],
                    # 注意：Responses API可能不直接支持seed，但我们记录它
                )

                # 提取图像数据
                image_data = [
                    output.result
                    for output in response.output
                    if output.type == "image_generation_call"
                ]

                images = []
                response_text = ""

                # 处理生成的图像
                for img_base64 in image_data:
                    try:
                        img_data = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_data))
                        images.append(img)
                    except Exception as e:
                        print(f"Failed to decode image: {e}")
                        continue

                # 获取文本输出
                text_outputs = [
                    output.result
                    for output in response.output
                    if hasattr(output, "type") and output.type == "text"
                ]
                if text_outputs:
                    response_text = text_outputs[0]
                elif hasattr(response, "output_text"):
                    response_text = response.output_text
                else:
                    response_text = f"Generated {len(images)} image(s)"

                history.append(("user", prompt))
                history.append(("assistant", response_text))

                return images, response_text, history

            except Exception as exception:
                last_exception = exception
                time.sleep(3)

        raise last_exception or Exception("All retry attempts failed")
