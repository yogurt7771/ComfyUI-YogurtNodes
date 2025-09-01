import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
import comfy.model_management as model_management

from .openai_client import build_messages, image_to_base64


class FreedomGPTClient:
    """
    FreedomGPT API 客户端封装类
    """

    def __init__(self, api_key: str = "", proxy_url: str = ""):
        """
        初始化 FreedomGPT 客户端

        Args:
            api_key (str): FreedomGPT API 密钥
            proxy_url (str): 代理URL，格式为 protocol://user:pass@addr:port，支持http,https,socks5,socks5h

        API Key 支持三种获取方式，优先级如下：
        1. 直接通过参数 api_key 传入（推荐用于编程调用）
        2. 当前目录下 api_key.json 文件，格式为 {"freedomgpt": "你的API密钥"}
        3. 环境变量 FREEDOMGPT_API_KEY

        Proxy 支持三种获取方式，优先级如下：
        1. 直接通过参数 proxy_url 传入
        2. 当前目录下 api_key.json 文件，格式为 {"proxy": "代理URL"}
        3. 环境变量 HTTP_PROXY, HTTPS_PROXY, ALL_PROXY

        如三者均未设置，将抛出异常。
        """
        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从 api_key.json 文件中读取
            current_dir = os.path.dirname(os.path.abspath(__file__))
            api_key_path = os.path.join(current_dir, "api_key.json")
            if os.path.exists(api_key_path):
                with open(api_key_path, "r", encoding="utf-8") as f:
                    api_keys = json.load(f)
                    if "freedomgpt" in api_keys:
                        api_key = api_keys["freedomgpt"]

        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从环境变量中读取
            api_key = os.getenv("FREEDOMGPT_API_KEY", "")

        if len(api_key) == 0:
            raise ValueError("FreedomGPT API key is not set")

        self.api_key = api_key

        # FreedomGPT 固定基础 URL
        self.base_url = "https://chat.freedomgpt.com/api/v1"

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
        top_k: int = 40,
        batch_size: int = 1,
    ) -> tuple[str, List[tuple[str, str]], Any]:
        """
        生成文本 - 扩展支持FreedomGPT特有参数

        Args:
            top_k (int): top-k 采样参数，FreedomGPT特有
            batch_size (int): 批处理大小，FreedomGPT特有
            其他参数同父类，但frequency_penalty和presence_penalty会被忽略
        """
        messages, history = build_messages(
            system_prompt=system_prompt,
            prompt=prompt,
            images=images,
            history=history,
            chat_template=chat_template,
        )

        # 调用父类方法构建基础payload，然后添加FreedomGPT特有参数
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if top_p > 0:
            payload["top_p"] = top_p
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        # FreedomGPT 特有参数
        if top_k > 0:
            payload["top_k"] = top_k
        if batch_size > 0:
            payload["batch_size"] = batch_size
        # 注意：忽略 frequency_penalty 和 presence_penalty，FreedomGPT不支持

        # 重用父类的请求逻辑，只是payload有所不同
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

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    if content:
                        history.append(("assistant", content))
                        return content, history, payload
                    raise ValueError("Empty response content from API")
                else:
                    error_msg = f"API request failed with status {response.status_code if response is not None else 'None'}: {response.text if response is not None else 'None'}"
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
                return []

        except (ConnectionError, TimeoutError, requests.RequestException):
            return []
        except Exception:
            return []

    def get_all_models(self) -> List[str]:
        """获取所有可用模型列表"""
        try:
            models = self.get_models()
            model_ids = []

            for model in models:
                model_id = model.get("model", "")
                if model_id:
                    model_ids.append(model_id)

            return sorted(model_ids)

        except (ConnectionError, TimeoutError, requests.RequestException):
            return []
        except Exception:
            return []

    @staticmethod
    def get_cached_models() -> List[str]:
        """获取缓存的模型列表（离线备用）"""
        return [
            "liberty",
            "llama-2-7b-chat",
            "llama-2-13b-chat",
            "alpaca-7b",
            "alpaca-13b",
            "gpt4all-13b-snoozy",
            "wizard-vicuna-13b-uncensored",
            "orca-mini-3b",
            "orca-mini-7b",
            "orca-mini-13b",
            "replit-code-v1-3b",
            "starcoder-15.5b",
            "falcon-7b-instruct",
            "falcon-40b-instruct",
            "mpt-7b-chat",
            "mpt-30b-chat",
            "nous-hermes-13b",
        ]

    @staticmethod
    def get_vision_models() -> List[str]:
        """获取支持视觉的模型列表"""
        # FreedomGPT 目前主要支持文本模型，视觉支持需要进一步确认
        return [
            "liberty",
        ]

    @staticmethod
    def get_image_models() -> List[str]:
        """获取支持图像生成的模型列表"""
        # FreedomGPT 的图像生成模型需要进一步确认
        return [
            "stable-diffusion-v1-5",
            "stable-diffusion-v2-1",
        ]

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
        top_k: int = 40,
        batch_size: int = 1,
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
            top_k (int): top-k 采样参数
            batch_size (int): 批处理大小

        Returns:
            tuple: (图像理解结果, 对话历史, 请求参数)
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
            top_k=top_k,
            batch_size=batch_size,
        )

    def generate_image(
        self,
        model_name: str = "liberty",
        prompt: str = "",
        number_of_images: int = 1,
        retry_count: int = 3,
        system_prompt: str = "",
        history: List[tuple[str, str]] | None = None,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """
        使用FreedomGPT API生成图像

        Args:
            model_name (str): 模型名称，默认为 liberty
            prompt (str): 图像生成提示词
            number_of_images (int): 生成图像数量
            retry_count (int): 重试次数
            system_prompt (str): 系统提示词（用于修饰prompt）
            history (List[tuple[str, str]]): 对话历史

        Returns:
            tuple: (图像列表, 响应文本, 对话历史)
        """
        if history is None:
            history = []

        # 处理提示词
        final_prompt = prompt
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": model_name,
            "prompt": final_prompt,
            "numberOfImages": number_of_images,
        }

        last_exception = None

        for attempt in range(retry_count):
            model_management.throw_exception_if_processing_interrupted()
            try:
                response = requests.post(
                    f"{self.base_url}/images/generations",
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                    proxies=self.proxies,
                )

                if response.status_code == 200:
                    result = response.json()
                    images = []

                    # 从响应中获取图像URL并下载
                    if "data" in result:
                        for image_data in result["data"]:
                            if "url" in image_data:
                                try:
                                    # 下载图像
                                    img_response = requests.get(
                                        image_data["url"],
                                        timeout=60,
                                        proxies=self.proxies,
                                    )
                                    if img_response.status_code == 200:
                                        img = Image.open(
                                            io.BytesIO(img_response.content)
                                        )
                                        images.append(img)
                                except Exception as e:
                                    print(f"Failed to download image: {e}")
                                    continue

                    response_text = (
                        f"Generated {len(images)} image(s) using model {model_name}"
                    )
                    if "delay" in result:
                        response_text += f" (took {result['delay']}s)"

                    history.append(("user", prompt))
                    history.append(("assistant", response_text))

                    return images, response_text, history
                else:
                    error_msg = f"Image generation failed with status {response.status_code}: {response.text}"
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
