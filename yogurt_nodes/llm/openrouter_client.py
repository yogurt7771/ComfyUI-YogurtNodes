import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from pprint import pprint

from .openai_client import build_messages


class OpenRouterClient:
    """
    OpenRouter API 客户端封装类
    """

    def __init__(self, api_key: str = ""):
        """
        初始化 OpenRouter 客户端

        Args:
            api_key (str): OpenRouter API 密钥

        API Key 支持三种获取方式，优先级如下：
        1. 直接通过参数 api_key 传入（推荐用于编程调用）
        2. 当前目录下 api_key.json 文件，格式为 {"openrouter": "你的API密钥"}
        3. 环境变量 OPENROUTER_API_KEY

        如三者均未设置，将抛出异常。
        """
        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从 api_key.json 文件中读取
            current_dir = os.path.dirname(os.path.abspath(__file__))
            api_key_path = os.path.join(current_dir, "api_key.json")
            if os.path.exists(api_key_path):
                with open(api_key_path, "r", encoding="utf-8") as f:
                    api_keys = json.load(f)
                    if "openrouter" in api_keys:
                        api_key = api_keys["openrouter"]

        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从环境变量中读取
            api_key = os.getenv("OPENROUTER_API_KEY", "")

        if len(api_key) == 0:
            raise ValueError("OpenRouter API key is not set")

        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate_text(
        self,
        model_name: str,
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        retry_count: int = 3,
        provider: Optional[str] = None,
    ) -> str:
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

        Returns:
            str: 生成的文本
        """
        messages = build_messages(
            system_prompt=system_prompt,
            prompt=prompt,
            images=images,
        )

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        # 添加provider参数（基础设施提供商）
        if provider and provider != "auto":
            payload["provider"] = {"allow_fallbacks": False, "order": [provider]}
        pprint(self.headers)
        pprint(payload)
        last_exception = None
        response = None
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )
                print(response)

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    if content:
                        return content
                    raise ValueError("Empty response content from API")
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Attempt {attempt + 1} failed: {error_msg}")
                    last_exception = Exception(error_msg)

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                last_exception = e

        raise last_exception or Exception("All retry attempts failed")

    def understand_image(
        self,
        model_name: str,
        prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        system_prompt: str = "",
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        retry_count: int = 3,
        provider: Optional[str] = None,
    ) -> str:
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

        Returns:
            str: 图像理解结果
        """
        return self.generate_text(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            provider=provider,
        )

    def get_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        try:
            response = requests.get(
                f"{self.base_url}/models", headers=self.headers, timeout=30
            )

            if response.status_code == 200:
                return response.json()["data"]
            else:
                print(
                    f"Error getting models: HTTP {response.status_code}: {response.text}"
                )
                return []

        except Exception as e:
            print(f"Error getting models: {e}")
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
            print(f"Error getting models: {e}")
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

    @staticmethod
    def get_cached_models() -> List[str]:
        """获取缓存的模型列表（离线备用）"""
        return [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.5-haiku",
            "anthropic/claude-3-opus",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
            "google/gemini-2.0-flash",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.2-70b-instruct",
            "meta-llama/llama-3.1-405b-instruct",
            "mistralai/mistral-large",
            "mistralai/mistral-medium",
            "deepseek/deepseek-r1",
            "qwen/qwen-2.5-72b-instruct",
        ]

    def get_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """获取生成信息（费用、tokens等）"""
        try:
            response = requests.get(
                f"{self.base_url}/generation/{generation_id}",
                headers=self.headers,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(
                    f"Error getting generation info: HTTP {response.status_code}: {response.text}"
                )
                return {}

        except Exception as e:
            print(f"Error getting generation info: {e}")
            return {}
