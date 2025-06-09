import base64
import io
import json
import os
import re
from pathlib import Path
from tempfile import template
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from pprint import pprint


def image_to_base64(image: Image.Image) -> str:
    """将PIL Image转换为base64编码"""
    img_bytes_io = io.BytesIO()
    image.save(img_bytes_io, format="JPEG", quality=95)
    img_bytes = img_bytes_io.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def build_messages(
    system_prompt: str = "",
    prompt: str = "",
    images: Optional[List[Image.Image]] = None,
    system_role: str = "system",
    user_role: str = "user",
    model_role: str = "assistant",
) -> List[Dict[str, Any]]:
    """构建消息列表，防止上下文长度超过限制"""
    if system_prompt is None:
        system_prompt = ""
    if prompt is None:
        prompt = ""
    if images is None:
        images = []
    messages = []
    template_path = Path(__file__).parent / "template.txt"
    if template_path.exists():
        template_content = template_path.read_text(encoding="utf-8").strip()
    else:
        template_content = (
            "<-system->\n"
            "{{system_instruction}}\n"
            "<-/system->\n"
            "<-user->\n"
            "{{prompt}}\n"
            "<-/user->"
        )
    role = None
    content_lines = []
    with_user_prompt = False
    added_image = False
    for line in template_content.splitlines():
        stripped_line = line.strip()
        if role is not None:
            if re.match(r"^<-/\w+->$", stripped_line):
                # 遇到结束标记，添加当前内容到消息列表
                message_content = "\n".join(content_lines).strip()
                if message_content:
                    if "{{prompt}}" in message_content:
                        with_user_prompt = True
                    message_content = message_content.replace(
                        "{{system_instruction}}", system_prompt
                    )
                    message_content = message_content.replace("{{prompt}}", prompt)
                    if role == "system":
                        role = system_role
                    elif role == "user":
                        role = user_role
                    elif role == "assistant":
                        role = model_role
                    contents = [
                        {
                            "type": "text",
                            "text": message_content,
                        }
                    ]
                    if with_user_prompt and not added_image:
                        # 添加图像内容
                        if images:
                            for image in images:
                                base64_image = image_to_base64(image)
                                contents.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                })
                            added_image = True
                    messages.append(
                        {
                            "role": role,
                            "content": contents,
                        }
                    )
                content_lines = []
                role = None  # 重置角色
                with_user_prompt = False
            else:
                content_lines.append(line)
        else:
            if re.match(r"^<-\w+->$", stripped_line):
                role = stripped_line[2:-2]

    return messages


class OpenAIClient:
    """
    OpenAI API 客户端封装类
    """

    def __init__(self, api_key: str = "", base_url: str = ""):
        """
        初始化 OpenAI 客户端

        Args:
            api_key (str): OpenAI API 密钥
            base_url (str): API 基础 URL，默认为官方 API

        API Key 支持三种获取方式，优先级如下：
        1. 直接通过参数 api_key 传入（推荐用于编程调用）
        2. 当前目录下 api_key.json 文件，格式为 {"openai": "你的API密钥"}
        3. 环境变量 OPENAI_API_KEY

        如三者均未设置，将抛出异常。
        """
        if len(api_key) == 0:  # 如果 api_key 为空，则尝试从 api_key.json 文件中读取
            current_dir = os.path.dirname(os.path.abspath(__file__))
            api_key_path = os.path.join(current_dir, "api_key.json")
            if os.path.exists(api_key_path):
                with open(api_key_path, "r", encoding="utf-8") as f:
                    api_keys = json.load(f)
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
            current_dir = os.path.dirname(os.path.abspath(__file__))
            api_key_path = os.path.join(current_dir, "api_key.json")
            if os.path.exists(api_key_path):
                with open(api_key_path, "r", encoding="utf-8") as f:
                    api_keys = json.load(f)
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

    def generate_text(
        self,
        model_name: str,
        prompt: str = "",
        system_prompt: Optional[str] = None,
        images: Optional[List[Image.Image]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        retry_count: int = 3,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
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
            frequency_penalty (float): 频率惩罚
            presence_penalty (float): 存在惩罚
            seed (int): 随机种子

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
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        # 添加可选参数
        if seed is not None:
            payload["seed"] = seed

        pprint(f"Generating text with model {model_name}...")
        pprint(f"Base URL: {self.base_url}")
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
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        retry_count: int = 3,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
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
            frequency_penalty (float): 频率惩罚
            presence_penalty (float): 存在惩罚
            seed (int): 随机种子

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
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
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

        except Exception as e:
            print(f"Error parsing usage info: {e}")
            return {}
