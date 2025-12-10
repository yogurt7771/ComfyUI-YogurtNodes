import io
import json
import os
import random
import time
from io import BytesIO
from typing import List, Optional

from google import genai
from google.genai import types
from PIL import Image

import comfy.model_management as model_management
from .proxy_utils import SetProxyEnv
from .openai_client import build_messages

thinking_models = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
]


def load_api_keys_from_file(file_path: str) -> dict:
    """从指定的 JSON 文件中加载 API 密钥"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_key_path = os.path.join(current_dir, "api_key.json")
    if os.path.exists(api_key_path):
        with open(api_key_path, "r", encoding="utf-8") as f:
            api_keys = json.load(f)
            return api_keys
    return {}


class GeminiClient:
    """
    Gemini API 客户端封装类
    """

    def __init__(
        self,
        api_key: str = "",
        use_vertex_ai=False,
        vertex_ai_project=None,
        vertex_ai_json=None,
        vertex_ai_region=None,
        proxy_url: str = "",
    ):
        """
        初始化 Gemini 客户端

        Args:
            api_key (str): Gemini API 密钥（可选，优先级最高，如果未设置，则尝试从 api_key.json 文件中读取，如果未设置，则尝试从环境变量中读取）
            proxy_url (str): 代理URL，格式为 protocol://user:pass@addr:port，支持http,https,socks5,socks5h

        API Key 支持三种获取方式，优先级如下：
        1. 直接通过参数 api_key 传入（推荐用于编程调用）
        2. 当前目录下 api_key.json 文件，格式为 {"gemini": "你的API密钥"}
        3. 环境变量 GEMINI_API_KEY

        Proxy 支持三种获取方式，优先级如下：
        1. 直接通过参数 proxy_url 传入
        2. 当前目录下 api_key.json 文件，格式为 {"proxy": "代理URL"}
        3. 环境变量 HTTP_PROXY, HTTPS_PROXY, ALL_PROXY

        如API Key未设置，将抛出异常。
        """
        self.proxy_url = proxy_url

        with SetProxyEnv(self.proxy_url):
            if use_vertex_ai:
                print("Using Vertex AI Gemini API")
                # 创建 Vertex AI Credentials
                from google.oauth2 import service_account
                api_keys = None
                SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
                if vertex_ai_json is None or len(vertex_ai_json) == 0:
                    if api_keys is None:
                        api_keys = load_api_keys_from_file("api_key.json")
                    vertex_ai_json = api_keys.get("vertex_ai_json", "")
                    if len(vertex_ai_json) == 0:
                        print("Warning: Vertex AI service account JSON is not set. Trying to read from environment variable GOOGLE_APPLICATION_CREDENTIALS.")
                        credentials = None
                    else:
                        print(f"Loading Vertex AI credentials from file: {vertex_ai_json}")
                        credentials = service_account.Credentials.from_service_account_file(
                            vertex_ai_json
                        )
                else:
                    credentials = service_account.Credentials.from_service_account_info(
                        json.loads(vertex_ai_json)
                    )
                if credentials is not None:
                    credentials = credentials.with_scopes(SCOPES)
                if vertex_ai_project is None or len(vertex_ai_project) == 0:
                    if api_keys is None:
                        api_keys = load_api_keys_from_file("api_key.json")
                    vertex_ai_project = api_keys.get("vertex_ai_project", "")
                    if len(vertex_ai_project) == 0:
                        print("Warning: Vertex AI project ID is not set. Trying to read from environment variable GOOGLE_CLOUD_PROJECT.")
                        vertex_ai_project = None
                    else:
                        print(f"Using Vertex AI project ID: {vertex_ai_project}")
                if vertex_ai_region is None or len(vertex_ai_region) == 0:
                    if api_keys is None:
                        api_keys = load_api_keys_from_file("api_key.json")
                    vertex_ai_region = api_keys.get("vertex_ai_region", "")
                    if len(vertex_ai_region) == 0:
                        print("Warning: Vertex AI region is not set. Trying to read from environment variable VERTEX_AI_REGION.")
                        vertex_ai_region = None
                    else:
                        print(f"Using Vertex AI region: {vertex_ai_region}")
                http_options = types.HttpOptions()
                http_options.api_version = "v1"
                self.client = genai.Client(http_options=http_options, vertexai=True, credentials=credentials, project=vertex_ai_project, location=vertex_ai_region)
            else:
                print("Using Google Gemini API")
                if len(api_key) == 0:  # 如果 api_key 为空，则尝试从 api_key.json 文件中读取
                    # 读取 api_key.json 文件
                    api_keys = load_api_keys_from_file("api_key.json")
                    api_key = api_keys.get("gemini", "")
                if len(api_key) == 0:  # 如果 api_key 为空，则尝试从环境变量中读取
                    print("Warning: Gemini API key is not set. Trying to read from environment variable GEMINI_API_KEY.")
                self.client = genai.Client(http_options=http_options)

    def _get_safety_settings(
        self, disable_safety_settings: bool, safety_level: str
    ) -> Optional[list]:
        """获取安全设置"""
        if disable_safety_settings:
            return None
        if safety_level in ["BLOCK_NONE"]:
            safety_level_value = types.HarmBlockThreshold.BLOCK_NONE
        elif safety_level in ["BLOCK_ONLY_HIGH"]:
            safety_level_value = types.HarmBlockThreshold.BLOCK_ONLY_HIGH
        elif safety_level in ["BLOCK_MEDIUM_AND_ABOVE"]:
            safety_level_value = types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        elif safety_level in ["BLOCK_LOW_AND_ABOVE"]:
            safety_level_value = types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
        elif safety_level in ["OFF"]:
            safety_level_value = types.HarmBlockThreshold.OFF
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=safety_level_value,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=safety_level_value,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=safety_level_value,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=safety_level_value,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=safety_level_value,
            ),
        ]

    def _get_contents(
        self,
        system_prompt: str,
        prompt: str,
        images: Optional[List[Image.Image]],
        history: List[tuple[str, str]],
        disable_system_prompt: bool = False,
        chat_template: str = "",
    ):
        """获取对话内容"""
        messages, history = build_messages(
            system_prompt=system_prompt,
            prompt=prompt,
            history=history,
            system_role="system" if not disable_system_prompt else "user",
            model_role="model",
            chat_template=chat_template,
        )

        results = []
        system_instruction = None
        added_images = False
        for message in messages:
            if message["role"] == "system":
                system_instruction = [
                    types.Part.from_text(text=part["text"])
                    for part in message["content"]
                    if part["type"] == "text"
                ]
                continue
            parts = []
            parts.extend(
                [
                    types.Part.from_text(text=part["text"])
                    for part in message["content"]
                    if part["type"] == "text"
                ]
            )
            if message["role"] == "user" and not added_images:
                if images:
                    for image in images:
                        img_bytes_io = io.BytesIO()
                        image.save(img_bytes_io, format="JPEG", quality=95)
                        img_bytes = img_bytes_io.getvalue()
                        image_mime_type = "image/jpeg"
                        parts.append(
                            types.Part.from_bytes(
                                data=img_bytes, mime_type=image_mime_type
                            )
                        )
                added_images = True
            results.append(
                types.Content(
                    role=message["role"],
                    parts=parts,
                )
            )

        return system_instruction, results, history

    def _get_thinking_config(
        self, model_name, thinking_budget: int, thinking_level: str = "OFF"
    ) -> Optional[types.ThinkingConfig]:
        """获取思考配置"""
        thinking_config = None
        if model_name in thinking_models:
            if thinking_budget != 0:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=thinking_budget if thinking_budget > 0 else None,
                )
            elif thinking_level == "OFF":
                thinking_config = types.ThinkingConfig(
                    include_thoughts=False  # type: ignore
                )
            elif thinking_level == "AUTO":
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED,
                )
            elif thinking_level == "HIGH":
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=types.ThinkingLevel.HIGH,
                )
            elif thinking_level == "LOW":
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=types.ThinkingLevel.LOW,
                )
        return thinking_config

    def _build_params(
        self,
        model_name: str,
        prompt: str = "",
        system_prompt: str = "",
        images: List[Image.Image] | None = None,
        history: List[tuple[str, str]] | None = None,
        temperature: float = 1,
        top_p: float = 0.95,
        top_k: int = 64,
        max_output_tokens: int = 8192,
        disable_safety_settings: bool = False,
        disable_system_prompt: bool = False,
        safety_level: str = "BLOCK_NONE",
        thinking_budget: int = 0,
        thinking_level: str = "OFF",
        chat_template: str = "",
        is_image_generation: bool = False,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
    ):
        config_params = {
            "temperature": temperature,
            "top_p": top_p if top_p > 0 else None,
            "top_k": top_k if top_k > 0 else None,
            "max_output_tokens": max_output_tokens if max_output_tokens > 0 else None,
        }

        # 针对图像生成模型的特殊配置
        if is_image_generation:
            config_params["response_modalities"] = ["IMAGE", "TEXT"]
        else:
            config_params["response_mime_type"] = "text/plain"

        config = types.GenerateContentConfig(**config_params)

        thinking_config = self._get_thinking_config(model_name, thinking_budget, thinking_level)
        if thinking_config is not None:
            config.thinking_config = thinking_config

        safety_settings = self._get_safety_settings(
            disable_safety_settings, safety_level
        )
        if safety_settings is not None:
            config.safety_settings = safety_settings

        system_instruction, contents, history = self._get_contents(
            system_prompt=system_prompt,
            prompt=prompt,
            images=images,
            history=history if history is not None else [],
            chat_template=chat_template,
            disable_system_prompt=disable_system_prompt,
        )
        if system_instruction is not None:
            config.system_instruction = system_instruction  # type: ignore

        if aspect_ratio is not None or image_size is not None:
            image_config = types.ImageConfig()
            if aspect_ratio is not None:
                image_config.aspect_ratio = aspect_ratio
            if image_size is not None:
                image_config.image_size = image_size
            config.image_config = image_config
        return config, contents, history

    def generate_text(
        self,
        model_name: str = "",
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        history: List[tuple[str, str]] | None = None,
        temperature: float = 1,
        top_p: float = 0,
        top_k: int = 0,
        max_output_tokens: int = 8192,
        retry_count: int = 3,
        disable_safety_settings: bool = False,
        disable_system_prompt: bool = False,
        safety_level: str = "BLOCK_NONE",
        thinking_budget: int = 0,
        thinking_level: str = "OFF",
        chat_template: str = "",
        seed: int = -1,
    ) -> tuple[str, List[tuple[str, str]]]:
        """
        生成文本

        Args:
            model_name (str): 模型名称
            system_prompt (str): 系统提示词
            prompt (str): 用户提示词
            temperature (float): 采样温度
            top_p (float): 采样概率阈值
            top_k (int): 考虑的最高概率标记数
            max_output_tokens (int): 生成文本的最大标记数
            retry_count (int): 重试次数
            disable_safety_settings (bool): 是否禁用安全设置
            disable_system_prompt (bool): 是否禁用系统提示词
            safety_level (str): 安全等级
            thinking_budget (int): 思考预算
            chat_template (str): 聊天模板

        Returns:
            str: 生成的文本
        """
        config, contents, history = self._build_params(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            history=history,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            disable_safety_settings=disable_safety_settings,
            disable_system_prompt=disable_system_prompt,
            safety_level=safety_level,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            chat_template=chat_template,
        )
        # pprint(f"Generating text with model {model_name}...")
        # pprint(contents)
        # pprint(config)
        last_exception = None
        response = None
        with SetProxyEnv(self.proxy_url):
            for _attempt in range(retry_count):
                model_management.throw_exception_if_processing_interrupted()
                try:
                    if seed < 0:
                        seed = random.randint(0, 2**31 - 1)
                    config.seed = seed

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    # print(response)
                    text = response.text
                    text = text.strip() if text else None
                    if text is not None and len(text) > 0:
                        history.append(("assistant", text))
                        return text, history
                    raise ValueError(f"Model {model_name} returned empty text response.")
                except (ValueError, ConnectionError, TimeoutError) as exception:
                    # print(f"Attempt {attempt + 1}/{retry_count} failed for model {model_name}: {exception}")
                    last_exception = exception
                    time.sleep(3)
                except Exception as exception:
                    # print(f"Unexpected error in attempt {attempt + 1}/{retry_count} for model {model_name}: {exception}")
                    last_exception = exception
                    time.sleep(3)

        raise RuntimeError(
            f"Failed to generate text after {retry_count} retries. "
            f"Last error: {last_exception}. Response: {response}"
        )

    def generate_image(
        self,
        model_name: str,
        prompt: str = "",
        system_prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        history: List[tuple[str, str]] | None = None,
        temperature: float = 1,
        top_p: float = 0,
        top_k: int = 0,
        max_output_tokens: int = 8192,
        retry_count: int = 3,
        disable_safety_settings: bool = False,
        disable_system_prompt: bool = False,
        safety_level: str = "BLOCK_NONE",
        thinking_budget: int = 0,
        thinking_level: str = "OFF",
        chat_template: str = "",
        seed: int = -1,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
    ) -> tuple[List[Image.Image], str, List[tuple[str, str]]]:
        """
        生成图片
        返回 (PIL.Image, text)
        """
        config, contents, history = self._build_params(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            history=history,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            disable_safety_settings=disable_safety_settings,
            disable_system_prompt=disable_system_prompt,
            safety_level=safety_level,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            chat_template=chat_template,
            is_image_generation=True,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        )

        images = []
        last_text = ""
        last_exception = None
        # pprint(f"Generating image with model {model_name}...")
        # pprint(contents)
        # pprint(config)
        response = None
        response_logged = False
        with SetProxyEnv(self.proxy_url):
            for _attempt in range(retry_count):
                model_management.throw_exception_if_processing_interrupted()
                try:
                    if seed < 0:
                        seed = random.randint(0, 2**31 - 1)
                    config.seed = seed
                    response = self.client.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    if not response_logged:
                        # print(response)
                        response_logged = True
                    for chunk in response:
                        if (
                            chunk.candidates is None
                            or chunk.candidates[0].content is None
                            or chunk.candidates[0].content.parts is None
                        ):
                            continue
                        if chunk.candidates[0].content.parts[0].inline_data:
                            inline_data = chunk.candidates[0].content.parts[0].inline_data
                            data_buffer = inline_data.data
                            if data_buffer is not None:
                                image = Image.open(BytesIO(data_buffer)).convert("RGB")
                                images.append(image)
                            else:
                                raise ValueError(
                                    f"Model {model_name} returned empty image data."
                                )
                        else:
                            if hasattr(chunk, "text") and chunk.text:
                                last_text += chunk.text
                    history.append(("assistant", last_text))
                    return images, last_text, history
                except (ValueError, ConnectionError, TimeoutError) as exception:
                    # print(f"Attempt {attempt + 1}/{retry_count} failed for model {model_name}: {exception}")
                    last_exception = exception
                    time.sleep(3)
                except Exception as exception:
                    # print(f"Unexpected error in attempt {attempt + 1}/{retry_count} for model {model_name}: {exception}")
                    last_exception = exception
                    time.sleep(3)

        raise RuntimeError(
            f"Failed to generate image after {retry_count} retries. "
            f"Last error: {last_exception}. Response: {response}"
        )
