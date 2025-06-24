import io
import json
import os
import time
from io import BytesIO
from typing import List, Optional

from google import genai
from google.genai import types
from PIL import Image
from pprint import pprint

from .openai_client import build_messages

thinking_models = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-06-05",
]


class GeminiClient:
    """
    Gemini API 客户端封装类
    """

    def __init__(self, api_key: str = ""):
        """
        初始化 Gemini 客户端

        Args:
            api_key (str): Gemini API 密钥（可选，优先级最高，如果未设置，则尝试从 api_key.json 文件中读取，如果未设置，则尝试从环境变量中读取）

        API Key 支持三种获取方式，优先级如下：
        1. 直接通过参数 api_key 传入（推荐用于编程调用）
        2. 当前目录下 api_key.json 文件，格式为 {"gemini": "你的API密钥"}
        3. 环境变量 GEMINI_API_KEY

        如三者均未设置，将抛出异常。
        """
        # 判断是否是是google vertex ai
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
            print("Using Google Vertex AI")
            self.client = genai.Client(http_options=types.HttpOptions(api_version="v1"))
        else:
            print("Using Google Gemini API")
            if len(api_key) == 0:  # 如果 api_key 为空，则尝试从 api_key.json 文件中读取
                # 读取 api_key.json 文件
                if os.path.exists("api_key.json"):
                    api_keys = json.load(open("api_key.json", "r", encoding="utf-8"))
                    if "gemini" in api_keys:
                        api_key = api_keys["gemini"]
            if len(api_key) == 0:  # 如果 api_key 为空，则尝试从环境变量中读取
                api_key = os.getenv("GEMINI_API_KEY", "")
            if len(api_key) == 0:
                raise ValueError("API key is not set")
            self.client = genai.Client(api_key=api_key)

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
        self, model_name, thinking_budget: int
    ) -> Optional[types.ThinkingConfig]:
        """获取思考配置"""
        thinking_config = None
        if model_name in thinking_models:
            if thinking_budget != 0:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=thinking_budget if thinking_budget > 0 else None,
                )
            else:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=False  # type: ignore
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
        chat_template: str = "",
    ):
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p if top_p > 0 else None,
            top_k=top_k if top_k > 0 else None,
            max_output_tokens=max_output_tokens if max_output_tokens > 0 else None,
            response_mime_type="text/plain",
        )

        thinking_config = self._get_thinking_config(model_name, thinking_budget)
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
        chat_template: str = "",
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
            chat_template=chat_template,
        )
        pprint(f"Generating image with model {model_name}...")
        pprint(contents)
        pprint(config)
        e = None
        response = None
        for _ in range(retry_count):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                print(response)
                text = response.text
                text = text.strip() if text else None
                if text is not None and len(text) > 0:
                    history.append(("assistant", text))
                    return text, history
                raise ValueError(f"Model {model_name} returned empty text response.")
            except Exception as e_inter:
                print(f"Error {model_name} generating text: {e_inter}")
                e = e_inter
                time.sleep(3)
        raise RuntimeError(
            f"Failed to generate text after {retry_count} retries.\n{e}\n{response}"
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
        chat_template: str = "",
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
            chat_template=chat_template,
        )

        images = []
        last_text = ""
        e = None
        pprint(f"Generating image with model {model_name}...")
        pprint(contents)
        pprint(config)
        response = None
        response_logged = False
        for _ in range(retry_count):
            try:
                response = self.client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                if not response_logged:
                    print(response)
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
            except Exception as e_inner:
                print(f"Error {model_name} generating image: {e_inner}")
                e = e_inner
                time.sleep(3)
        raise RuntimeError(
            f"Failed to generate text after {retry_count} retries.\n{e}\n{response}"
        )
