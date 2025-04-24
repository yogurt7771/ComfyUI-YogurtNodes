import os
from google import genai
from google.genai import types


class GeminiGenerateText:
    """
    Gemini Generate Text
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "用于访问Gemini API的密钥",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "gemini-2.0-flash-exp",
                        "tooltip": "Gemini模型名称，默认使用gemini-2.0-flash-exp",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "tooltip": "系统级提示词，影响整体对话风格"},
                ),
                "prompt": (
                    "STRING",
                    {"multiline": True, "tooltip": "用户输入的主要提示内容"},
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": 0.0,
                        "step": 0.01,
                        "tooltip": "采样温度，越高输出越随机",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "采样概率阈值，控制输出多样性",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "step": 1,
                        "tooltip": "采样时考虑的最高概率词数量",
                    },
                ),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "生成文本的最大Token数",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "step": 1,
                        "tooltip": "请求失败时的重试次数",
                    },
                ),
                "disable_safety_settings": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否关闭安全设置（不建议）"},
                ),
                "disable_system_prompt": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "是否禁用系统提示词"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "generate_text"

    _NODE_NAME = "Gemini Generate Text"
    DESCRIPTION = "Generate text using Gemini API"
    CATEGORY = "YogurtNodes/LLM"

    def generate_text(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_output_tokens: int,
        retry_count: int,
        disable_safety_settings: bool,
        disable_system_prompt: bool,
    ):
        if len(api_key) == 0:
            api_key = os.getenv("GEMINI_API_KEY")
            if len(api_key) == 0:
                raise ValueError("API key is not set")

        client = genai.Client(api_key=api_key)

        # 构造parts
        parts = []
        # 如果不启用system instruction，则把system_prompt也放到parts最前面
        if disable_system_prompt or not system_prompt:
            if system_prompt:
                parts.append(types.Part.from_text(text=system_prompt))
        parts.append(types.Part.from_text(text=prompt))

        contents = [
            types.Content(
                role="user",
                parts=parts,
            )
        ]

        # 构造system_instruction
        system_instruction = None
        if not disable_system_prompt and system_prompt:
            system_instruction = [types.Part.from_text(text=system_prompt)]
        else:
            system_instruction = None

        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            safety_settings=(
                [
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ]
                if not disable_safety_settings
                else None
            ),
            response_mime_type="text/plain",
            system_instruction=system_instruction,
        )

        for _ in range(retry_count):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                text = response.text
                if len(text) > 0:
                    return (text,)
            except Exception as e:
                print(f"Error {model_name} generating text: {e}")
        return ("",)
