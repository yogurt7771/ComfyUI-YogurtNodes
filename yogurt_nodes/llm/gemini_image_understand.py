import io
import os
from PIL import Image
from google import genai
from google.genai import types
import torch


class GeminiImageUnderstand:
    """
    Gemini Image Understand
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image"}),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "API key for accessing Gemini API",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "gemini-2.0-flash-exp",
                        "tooltip": "Gemini model name, default is gemini-2.0-flash-exp",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "System-level prompt that affects the overall conversation style",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "Main prompt content input by the user",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": 0.0,
                        "step": 0.01,
                        "tooltip": "Sampling temperature, higher values produce more random outputs",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Sampling probability threshold, controls output diversity",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "step": 1,
                        "tooltip": "Number of highest probability tokens to consider during sampling",
                    },
                ),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Maximum number of tokens in the generated text",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "step": 1,
                        "tooltip": "Number of retries when request fails",
                    },
                ),
                "disable_safety_settings": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to disable safety settings (not recommended)",
                    },
                ),
                "disable_system_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to disable the system prompt",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "understand_image"

    _NODE_NAME = "Gemini Image Understand"
    DESCRIPTION = "Image understanding using Gemini API"
    CATEGORY = "YogurtNodes/LLM"

    def understand_image(
        self,
        image: torch.Tensor,
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

        # 读取图片内容
        img_bytes = None
        image_mime_type = None
        pil_image = Image.fromarray(image.cpu().numpy())
        img_bytes_io = io.BytesIO()
        pil_image.save(img_bytes_io, format="JPEG", quality=95)
        img_bytes = img_bytes_io.getvalue()
        image_mime_type = "image/jpeg"

        # 构造parts
        parts = []
        # 如果不启用system instruction，则把system_prompt也放到parts最前面
        if disable_system_prompt or not system_prompt:
            if system_prompt:
                parts.append(types.Part.from_text(text=system_prompt))
        parts.append(types.Part.from_text(text=prompt))
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=image_mime_type))

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
                print(f"Error {model_name} image understand: {e}")
        return ("",)
