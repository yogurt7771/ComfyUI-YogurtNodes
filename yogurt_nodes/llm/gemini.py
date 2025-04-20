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
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model_name": ("STRING", {"default": "gemini-2.0-flash"}),
                "system_prompt": ("STRING", {"multiline": True}),
                "prompt": ("STRING", {"multiline": True}),
                "temperature": (
                    "FLOAT",
                    {"default": 1, "min": 0.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 64, "min": 0, "step": 1}),
                "max_output_tokens": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 8192, "step": 1},
                ),
                "retry_count": ("INT", {"default": 3, "min": 1, "step": 1}),
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
    ):
        if len(api_key) == 0:
            api_key = os.getenv("GEMINI_API_KEY")
            if len(api_key) == 0:
                raise ValueError("API key is not set")

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
        )
        contents = [prompt]
        for _ in range(retry_count):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                text = response.text
                if len(text) > 0:
                    return text
            except Exception as e:
                print(f"Error {model_name} generating text: {e}")
        return ""
