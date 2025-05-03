from PIL import Image
from .lib.gemini_client import GeminiClient


class GeminiImageUnderstand:
    """
    使用 Gemini API 理解图像内容
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
                        "tooltip": "API key for accessing Gemini API",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "gemini-pro-vision",
                        "tooltip": "Gemini model name, default is gemini-pro-vision",
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
                        "default": 8192,
                        "min": 1,
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
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "understand_image"

    _NODE_NAME = "Gemini Image Understand"
    DESCRIPTION = "Understand image content using Gemini API"
    CATEGORY = "YogurtNodes/LLM"

    def understand_image(
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
        image: Image.Image = None,
        image1: Image.Image = None,
        image2: Image.Image = None,
        image3: Image.Image = None,
        image4: Image.Image = None,
    ):
        # 收集所有非空图像
        images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if isinstance(img, list) and len(img) > 0:
                    images.append(Image.fromarray(img[0]))
                elif isinstance(img, Image.Image):
                    images.append(img)

        client = GeminiClient(api_key)
        text = client.generate_text(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            images=images,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            retry_count=retry_count,
            disable_safety_settings=disable_safety_settings,
            disable_system_prompt=disable_system_prompt,
        )
        return (text,)
