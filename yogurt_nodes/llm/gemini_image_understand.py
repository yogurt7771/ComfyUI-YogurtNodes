from typing import List
import torch
import torchvision

from .gemini_client import GeminiClient


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
                        "default": "gemini-2.5-flash",
                        "tooltip": "Gemini model name, default is gemini-2.5-flash",
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
                        "default": 0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Sampling probability threshold, controls output diversity",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "step": 1,
                        "tooltip": "Number of highest probability tokens to consider during sampling",
                    },
                ),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 0,
                        "step": 1,
                        "tooltip": "Maximum number of tokens in the generated text",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 1,
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
                "chat_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "<-system->\n"
                            "{{system_instruction}}\n"
                            "<-/system->\n"
                            "<-user->\n"
                            "{{prompt}}\n"
                            "<-/user->"
                        ),
                        "tooltip": "Content template for the generated text",
                    },
                ),
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "代理URL，格式: protocol://user:pass@addr:port，支持http,https,socks5,socks5h",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "随机种子，设置为-1时随机种子",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "history": ("HISTORY",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING", "HISTORY")
    RETURN_NAMES = ("text", "history")

    FUNCTION = "understand_image"

    _NODE_NAME = "Gemini Image Understand"
    DESCRIPTION = "Understand image content using Gemini API"
    CATEGORY = "YogurtNodes/LLM"

    def understand_image(
        self,
        api_key: str = "",
        model_name: str = "",
        system_prompt: str = "",
        prompt: str = "",
        temperature: float = 1,
        top_p: float = 0,
        top_k: int = 0,
        max_output_tokens: int = 8192,
        retry_count: int = 3,
        disable_safety_settings: bool = False,
        disable_system_prompt: bool = False,
        chat_template: str = "",
        proxy_url: str = "",
        seed: int = -1,
        image: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        image3: torch.Tensor | None = None,
        image4: torch.Tensor | None = None,
        history: List[tuple[str, str]] | None = None,
    ):
        # 收集所有非空图像
        images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if len(img.shape) == 4:
                    img = img[0]
                img = img.permute(2, 0, 1)
                images.append(torchvision.transforms.ToPILImage()(img))

        client = GeminiClient(api_key=api_key, proxy_url=proxy_url)
        text, history = client.generate_text(
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
            chat_template=chat_template,
            history=history,
            seed=seed,
        )
        return (text, history)


class VertexAIImageUnderstand:
    """
    Vertex AI Image Understand
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "credentials": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Credentials JSON for accessing Vertex AI",
                    },
                ),
                "project_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Google Cloud project ID",
                    },
                ),
                "location": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Vertex AI location/region",
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "gemini-2.5-flash",
                        "tooltip": "Gemini model name, default is gemini-2.5-flash",
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
                        "default": 0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Sampling probability threshold, controls output diversity",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "step": 1,
                        "tooltip": "Number of highest probability tokens to consider during sampling",
                    },
                ),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 0,
                        "step": 1,
                        "tooltip": "Maximum number of tokens in the generated text",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 1,
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
                "chat_template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "<-system->\n"
                            "{{system_instruction}}\n"
                            "<-/system->\n"
                            "<-user->\n"
                            "{{prompt}}\n"
                            "<-/user->"
                        ),
                        "tooltip": "Content template for the generated text",
                    },
                ),
                "proxy_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "代理URL，格式: protocol://user:pass@addr:port，支持http,https,socks5,socks5h",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "随机种子，设置为-1时随机种子",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "history": ("HISTORY",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING", "HISTORY")
    RETURN_NAMES = ("text", "history")

    FUNCTION = "understand_image"

    _NODE_NAME = "Gemini Image Understand"
    DESCRIPTION = "Understand image content using Gemini API"
    CATEGORY = "YogurtNodes/LLM"

    def understand_image(
        self,
        credentials: str = "",
        project_id: str = "",
        location: str = "",
        model_name: str = "",
        system_prompt: str = "",
        prompt: str = "",
        temperature: float = 1,
        top_p: float = 0,
        top_k: int = 0,
        max_output_tokens: int = 8192,
        retry_count: int = 3,
        disable_safety_settings: bool = False,
        disable_system_prompt: bool = False,
        chat_template: str = "",
        proxy_url: str = "",
        seed: int = -1,
        image: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        image3: torch.Tensor | None = None,
        image4: torch.Tensor | None = None,
        history: List[tuple[str, str]] | None = None,
    ):
        # 收集所有非空图像
        images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if len(img.shape) == 4:
                    img = img[0]
                img = img.permute(2, 0, 1)
                images.append(torchvision.transforms.ToPILImage()(img))

        client = GeminiClient(
            use_vertex_ai=True,
            vertex_ai_json=credentials,
            vertex_ai_project=project_id,
            vertex_ai_region=location,
            proxy_url=proxy_url,
        )
        text, history = client.generate_text(
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
            chat_template=chat_template,
            history=history,
            seed=seed,
        )
        return (text, history)
