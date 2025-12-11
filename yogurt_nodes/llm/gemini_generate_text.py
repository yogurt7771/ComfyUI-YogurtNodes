from typing import List

import torchvision

from ..utils import GeminiClient


class GeminiGenerateTextBase:
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
                        "tooltip": "API key for accessing Gemini API",
                    },
                ),
                "vertex": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use Vertex AI for Gemini API",
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
                        "default": 65535,
                        "min": 0,
                        "max": 2147483647,
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
                        "tooltip": "Whether to disable safety settings, if true, the safety settings will not be set",
                    },
                ),
                "disable_system_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to disable the system prompt, if true, the system prompt will sent as a user prompt",
                    },
                ),
                "safety_level": (
                    "STRING",
                    {
                        "default": "BLOCK_NONE",
                        "values": [
                            "OFF",
                            "BLOCK_NONE",
                            "BLOCK_ONLY_HIGH",
                            "BLOCK_MEDIUM_AND_ABOVE",
                            "BLOCK_LOW_AND_ABOVE",
                        ],
                        "tooltip": "Safety level for the generated text",
                    },
                ),
                "thinking_budget": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Thinking budget for the model, if set to -1, the model will not limit thinking budget, if set to 0, the model will disable thinking",
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
                "thinking_level": (
                    ["OFF", "AUTO", "LOW", "HIGH"],
                    {
                        "default": "OFF",
                        "tooltip": "Thinking level for the model, if thinking budget is not 0, this parameter will be ignored",
                    },
                ),
            },
            "optional": {
                "history": ("HISTORY",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING", "HISTORY", "STRING")
    RETURN_NAMES = ("text", "history", "thought")

    FUNCTION = "generate_text"

    CATEGORY = "YogurtNodes/LLM"

    def create_client(self, /, **kwargs) -> GeminiClient:
        raise NotImplementedError()

    def generate_text(
        self,
        **kwargs,
    ):
        client = self.create_client(**kwargs)
        image = kwargs.get("image", None),
        image1 = kwargs.get("image1", None),
        image2 = kwargs.get("image2", None),
        image3 = kwargs.get("image3", None),
        image4 = kwargs.get("image4", None),
        images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if len(img.shape) == 4:
                    img = img[0]
                img = img.permute(2, 0, 1)
                images.append(torchvision.transforms.ToPILImage()(img))
        text, thought, history = client.generate_text(
            model_name=kwargs.get("model_name", ""),
            system_prompt=kwargs.get("system_prompt", ""),
            prompt=kwargs.get("prompt", ""),
            temperature=kwargs.get("temperature", 1),
            top_p=kwargs.get("top_p", 0),
            top_k=kwargs.get("top_k", 0),
            max_output_tokens=kwargs.get("max_output_tokens", 65535),
            retry_count=kwargs.get("retry_count", 1),
            disable_safety_settings=kwargs.get("disable_safety_settings", False),
            disable_system_prompt=kwargs.get("disable_system_prompt", False),
            safety_level=kwargs.get("safety_level", "BLOCK_NONE"),
            thinking_budget=kwargs.get("thinking_budget", 0),
            thinking_level=kwargs.get("thinking_level", "OFF"),
            chat_template=kwargs.get("chat_template", ""),
            history=kwargs.get("history", ""),
            seed=kwargs.get("seed", -1),
            images=images,
        )
        return (text, history, thought)


class GeminiGenerateText(GeminiGenerateTextBase):
    """
    Gemini Generate Text
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_types = GeminiGenerateTextBase.INPUT_TYPES()
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
                "vertex": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use Vertex AI for Gemini API",
                    },
                ),
                **input_types["required"],
            },
            "optional": {
                **input_types["optional"],
            },
        }

    _NODE_NAME = "Gemini Generate Text"
    DESCRIPTION = "Generate text using Gemini API"

    def create_client(self, /, **kwargs):
        api_key = kwargs.get("api_key", "")
        vertex = kwargs.get("vertex", False)
        proxy_url = kwargs.get("proxy_url", "")
        return GeminiClient(
            api_key=api_key, use_vertex_api_key=vertex, proxy_url=proxy_url
        )


class GeminiImageUnderstand(GeminiGenerateText):
    @classmethod
    def INPUT_TYPES(cls):
        input_types = GeminiGenerateText.INPUT_TYPES()
        return {
            "required": {
                **input_types["required"],
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                **input_types["optional"],
            },
        }

    _NODE_NAME = "Gemini Image Understand"
    DESCRIPTION = "Understand images using Gemini API"


class VertexAIGenerateText(GeminiGenerateTextBase):
    """
    Vertex AI Generate Text
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_types = GeminiGenerateTextBase.INPUT_TYPES()
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
                **input_types["required"],
            },
            "optional": {
                **input_types["optional"],
            },
        }

    _NODE_NAME = "Vertex AI Generate Text"
    DESCRIPTION = "Generate text using Vertex AI"

    def create_client(self, /, **kwargs):
        credentials = kwargs.get("credentials", "")
        project_id = kwargs.get("project_id", "")
        location = kwargs.get("location", "")
        proxy_url = kwargs.get("proxy_url", "")
        return GeminiClient(
            credentials_json=credentials,
            project_id=project_id,
            location=location,
            use_vertex_api_key=True,
            proxy_url=proxy_url,
        )


class VertexAIImageUnderstand(VertexAIGenerateText):
    @classmethod
    def INPUT_TYPES(cls):
        input_types = VertexAIGenerateText.INPUT_TYPES()
        return {
            "required": {
                **input_types["required"],
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                **input_types["optional"],
            },
        }

    _NODE_NAME = "Vertex Image Understand"
    DESCRIPTION = "Understand images using Vertex AI"
