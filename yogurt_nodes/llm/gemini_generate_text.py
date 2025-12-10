from typing import List

from ..utils import GeminiClient


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
                        "default": 8192,
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

    _NODE_NAME = "Gemini Generate Text"
    DESCRIPTION = "Generate text using Gemini API"
    CATEGORY = "YogurtNodes/LLM"

    def generate_text(
        self,
        api_key: str = "",
        vertex: bool = False,
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
        safety_level: str = "BLOCK_NONE",
        thinking_budget: int = 0,
        chat_template: str = "",
        proxy_url: str = "",
        history: List[tuple[str, str]] | None = None,
        seed: int = -1,
        thinking_level: str = "OFF",
    ):
        client = GeminiClient(api_key=api_key, use_vertex_api_key=vertex, proxy_url=proxy_url)
        text, thought, history = client.generate_text(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            retry_count=retry_count,
            disable_safety_settings=disable_safety_settings,
            disable_system_prompt=disable_system_prompt,
            safety_level=safety_level,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            chat_template=chat_template,
            history=history,
            seed=seed,
        )
        return (text, history, thought)


class VertexAIGenerateText:
    """
    Vertex AI Generate Text
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

    _NODE_NAME = "Vertex AI Generate Text"
    DESCRIPTION = "Generate text using Vertex AI"
    CATEGORY = "YogurtNodes/LLM"

    def generate_text(
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
        safety_level: str = "BLOCK_NONE",
        thinking_budget: int = 0,
        chat_template: str = "",
        proxy_url: str = "",
        history: List[tuple[str, str]] | None = None,
        seed: int = -1,
        thinking_level: str = "OFF",
    ):
        client = GeminiClient(
            use_vertex_ai=True,
            vertex_ai_json=credentials,
            vertex_ai_project=project_id,
            vertex_ai_region=location,
            proxy_url=proxy_url,
        )
        text, thought, history = client.generate_text(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            retry_count=retry_count,
            disable_safety_settings=disable_safety_settings,
            disable_system_prompt=disable_system_prompt,
            safety_level=safety_level,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            chat_template=chat_template,
            history=history,
            seed=seed,
        )
        return (text, history, thought)
