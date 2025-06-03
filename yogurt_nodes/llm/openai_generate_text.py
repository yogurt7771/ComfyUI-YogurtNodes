from .openai_client import OpenAIClient


class OpenAIGenerateText:
    """
    OpenAI Generate Text
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 尝试获取动态模型列表，如果失败则使用缓存列表
        try:
            # 创建临时客户端获取列表（使用环境变量或缓存）
            temp_client = OpenAIClient()
            models = temp_client.get_all_models()
        except Exception:
            # 如果API调用失败，使用缓存列表
            models = OpenAIClient.get_cached_models()

        # 确保列表不为空
        if not models:
            models = OpenAIClient.get_cached_models()

        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "API key for accessing OpenAI API",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Base URL for OpenAI API (leave blank for official API)",
                    },
                ),
                "model_name": (
                    models,
                    {
                        "default": "gpt-4o-mini",
                        "tooltip": "OpenAI model name",
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Sampling temperature, higher values produce more random outputs",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Sampling probability threshold, controls output diversity",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                        "tooltip": "Maximum number of tokens in the generated text",
                    },
                ),
                "max_context_length": (
                    "INT",
                    {
                        "default": 128000,
                        "min": 1000,
                        "max": 2147483647,
                        "step": 1000,
                        "tooltip": "Maximum context length for the model",
                    },
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Frequency penalty to reduce repetition",
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Presence penalty to encourage new topics",
                    },
                ),
                "retry_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of retry attempts if the request fails",
                    },
                ),
            },
            "optional": {
                "seed": ("SEED",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "generate_text"

    _NODE_NAME = "OpenAI Generate Text"
    DESCRIPTION = "Generate text using OpenAI API"
    CATEGORY = "YogurtNodes/LLM"

    def generate_text(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        system_prompt: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_context_length: int,
        frequency_penalty: float,
        presence_penalty: float,
        retry_count: int,
        seed: int = -1,
    ):
        client = OpenAIClient(api_key, base_url)

        # 处理 seed 参数
        seed_value = None if seed == -1 else seed

        text = client.generate_text(
            model_name=model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retry_count=retry_count,
            max_context_length=max_context_length,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed_value,
        )
        return (text,)
