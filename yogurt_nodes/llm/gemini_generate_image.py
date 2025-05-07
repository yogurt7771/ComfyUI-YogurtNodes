import os
from .gemini_client import GeminiClient
import torchvision
import torch


class GeminiGenerateImage:
    """
    Gemini Generate Image
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
                        "default": "gemini-2.0-flash-exp-image-generation",
                        "tooltip": "Gemini model name, default is gemini-2.0-flash-exp-image-generation",
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
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "text", "num_images")

    FUNCTION = "generate_image"

    _NODE_NAME = "Gemini Generate Image"
    DESCRIPTION = (
        "Generate image using Gemini API and return as torch.Tensor (h,w,c) and text"
    )
    CATEGORY = "YogurtNodes/LLM"

    def generate_image(
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

        client = GeminiClient(api_key)
        images, text = client.generate_image(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            retry_count=retry_count,
            disable_safety_settings=disable_safety_settings,
            disable_system_prompt=disable_system_prompt,
        )
        tensor_imgs = []
        for image in images:
            tensor_img = torchvision.transforms.ToTensor()(image)
            tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
            tensor_imgs.append(tensor_img)
        if len(tensor_imgs) == 1:
            return (tensor_imgs[0], text, 1)
        elif len(tensor_imgs) > 1:
            return (torch.cat(tensor_imgs, dim=0), text, len(tensor_imgs))
        else:
            return (
                torch.zeros(1, 3, 1, 1, dtype=torch.float32),
                text,
                0,
            )
