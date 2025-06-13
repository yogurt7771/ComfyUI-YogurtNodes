from typing import Optional

import torch
import torchvision

from .gemini_client import GeminiClient


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
            },
            "optional": {
                "image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

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
        safety_level: str = "BLOCK_NONE",
        thinking_budget: int = 0,
        chat_template: str = "",
        image: Optional[torch.Tensor] = None,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
    ):
        # 收集所有非空图像
        images = []
        for img in [image, image1, image2, image3, image4]:
            if img is not None:
                if len(img.shape) == 4:
                    img = img[0]
                img = img.permute(2, 0, 1)
                images.append(torchvision.transforms.ToPILImage()(img))
        client = GeminiClient(api_key)
        images, text = client.generate_image(
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            images=images,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            retry_count=retry_count,
            disable_safety_settings=disable_safety_settings,
            disable_system_prompt=disable_system_prompt,
            safety_level=safety_level,
            thinking_budget=thinking_budget,
            chat_template=chat_template,
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
