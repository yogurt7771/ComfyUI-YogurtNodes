from typing import Optional
from typing_extensions import List

import torch
import torchvision

from .gemini_client import GeminiClient


def inputs_def():
    return {
        "required": {
            "model_name": (
                "STRING",
                {
                    "default": "gemini-2.5-flash-image-preview",
                    "tooltip": "Gemini model name, default is gemini-2.5-flash-image-preview",
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
            "aspect_ratio": (
                ["auto", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9"],
                {
                    "default": "auto",
                    "tooltip": "Aspect ratio for the generated image",
                },
            ),
            "image_size": (
                ["1k", "2k", "4k"],
                {
                    "default": "2k",
                    "tooltip": "Image size for the generated image",
                }
            )
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


def generate_image(
    client: GeminiClient,
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
    seed: int = -1,
    aspect_ratio: str | None = None,
    image_size: str | None = None,
    image: Optional[torch.Tensor] = None,
    image1: Optional[torch.Tensor] = None,
    image2: Optional[torch.Tensor] = None,
    image3: Optional[torch.Tensor] = None,
    image4: Optional[torch.Tensor] = None,
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
    images, text, history = client.generate_image(
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
        history=history,
        seed=seed,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )
    tensor_imgs = []
    for image in images:  # type: ignore
        tensor_img = torchvision.transforms.ToTensor()(image)
        tensor_img = tensor_img.permute(1, 2, 0).unsqueeze(0)
        tensor_imgs.append(tensor_img)
    if len(tensor_imgs) == 1:
        return (tensor_imgs[0], text, 1, history)
    elif len(tensor_imgs) > 1:
        return (torch.cat(tensor_imgs, dim=0), text, len(tensor_imgs), history)
    else:
        return (torch.zeros(1, 3, 1, 1, dtype=torch.float32), text, 0, history)


class GeminiGenerateImage:
    """
    Gemini Generate Image
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_types = inputs_def()
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
                **input_types["required"],
            },
            "optional": {
                **input_types["optional"],
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "HISTORY")
    RETURN_NAMES = ("image", "text", "num_images", "history")

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
        proxy_url: str = "",
        seed: int = -1,
        aspect_ratio: str = "auto",
        image_size: str = "2k",
        image: Optional[torch.Tensor] = None,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        history: List[tuple[str, str]] | None = None,
    ):
        client = GeminiClient(api_key=api_key, proxy_url=proxy_url)
        return generate_image(
            client=client,
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
            chat_template=chat_template,
            seed=seed,
            aspect_ratio=None if aspect_ratio == "auto" else aspect_ratio,
            image_size=image_size,
            image=image,
            image1=image1,
            image2=image2,
            image3=image3,
            image4=image4,
            history=history,
        )


class VertexAIGenerateImage:
    """
    Vertex AI Generate Image
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
                **inputs_def()["required"],
            },
            "optional": {
                **inputs_def()["optional"],
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "HISTORY")
    RETURN_NAMES = ("image", "text", "num_images", "history")

    FUNCTION = "generate_image"

    _NODE_NAME = "Vertex AI Generate Image"
    DESCRIPTION = (
        "Generate image using Vertex AI API and return as torch.Tensor (h,w,c) and text"
    )
    CATEGORY = "YogurtNodes/LLM"

    def generate_image(
        self,
        credentials: str = "",
        project_id: str = "",
        location: str = "global",
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
        seed: int = -1,
        aspect_ratio: str = "auto",
        image_size: str = "2k",
        image: Optional[torch.Tensor] = None,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        history: List[tuple[str, str]] | None = None,
    ):
        client = GeminiClient(
            use_vertex_ai=True,
            vertex_ai_json=credentials,
            vertex_ai_project=project_id,
            vertex_ai_region=location,
            proxy_url=proxy_url,
        )
        return generate_image(
            client=client,
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
            chat_template=chat_template,
            seed=seed,
            aspect_ratio=None if aspect_ratio == "auto" else aspect_ratio,
            image_size=image_size,
            image=image,
            image1=image1,
            image2=image2,
            image3=image3,
            image4=image4,
            history=history,
        )
