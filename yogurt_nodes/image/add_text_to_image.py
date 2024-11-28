from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).parents[2] / "resources/fonts"


def wraptext(font: ImageFont.FreeTypeFont, text: str, width: int):
    lines = []
    words = text.split("\n")
    for word in words:
        if font.getbbox(word)[2] <= width:
            lines.append(word)
            word = ""
        while len(word) > 0:
            left = 0
            right = len(word)
            while left < right:
                mid = (left + right) // 2
                if font.getbbox(word[:mid])[2] < width:
                    left = mid + 1
                else:
                    right = mid
            lines.append(word[:left])
            word = word[left:]
    total_width, total_height = 0, 0
    for line in lines:
        bbox = font.getbbox(line)
        total_width = max(total_width, bbox[2])
        total_height += bbox[3]
    return lines, (total_width, total_height)


class AddTextToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to draw."}),
                "location": (["top", "bottom"], {"default": "top", "tooltip": "The location of the text."}),
                "centered": (["false", "true"], {"default": "true", "tooltip": "Center the text."}),
                "font": ([font.name for font in FONT_DIR.iterdir()], {"default": "msyh.ttc", "tooltip": "The font to use."}),
                "font_size": ("INT", {"default": 48, "tooltip": "The font size."}),
                "text_color": ("STRING", {"default": "#00000000", "tooltip": "The color of the text."}),
                "background_color": ("STRING", {"default": "#FFFFFFFF", "tooltip": "The color of the background."}),
                "text": ("STRING", {"default": "", "multiline": True, "tooltip": "The text to add."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "draw_text"

    OUTPUT_NODE = False

    _NODE_NAME = "Add Text To Image"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "Add text to image."

    def draw_text(self, images: torch.Tensor, location="top", centered="true", font="msyh.ttc", font_size=48, text_color="#00000000", background_color="#FFFFFFFF", text=""):
        """
        Add text to image.

        Args:
            images (torch.Tensor): The images to save. (N, H, W, C)
            location (str, optional): location to add, [top or bottom]. Defaults to "top".
            centered (str, optional): put the text to the center. Defaults to "true".
            font (str, optional): font name. Defaults to "msyh.ttc".
            font_size (int, optional): _description_. Defaults to 48.
            text_color (str, optional): _description_. Defaults to "#00000000".
            background_color (str, optional): _description_. Defaults to "#FFFFFFFF".
            text (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """
        device = images.device
        image_w, image_h = images.shape[2], images.shape[1]
        font = ImageFont.truetype(str(FONT_DIR / font), font_size)
        # create a text banner and concatenate it with the image
        lines, (text_width, text_height) = wraptext(font, text, image_w - font_size // 2)
        text_height = text_height + font_size // 4
        text_image = Image.new("RGBA", (image_w, text_height), background_color)
        draw = ImageDraw.Draw(text_image)
        current_height = 0
        for line in lines:
            bbox = font.getbbox(line)
            if centered == "true":
                x = (image_w - bbox[2]) // 2
            else:
                x = 0
            y = current_height
            draw.text((x, y), line, text_color, font=font)
            current_height += bbox[3]
        text_image = (torch.from_numpy(np.array(text_image)).to(images.dtype).to(device) / 255).unsqueeze(0)[..., :images.shape[-1]]
        text_image = text_image.repeat(images.shape[0], 1, 1, 1)
        if location == "top":
            result_images = torch.cat([text_image, images], dim=1)
        elif location == "bottom":
            result_images = torch.cat([images, text_image], dim=1)
        else:
            result_images = images
        return (result_images,)
