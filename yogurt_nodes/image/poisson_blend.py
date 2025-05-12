import torch
import numpy as np
import cv2


class PoissonBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE", {"tooltip": "背景图像 (N,H,W,C)"}),
                "foreground": ("IMAGE", {"tooltip": "前景图像 (N,H,W,C)"}),
                "mask": ("IMAGE", {"tooltip": "掩码图像 (N,H,W,C)，单通道或三通道"}),
                "x": ("INT", {"default": 0, "tooltip": "融合左上角X坐标"}),
                "y": ("INT", {"default": 0, "tooltip": "融合左上角Y坐标"}),
                "width": ("INT", {"default": 0, "tooltip": "融合宽度"}),
                "height": ("INT", {"default": 0, "tooltip": "融合高度"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Poisson Blend"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "使用OpenCV泊松融合(seamlessClone)将前景融合到背景。"

    def execute(self, background, foreground, mask, x, y, width, height):
        device = background.device
        _, h, w, c = background.shape
        bg = (background[0].cpu().numpy() * 255).astype(np.uint8).squeeze()
        fg = (foreground[0].cpu().numpy() * 255).astype(np.uint8).squeeze()
        m = (mask[0].cpu().numpy() * 255).astype(np.uint8).squeeze()
        if len(m.shape) > 2 and m.shape[-1] > 1:
            m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        # 将前景图像通过mask裁剪出来，然后缩放到指定大小
        box = cv2.boundingRect(m)
        fg = fg[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
        m = m[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
        fg = cv2.resize(fg, (width, height), interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
        # 融合
        blended = cv2.seamlessClone(
            fg, bg, m, (x + width // 2, y + height // 2), cv2.NORMAL_CLONE
        )
        # 转换为tensor
        blended = torch.from_numpy(blended.astype(np.float32) / 255.0).to(device)
        if blended.ndim == 2:
            blended = blended.unsqueeze(-1)
        return (blended,)
