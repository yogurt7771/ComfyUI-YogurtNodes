import math

import torch

import comfy.utils


class ImageScaleToTotalPixelsAdvanced:
    """Image Scale To Total Pixels Advanced node."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The images to batch."}),
                "upscale_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {"default": "lanczos"},
                ),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "step": 0.1},
                ),
                "divide_by": (
                    "INT",
                    {"default": 1, "min": 1, "max": 2**31 - 1, "step": 1},
                ),
                "pad_value": ("STRING", {"default": "#000000"}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "INT",
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "width",
        "height",
        "channels",
        "longest_side",
        "shortest_side",
        "aspect_ratio",
        "pixels",
    )
    FUNCTION = "execute"

    OUTPUT_NODE = False

    _NODE_NAME = "Image Scale To Total Pixels Advanced"
    DESCRIPTION = "Image Scale To Total Pixels Advanced."

    @classmethod
    def _parse_hex_rgb(cls, hex_str: str):
        s = hex_str.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 6:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
        elif len(s) == 3:
            r = int(s[0] * 2, 16)
            g = int(s[1] * 2, 16)
            b = int(s[2] * 2, 16)
        else:
            raise ValueError(f"pad_value hex must be #RRGGBB or #RGB, got: {hex_str}")
        return r, g, b  # 0..255

    @classmethod
    def _rgb_to_gray_u8(cls, r, g, b):
        # Rec.709: 0.2126 R + 0.7152 G + 0.0722 B
        return int(round(0.2126 * r + 0.7152 * g + 0.0722 * b))

    @classmethod
    def _make_color_vector(cls, dtype, device, C, hex_rgb):
        """
        将 #RRGGBB 映射为与输入 dtype/通道数匹配的 per-channel 颜色向量 (C,)。
        - float/bfloat16 → 0..1
        - 整型 → 0..255
        - C==1 → 灰度
        - C==3 → RGB
        - C==4 → RGBA（A=1 或 255）
        - 其它通道 → 全用灰度
        """
        r_u8, g_u8, b_u8 = hex_rgb
        is_float = dtype in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        )

        if is_float:
            r = r_u8 / 255.0
            g = g_u8 / 255.0
            b = b_u8 / 255.0
            a = 1.0
        else:
            r = r_u8
            g = g_u8
            b = b_u8
            a = 255

        if C == 1:
            gray = cls._rgb_to_gray_u8(r_u8, g_u8, b_u8)
            gray = gray / 255.0 if is_float else gray
            vec = [gray]
        elif C == 3:
            vec = [r, g, b]
        elif C == 4:
            vec = [r, g, b, a]
        else:
            # 非常规通道：全部用灰度
            gray = cls._rgb_to_gray_u8(r_u8, g_u8, b_u8)
            gray = gray / 255.0 if is_float else gray
            vec = [gray] * C

        return torch.tensor(vec, dtype=dtype, device=device)  # (C,)

    @staticmethod
    def _white_value(dtype):
        return (
            1.0
            if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)
            else 255
        )

    def execute(
        self, image, upscale_method, megapixels, divide_by, pad_value: str
    ):
        """
        - image: [..., H, W, C]  (torch.Tensor)
        - upscale_method: 传给 comfy.utils.common_upscale 的插值方法
        - megapixels: 目标百万像素，如 1.0 -> 1,048,576
        - divide_by: 最终宽高需要被整除的数（如 8/16/32），<=1 则不约束
        - pad_value: '#RRGGBB' 或 '#RGB' 的填充色字符串
        返回：
        - io.NodeOutput(image_out, mask_out, width, height, channels, longest_side, shortest_side, aspect_ratio, pixels)
          image_out: [..., H_out, W_out, C]
          mask_out : [..., H_out, W_out, 1]  (白=有效，黑=pad)
        """
        # 1) HWC -> NCHW
        samples = image.movedim(-1, 1)  # [N, C, H, W]
        N, C, H, W = samples.shape
        dtype = samples.dtype
        device = samples.device

        # 2) 按 mega 等比缩放
        total = int(round(float(megapixels) * 1024 * 1024))
        scale: float = math.sqrt(max(total, 1) / max(W * H, 1))
        Wt = max(1, int(round(W * scale)))
        Ht = max(1, int(round(H * scale)))

        if divide_by > 1:
            # 分别计算以W和H为基准的输出尺寸，然后选取pad较小的那个
            W_Wout = (Wt // divide_by) * divide_by
            W_Ht = H * W_Wout // W
            W_Hout = ((W_Ht + divide_by - 1) // divide_by) * divide_by
            W_pad_h = W_Hout - W_Ht

            H_Hout = (Ht // divide_by) * divide_by
            H_Wt = W * H_Hout // H
            H_Wout = ((H_Wt + divide_by - 1) // divide_by) * divide_by
            H_pad_w = H_Wout - H_Wt

            if W_pad_h <= H_pad_w:
                Wout, Hout, Wt, Ht, sel_w, sel_h = (
                    W_Wout, W_Hout, W_Wout, W_Ht, 0, W_pad_h
                )
            else:
                Wout, Hout, Wt, Ht, sel_w, sel_h = (
                    H_Wout, H_Hout, H_Wt, H_Hout, H_pad_w, 0
                )
        else:
            Wout = Wt
            Hout = Ht
            sel_w = 0
            sel_h = 0

        print(f"ImageScaleToTotalPixelsAdvanced: input=({W}x{H}), output=({Wout}x{Hout}), scale=({Wt}x{Ht}), padW={sel_w}, padH={sel_h}")

        up = comfy.utils.common_upscale(
            samples, Wt, Ht, upscale_method, "disabled"
        )  # [N, C, Ht, Wt]

        # 4) 构造填充色画布，并把缩放图拷贝到中心
        if sel_w == 0 and sel_h == 0:
            final = up
        else:
            rgb = self._parse_hex_rgb(pad_value)
            color_vec = self._make_color_vector(dtype, device, C, rgb)  # (C,)
            canvas = color_vec.view(1, C, 1, 1).expand(N, C, Hout, Wout).clone()
            startH = (sel_h // 2) if sel_h > 0 else 0
            startW = (sel_w // 2) if sel_w > 0 else 0
            endH = startH + Ht
            endW = startW + Wt
            canvas[..., startH:endH, startW:endW] = up
            final = canvas

        # 5) 生成 mask（白=有效，黑=pad）
        white = self._white_value(dtype)
        if sel_w == 0 and sel_h == 0:
            mask_out = torch.full(
                (N, 1, Hout, Wout), fill_value=white, dtype=dtype, device=device
            )
        else:
            mask_out = torch.zeros((N, Hout, Wout), dtype=dtype, device=device)
            mask_out[:, startH:endH, startW:endW] = white

        # 6) 回到 HWC 与单通道 mask
        image_out = final.movedim(1, -1)  # [..., H_out, W_out, C]

        width = int(Wout)
        height = int(Hout)
        channels = int(C)
        longest_side = max(width, height)
        shortest_side = min(width, height)
        aspect_ratio = width / height if height != 0 else 0
        pixels = width * height

        return (
            image_out,
            mask_out,
            width,
            height,
            channels,
            longest_side,
            shortest_side,
            aspect_ratio,
            pixels,
        )
