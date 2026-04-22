import torch


class ImageCropByMask:
    """Image Crop By Mask node.

    Crop image to the minimum bounding box of the mask above threshold.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to crop."}),
                "mask": ("MASK", {"tooltip": "The mask to crop by."}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Threshold for mask values to be considered valid.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "x", "y", "width", "height")
    FUNCTION = "execute"

    OUTPUT_NODE = False

    _NODE_NAME = "Image Crop By Mask"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "Crop image to the minimum bounding box of the mask above threshold."

    def execute(self, image, mask, threshold):
        """
        根据 mask 裁剪图像到最小矩形区域
        
        Args:
            image: [N, H, W, C] 格式的图像张量
            mask: [N, H, W] 格式的 mask 张量
            threshold: float 阈值，mask 中大于该值的区域被视为有效区域
            
        Returns:
            (cropped_image, x, y, width, height)
            - cropped_image: 裁剪后的图像 [N, H', W', C]
            - x, y: 裁剪区域的左上角坐标
            - width, height: 裁剪区域的宽度和高度
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, H, W, C]
        N, H, W, C = image.shape

        # 确保 mask 的批次和尺寸与 image 匹配
        # 处理 mask 的维度：可能是 [N, H, W] 或 [H, W] 或 [N, H, W, 1]
        if mask.dim() == 4:
            mask = mask.squeeze()
        if mask.dim() == 2:
            # [H, W] -> [1, H, W]
            mask = mask.unsqueeze(0)

        # 现在 mask 应该是 [N, H, W]
        if mask.shape[0] != N:
            # 如果批次不匹配，使用第一个 mask
            mask = mask[0:1].expand(N, -1, -1)

        if mask.shape[1:] != (H, W):
            # 如果尺寸不匹配，进行插值
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1),  # [N, 1, H_mask, W_mask]
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [N, H, W]

        # 对所有批次的 mask 进行 OR 操作，找到整体的有效区域
        combined_mask = (mask > threshold).any(dim=0)  # [H, W]

        # 找到有效区域的边界
        rows = torch.any(combined_mask, dim=1)  # [H]
        cols = torch.any(combined_mask, dim=0)  # [W]

        # 如果没有有效区域，返回原图
        if not rows.any() or not cols.any():
            return (image, 0, 0, W, H)

        # 找到最小边界框
        row_indices = torch.where(rows)[0]
        col_indices = torch.where(cols)[0]

        y_min = row_indices.min().item()
        y_max = row_indices.max().item() + 1  # +1 因为切片是开区间
        x_min = col_indices.min().item()
        x_max = col_indices.max().item() + 1

        # 裁剪图像
        cropped_image = image[:, y_min:y_max, x_min:x_max, :]

        # 计算裁剪区域的宽度和高度
        crop_width = x_max - x_min
        crop_height = y_max - y_min

        return (cropped_image, x_min, y_min, crop_width, crop_height)
