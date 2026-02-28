import torch
import torch.nn.functional as F


def _make_positions(total_size: int, tile_size: int, seam: int) -> list[int]:
    if total_size <= tile_size:
        return [0]
    stride = tile_size - seam
    positions = list(range(0, total_size - tile_size + 1, stride))
    last = total_size - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _make_odd(x: int) -> int:
    x = int(x)
    if x > 0 and x % 2 == 0:
        return x + 1
    return x


def _gaussian_kernel_1d(radius: int, sigma: float, device, dtype):
    kernel_size = radius * 2 + 1
    x = torch.arange(kernel_size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_1d(signal: torch.Tensor, size: int) -> torch.Tensor:
    """
    Gaussian blur 1D signal, mirroring comfyui-inpaint-nodes behavior:
    - size is the kernel size (made odd)
    - sigma defaults to 0.3*(size-1)+0.8
    - padding uses reflect (clamped to fit)
    """
    size = _make_odd(int(size))
    if size <= 2:
        return signal

    original_dtype = signal.dtype
    sig_f32 = signal.to(torch.float32).view(1, 1, -1)  # [1,1,L]

    length = int(sig_f32.shape[-1])
    size = min(size, max(1, 2 * length - 1))
    radius = size // 2

    sigma = max(0.3 * (size - 1) + 0.8, 0.01)
    kernel_1d = _gaussian_kernel_1d(
        radius=radius,
        sigma=sigma,
        device=sig_f32.device,
        dtype=sig_f32.dtype,
    ).view(1, 1, -1)

    padded = F.pad(sig_f32, (radius, radius), mode="reflect")
    blurred = F.conv1d(padded, kernel_1d)
    return torch.clamp(blurred.view(-1), 0.0, 1.0).to(original_dtype)


def _make_rect_mask_pattern(
    *,
    tile_size: int,
    left_overlap: int,
    top_overlap: int,
    mask_blur: int,
    mask_expand: int,
    device,
    dtype,
) -> torch.Tensor:
    """
    Fast path for our tiling masks:
    - Base mask is a rectangle of 1s starting at (left_overlap, top_overlap) to bottom-right.
    - Optional expand is equivalent to shifting the rectangle start to the left/top.
    - Gaussian blur is separable -> build it from two 1D blurred steps and take outer product.
    - Keep the original white region unchanged via max(base, blurred).
    """
    tile_size = int(tile_size)
    left_overlap = max(int(left_overlap), 0)
    top_overlap = max(int(top_overlap), 0)
    mask_blur = int(mask_blur)
    mask_expand = max(int(mask_expand), 0)

    if left_overlap == 0 and top_overlap == 0:
        return torch.ones((tile_size, tile_size), device=device, dtype=dtype)

    idx = torch.arange(tile_size, device=device)
    ex = max(left_overlap - mask_expand, 0)
    ey = max(top_overlap - mask_expand, 0)

    step_x = (idx >= ex).to(torch.float32)
    step_y = (idx >= ey).to(torch.float32)
    expanded = (step_y[:, None] * step_x[None, :]).to(dtype)

    if mask_blur <= 0:
        return expanded

    blurred_x = _gaussian_blur_1d(step_x, mask_blur).to(torch.float32)
    blurred_y = _gaussian_blur_1d(step_y, mask_blur).to(torch.float32)
    blurred = (blurred_y[:, None] * blurred_x[None, :]).to(dtype)

    return torch.clamp(blurred, 0.0, 1.0)


class ImageTileWithSeamMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image. Output tiles are cropped from the top-left."}),
                "tile_size": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 1}),
                "seam": ("INT", {"default": 64, "min": 0, "max": 16384, "step": 1, "tooltip": "Overlap size in pixels."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1, "tooltip": "Gaussian blur kernel size (will be made odd)."}),
                "mask_expand": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1, "tooltip": "Grow (dilate) radius before blur."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "DICT")
    RETURN_NAMES = ("images", "masks", "tile_info")
    FUNCTION = "execute"

    OUTPUT_NODE = False

    _NODE_NAME = "Image Tile (Seam Mask)"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "Split image into overlapped tiles and generate inpaint masks (white=inpaint, black=reference)."

    def execute(
        self,
        image: torch.Tensor,
        tile_size: int,
        seam: int,
        mask_blur: int,
        mask_expand: int,
    ):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(f"Expected image as [N,H,W,C], got shape: {tuple(image.shape)}")

        tile_size = int(tile_size)
        seam = int(seam)
        mask_blur = int(mask_blur)
        mask_expand = int(mask_expand)

        if tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got: {tile_size}")
        if seam < 0:
            raise ValueError(f"seam must be >= 0, got: {seam}")
        if seam >= tile_size:
            raise ValueError(f"seam must be < tile_size, got seam={seam}, tile_size={tile_size}")
        if mask_expand < 0:
            raise ValueError(f"mask_expand must be >= 0, got: {mask_expand}")

        n, h, w, _ = image.shape

        effective_expand = mask_expand

        # If the image is smaller than the tile size, return it as a single tile.
        if w < tile_size or h < tile_size:
            masks_out = torch.ones((n, h, w), device=image.device, dtype=image.dtype)
            tile_info = {
                "version": 1,
                "tile_size": tile_size,
                "seam": seam,
                "mask_blur": mask_blur,
                "mask_expand": effective_expand,
                "orig_width": w,
                "orig_height": h,
                "crop_x": 0,
                "crop_y": 0,
                "crop_width": w,
                "crop_height": h,
                "x_positions": [0],
                "y_positions": [0],
                "rows": 1,
                "cols": 1,
                "tile_count": 1,
                "order": "row_major",
            }
            return (image, masks_out, tile_info)

        x_positions = _make_positions(w, tile_size, seam)
        y_positions = _make_positions(h, tile_size, seam)

        cols = len(x_positions)
        rows = len(y_positions)
        tile_count = int(rows * cols)

        images_out = torch.empty(
            (tile_count * n, tile_size, tile_size, image.shape[-1]),
            device=image.device,
            dtype=image.dtype,
        )
        masks_out = torch.empty(
            (tile_count * n, tile_size, tile_size),
            device=image.device,
            dtype=image.dtype,
        )

        mask_cache: dict[tuple[int, int], torch.Tensor] = {}

        tile_index = 0
        for row, y0 in enumerate(y_positions):
            for col, x0 in enumerate(x_positions):
                start = tile_index * n
                end = start + n

                images_out[start:end] = image[
                    :, y0 : y0 + tile_size, x0 : x0 + tile_size, :
                ]

                left_overlap = (
                    int((x_positions[col - 1] + tile_size) - x0) if col > 0 else 0
                )
                top_overlap = (
                    int((y_positions[row - 1] + tile_size) - y0) if row > 0 else 0
                )
                key = (left_overlap, top_overlap)

                mask_pattern = mask_cache.get(key)
                if mask_pattern is None:
                    mask_pattern = _make_rect_mask_pattern(
                        tile_size=tile_size,
                        left_overlap=left_overlap,
                        top_overlap=top_overlap,
                        mask_blur=mask_blur,
                        mask_expand=effective_expand,
                        device=image.device,
                        dtype=image.dtype,
                    )
                    mask_cache[key] = mask_pattern

                masks_out[start:end] = mask_pattern.unsqueeze(0).expand(n, -1, -1)

                tile_index += 1

        tile_info = {
            "version": 1,
            "tile_size": tile_size,
            "seam": seam,
            "mask_blur": mask_blur,
            "mask_expand": effective_expand,
            "orig_width": w,
            "orig_height": h,
            "crop_x": 0,
            "crop_y": 0,
            "crop_width": w,
            "crop_height": h,
            "x_positions": x_positions,
            "y_positions": y_positions,
            "rows": len(y_positions),
            "cols": len(x_positions),
            "tile_count": len(x_positions) * len(y_positions),
            "order": "row_major",
        }

        return (images_out, masks_out, tile_info)


class ImageUntileWithSeamMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {"tooltip": "Tiled images batch."}),
                "masks": ("MASK", {"tooltip": "Masks batch (white=inpaint, black=reference)."}),
                "tile_info": ("DICT", {"tooltip": "Tile info dict from Image Tile (Seam Mask)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"

    OUTPUT_NODE = False

    _NODE_NAME = "Image Untile (Seam Mask)"
    CATEGORY = "YogurtNodes/Image"
    DESCRIPTION = "Merge overlapped tiles back to one image using the mask as blend weight (white=tile, black=existing)."

    @staticmethod
    def _normalize_mask(mask: torch.Tensor, target_batch: int, target_h: int, target_w: int, device, dtype):
        if mask.dim() == 4:
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            elif mask.shape[1] == 1:
                mask = mask[:, 0, :, :]
            else:
                mask = mask.squeeze()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"Expected mask as [N,H,W] (or compatible), got shape: {tuple(mask.shape)}")

        if mask.shape[0] != target_batch:
            if mask.shape[0] == 1:
                mask = mask.expand(target_batch, -1, -1)
            else:
                raise ValueError(f"Mask batch must match tiles batch. mask={mask.shape[0]}, tiles={target_batch}")

        if mask.shape[1] != target_h or mask.shape[2] != target_w:
            mask = (
                F.interpolate(
                    mask.to(torch.float32).unsqueeze(1),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .to(mask.dtype)
            )

        mask = mask.to(device=device, dtype=torch.float32)
        return torch.clamp(mask, 0.0, 1.0).to(dtype)

    def execute(
        self,
        tiles: torch.Tensor,
        masks: torch.Tensor,
        tile_info: dict,
    ):
        if tiles.dim() == 3:
            tiles = tiles.unsqueeze(0)

        if tiles.dim() != 4:
            raise ValueError(f"Expected tiles as [B,H,W,C], got shape: {tuple(tiles.shape)}")
        if not isinstance(tile_info, dict):
            raise ValueError("tile_info must be a dict.")

        tile_size = int(tile_info.get("tile_size"))
        seam = int(tile_info.get("seam"))
        crop_x = int(tile_info.get("crop_x", 0))
        crop_y = int(tile_info.get("crop_y", 0))
        crop_w = int(tile_info.get("crop_width"))
        crop_h = int(tile_info.get("crop_height"))
        x_positions = tile_info.get("x_positions")
        y_positions = tile_info.get("y_positions")

        if tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got: {tile_size}")
        if seam < 0:
            raise ValueError(f"seam must be >= 0, got: {seam}")
        if seam >= tile_size:
            raise ValueError(f"seam must be < tile_size, got seam={seam}, tile_size={tile_size}")

        b_tiles, h_tile, w_tile, c = tiles.shape
        if crop_w <= 0 or crop_h <= 0:
            raise ValueError(f"crop_width/crop_height must be > 0, got: {crop_w}x{crop_h}")
        if not isinstance(x_positions, (list, tuple)) or not isinstance(y_positions, (list, tuple)):
            raise ValueError("tile_info.x_positions and tile_info.y_positions must be lists.")

        x_positions = [int(x) for x in x_positions]
        y_positions = [int(y) for y in y_positions]

        cols = len(x_positions)
        rows = len(y_positions)
        tile_count = int(rows * cols)

        # If there is only one tile, return as-is (supports both tiled and non-tiled cases).
        if tile_count == 1:
            return (tiles,)

        if h_tile != w_tile:
            raise ValueError(f"Tiles must be square. Got: {w_tile}x{h_tile}")
        if h_tile != tile_size:
            raise ValueError(f"tile_size mismatch. tile_info={tile_size}, tiles={h_tile}")

        if b_tiles % tile_count != 0:
            raise ValueError(f"Tiles batch size must be divisible by tile_count. tiles={b_tiles}, tile_count={tile_count}")

        n = int(b_tiles // tile_count)

        masks = self._normalize_mask(
            masks,
            target_batch=b_tiles,
            target_h=tile_size,
            target_w=tile_size,
            device=tiles.device,
            dtype=tiles.dtype,
        )

        canvas = torch.zeros((n, crop_h, crop_w, c), device=tiles.device, dtype=tiles.dtype)

        tile_index = 0
        for row, y0_abs in enumerate(y_positions):
            y0 = int(y0_abs - crop_y)
            for col, x0_abs in enumerate(x_positions):
                x0 = int(x0_abs - crop_x)
                start = tile_index * n
                end = start + n

                tile_batch = tiles[start:end, :, :, :]
                mask_batch = masks[start:end, :, :].unsqueeze(-1)

                region = canvas[:, y0 : y0 + tile_size, x0 : x0 + tile_size, :]
                canvas[:, y0 : y0 + tile_size, x0 : x0 + tile_size, :] = (
                    tile_batch * mask_batch + region * (1.0 - mask_batch)
                )

                tile_index += 1

        return (canvas,)
