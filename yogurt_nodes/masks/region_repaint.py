from __future__ import annotations

from collections import deque
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
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
        raise ValueError(f"Expected mask as [N,H,W] or compatible, got {tuple(mask.shape)}")
    return mask


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected image as [N,H,W,C], got {tuple(image.shape)}")
    return image


def _max_pool_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    return F.max_pool2d(
        mask.unsqueeze(1).to(torch.float32),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    ).squeeze(1)


def _min_pool_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask
    return 1.0 - _max_pool_mask(1.0 - mask, radius)


def _close_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return _min_pool_mask(_max_pool_mask(mask, radius), radius)


def _gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask

    kernel_size = radius * 2 + 1
    sigma = max(radius / 3.0, 0.1)
    coords = torch.arange(kernel_size, device=mask.device, dtype=torch.float32) - radius
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1)

    work = mask.unsqueeze(1).to(torch.float32)
    pad = (radius, radius, radius, radius)
    work = F.pad(work, pad, mode="replicate")
    work = F.conv2d(work, kernel_x)
    work = F.conv2d(work, kernel_y)
    return torch.clamp(work.squeeze(1), 0.0, 1.0).to(mask.dtype)


def _bbox_from_binary(binary: torch.Tensor) -> list[int] | None:
    rows = torch.any(binary, dim=1)
    cols = torch.any(binary, dim=0)
    if not rows.any() or not cols.any():
        return None
    y_indices = torch.where(rows)[0]
    x_indices = torch.where(cols)[0]
    x_min = int(x_indices.min().item())
    x_max = int(x_indices.max().item()) + 1
    y_min = int(y_indices.min().item())
    y_max = int(y_indices.max().item()) + 1
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _union_bbox(a: list[int], b: list[int]) -> list[int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return [x0, y0, x1 - x0, y1 - y0]


def _expand_bbox(bbox: list[int], padding: int, width: int, height: int) -> list[int]:
    x, y, w, h = bbox
    padding = max(int(padding), 0)
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, width)
    y1 = min(y + h + padding, height)
    return [x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)]


def _ceil_to_multiple(value: int, multiple: int) -> int:
    multiple = max(int(multiple), 1)
    value = int(value)
    return int(math.ceil(value / multiple) * multiple)


def _bbox_gap(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(bx - (ax + aw), ax - (bx + bw), 0)
    dy = max(by - (ay + ah), ay - (by + bh), 0)
    return float(math.hypot(dx, dy))


def _bbox_intersection_area(a: list[int], b: list[int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    return max(x1 - x0, 0) * max(y1 - y0, 0)


def _bbox_overlap_ratio(a: list[int], b: list[int]) -> float:
    denominator = max(min(_bbox_area(a), _bbox_area(b)), 1)
    return float(_bbox_intersection_area(a, b) / denominator)


def _place_tile_for_bbox(
    bbox: list[int],
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    multiple_of: int,
) -> tuple[list[int], bool]:
    x, y, w, h = bbox
    tile_width = min(_ceil_to_multiple(tile_width, multiple_of), image_width)
    tile_height = min(_ceil_to_multiple(tile_height, multiple_of), image_height)

    oversize = w > tile_width or h > tile_height
    if oversize:
        out_w = min(_ceil_to_multiple(w, multiple_of), image_width)
        out_h = min(_ceil_to_multiple(h, multiple_of), image_height)
    else:
        out_w = tile_width
        out_h = tile_height

    cx = x + w / 2.0
    cy = y + h / 2.0
    x0 = int(round(cx - out_w / 2.0))
    y0 = int(round(cy - out_h / 2.0))

    x0 = min(x0, x)
    y0 = min(y0, y)
    if x + w > x0 + out_w:
        x0 = x + w - out_w
    if y + h > y0 + out_h:
        y0 = y + h - out_h

    x0 = min(max(x0, 0), max(image_width - out_w, 0))
    y0 = min(max(y0, 0), max(image_height - out_h, 0))
    return [int(x0), int(y0), int(out_w), int(out_h)], oversize


def _place_tile_for_required_and_preferred_bbox(
    required_bbox: list[int],
    preferred_bbox: list[int],
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    multiple_of: int,
) -> tuple[list[int], bool]:
    preferred_width = int(preferred_bbox[2])
    preferred_height = int(preferred_bbox[3])
    if preferred_width <= int(tile_width) and preferred_height <= int(tile_height):
        return _place_tile_for_bbox(
            preferred_bbox,
            tile_width,
            tile_height,
            image_width,
            image_height,
            multiple_of,
        )
    return _place_tile_for_bbox(
        required_bbox,
        tile_width,
        tile_height,
        image_width,
        image_height,
        multiple_of,
    )


def _tile_feasible_range(
    safe_bbox: list[int],
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = safe_bbox
    x_min = max(x + w - tile_width, 0)
    x_max = min(x, max(image_width - tile_width, 0))
    y_min = max(y + h - tile_height, 0)
    y_max = min(y, max(image_height - tile_height, 0))
    if x_min > x_max:
        x_min = x_max
    if y_min > y_max:
        y_min = y_max
    return int(x_min), int(x_max), int(y_min), int(y_max)


def _bbox_area(bbox: list[int]) -> int:
    return max(int(bbox[2]), 0) * max(int(bbox[3]), 0)


def _vertical_gap(a: list[int], b: list[int]) -> int:
    ay0 = int(a[1])
    ay1 = int(a[1] + a[3])
    by0 = int(b[1])
    by1 = int(b[1] + b[3])
    return max(by0 - ay1, ay0 - by1, 0)


def _compact_region_tiles(
    regions: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
) -> list[dict[str, Any]]:
    if len(regions) <= 1:
        return regions

    rows: list[dict[str, Any]] = []
    for region in sorted(regions, key=lambda item: (item["safe_bbox"][1], item["safe_bbox"][0])):
        for row in rows:
            if _vertical_gap(row["safe_bbox"], region["safe_bbox"]) == 0:
                row["regions"].append(region)
                row["safe_bbox"] = _union_bbox(row["safe_bbox"], region["safe_bbox"])
                break
        else:
            rows.append({"safe_bbox": list(region["safe_bbox"]), "regions": [region]})

    compacted: list[dict[str, Any]] = []
    for row in rows:
        previous_x: int | None = None
        previous_width = 0
        row_regions = sorted(row["regions"], key=lambda item: item["safe_bbox"][0])
        if len(row_regions) == 1:
            compacted.append(row_regions[0])
            continue
        for region in row_regions:
            x_min, x_max, y_min, y_max = _tile_feasible_range(
                region["safe_bbox"],
                tile_width,
                tile_height,
                image_width,
                image_height,
            )
            if previous_x is None:
                current_x = int(region["bbox"][0])
                x0 = min(max(current_x, x_min), x_max)
            else:
                x0 = max(x_min, previous_x + previous_width)
                if x0 > x_max:
                    x0 = x_max
            current_y = int(region["bbox"][1])
            y0 = min(max(current_y, y_min), y_max)
            region["bbox"] = [int(x0), int(y0), int(tile_width), int(tile_height)]
            previous_x = int(x0)
            previous_width = int(tile_width)
            compacted.append(region)

    compacted.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return compacted


def _component_neighbors(connectivity: str) -> tuple[tuple[int, int], ...]:
    if str(connectivity) == "4":
        return ((1, 0), (-1, 0), (0, 1), (0, -1))
    return (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )


def _connected_components(binary: np.ndarray, connectivity: str) -> list[dict[str, Any]]:
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    neighbors = _component_neighbors(connectivity)
    components: list[dict[str, Any]] = []

    ys, xs = np.nonzero(binary)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []

        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and binary[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        comp_y = np.fromiter((p[0] for p in pixels), dtype=np.int64)
        comp_x = np.fromiter((p[1] for p in pixels), dtype=np.int64)
        comp_mask = np.zeros_like(binary, dtype=np.float32)
        comp_mask[comp_y, comp_x] = 1.0
        x0 = int(comp_x.min())
        x1 = int(comp_x.max()) + 1
        y0 = int(comp_y.min())
        y1 = int(comp_y.max()) + 1
        components.append(
            {
                "mask": comp_mask,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": int(len(pixels)),
                "center": [float(comp_x.mean()), float(comp_y.mean())],
            }
        )

    components.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return components


def _draw_rect(image: torch.Tensor, bbox: list[int], color: tuple[float, float, float]) -> None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return
    x1 = min(x + w - 1, image.shape[1] - 1)
    y1 = min(y + h - 1, image.shape[0] - 1)
    x = max(x, 0)
    y = max(y, 0)
    color_tensor = torch.tensor(color, device=image.device, dtype=image.dtype)
    image[y, x : x1 + 1, :3] = color_tensor
    image[y1, x : x1 + 1, :3] = color_tensor
    image[y : y1 + 1, x, :3] = color_tensor
    image[y : y1 + 1, x1, :3] = color_tensor


def _region_preview(image: torch.Tensor, regions: list[dict[str, Any]]) -> torch.Tensor:
    image = _normalize_image(image)
    preview = image[:1].clone()
    if preview.shape[-1] < 3:
        preview = preview.expand(-1, -1, -1, 3).clone()
    colors = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.4, 1.0),
        (1.0, 0.8, 0.0),
        (1.0, 0.0, 1.0),
    )
    for index, region in enumerate(regions):
        _draw_rect(preview[0], region["bbox"], colors[index % len(colors)])
        _draw_rect(preview[0], region["safe_bbox"], (1.0, 1.0, 1.0))
    return torch.clamp(preview, 0.0, 1.0)


def _mask_items_from_masks(
    masks: torch.Tensor,
    threshold: float,
) -> list[dict[str, Any]]:
    masks = _normalize_mask(masks)
    items: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        binary = mask > threshold
        bbox = _bbox_from_binary(binary)
        if bbox is None:
            continue
        items.append(
            {
                "id": int(index),
                "source_batch": int(index),
                "bbox": bbox,
                "area": int(binary.sum().item()),
                "center": [
                    float(bbox[0] + bbox[2] / 2.0),
                    float(bbox[1] + bbox[3] / 2.0),
                ],
            }
        )
    return items


class SplitMask:
    """Split Mask node.

    Split a combined mask into one mask batch item per connected component.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Input mask or mask batch."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_area": ("INT", {"default": 16, "min": 0, "max": 1048576, "step": 1}),
                "connectivity": (["8", "4"], {"default": "8"}),
                "close_radius": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "mode": (["split_components", "preserve_batch"], {"default": "split_components"}),
            }
        }

    RETURN_TYPES = ("MASK", "DICT")
    RETURN_NAMES = ("masks", "mask_info")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Split Mask"
    DESCRIPTION = "Split a mask into a batch of connected-component masks."

    def execute(
        self,
        mask: torch.Tensor,
        threshold: float,
        min_area: int,
        connectivity: str,
        close_radius: int,
        mode: str,
    ):
        masks = _normalize_mask(mask)
        device = masks.device
        dtype = masks.dtype
        binary_batch = (masks > float(threshold)).to(torch.float32)
        binary_batch = _close_mask(binary_batch, int(close_radius))
        binary_batch = binary_batch > 0.5

        output_masks: list[torch.Tensor] = []
        items: list[dict[str, Any]] = []

        if mode == "preserve_batch":
            for source_index, binary in enumerate(binary_batch):
                area = int(binary.sum().item())
                if area < int(min_area):
                    continue
                bbox = _bbox_from_binary(binary)
                if bbox is None:
                    continue
                out_id = len(output_masks)
                output_masks.append(binary.to(device=device, dtype=dtype))
                items.append(
                    {
                        "id": out_id,
                        "source_batch": int(source_index),
                        "bbox": bbox,
                        "area": area,
                        "center": [
                            float(bbox[0] + bbox[2] / 2.0),
                            float(bbox[1] + bbox[3] / 2.0),
                        ],
                    }
                )
        else:
            for source_index, binary in enumerate(binary_batch):
                components = _connected_components(binary.cpu().numpy().astype(bool), connectivity)
                for component in components:
                    if int(component["area"]) < int(min_area):
                        continue
                    out_id = len(output_masks)
                    output_masks.append(
                        torch.from_numpy(component["mask"]).to(device=device, dtype=dtype)
                    )
                    items.append(
                        {
                            "id": out_id,
                            "source_batch": int(source_index),
                            "bbox": component["bbox"],
                            "area": component["area"],
                            "center": component["center"],
                        }
                    )

        if not output_masks:
            _, height, width = masks.shape
            empty = torch.zeros((1, height, width), device=device, dtype=dtype)
            return (
                empty,
                {
                    "version": 1,
                    "count": 0,
                    "height": int(height),
                    "width": int(width),
                    "items": [],
                },
            )

        output = torch.stack(output_masks, dim=0)
        _, height, width = output.shape
        return (
            output,
            {
                "version": 1,
                "count": len(items),
                "height": int(height),
                "width": int(width),
                "items": items,
            },
        )


class MaskRegionPlanner:
    """Mask Region Planner node.

    Plan dynamic fixed-size repaint tiles from a mask batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Original image used for size and preview."}),
                "masks": ("MASK", {"tooltip": "Split mask batch."}),
                "tile_width": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 1}),
                "tile_height": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 1}),
                "context_padding": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
                "edit_grow": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "blend_expand": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "blend_feather": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
                "merge_distance": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Tie-break hint for merge order. Regions merge whenever their required safe boxes fit one tile.",
                    },
                ),
                "max_waste_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("DICT", "IMAGE")
    RETURN_NAMES = ("region_info", "preview")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Mask Region Planner"
    DESCRIPTION = "Plan dynamic fixed-size repaint tiles from split masks."

    def _make_region(
        self,
        item: dict[str, Any],
        image_width: int,
        image_height: int,
        tile_width: int,
        tile_height: int,
        required_padding: int,
        context_padding: int,
        multiple_of: int,
    ) -> dict[str, Any]:
        safe_bbox = _expand_bbox(item["bbox"], required_padding, image_width, image_height)
        preferred_bbox = _expand_bbox(
            item["bbox"],
            required_padding + max(int(context_padding), 0),
            image_width,
            image_height,
        )
        tile_bbox, oversize = _place_tile_for_required_and_preferred_bbox(
            safe_bbox,
            preferred_bbox,
            tile_width,
            tile_height,
            image_width,
            image_height,
            multiple_of,
        )
        return {
            "id": 0,
            "mask_ids": [int(item["id"])],
            "object_bbox": item["bbox"],
            "safe_bbox": safe_bbox,
            "preferred_bbox": preferred_bbox,
            "bbox": tile_bbox,
            "mask_area": int(item["area"]),
            "safe_area": _bbox_area(safe_bbox),
            "oversize": bool(oversize),
        }

    def _try_merge(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        image_width: int,
        image_height: int,
        tile_width: int,
        tile_height: int,
        multiple_of: int,
        merge_distance: int,
        max_waste_ratio: float,
    ) -> tuple[dict[str, Any], float] | None:
        safe_bbox = _union_bbox(left["safe_bbox"], right["safe_bbox"])
        preferred_bbox = _union_bbox(
            left.get("preferred_bbox", left["safe_bbox"]),
            right.get("preferred_bbox", right["safe_bbox"]),
        )
        gap = _bbox_gap(left["safe_bbox"], right["safe_bbox"])
        tile_overlap_ratio = _bbox_overlap_ratio(left["bbox"], right["bbox"])
        if safe_bbox[2] > tile_width or safe_bbox[3] > tile_height:
            return None

        tile_bbox, oversize = _place_tile_for_required_and_preferred_bbox(
            safe_bbox,
            preferred_bbox,
            tile_width,
            tile_height,
            image_width,
            image_height,
            multiple_of,
        )
        if oversize:
            return None

        mask_area = int(left["mask_area"]) + int(right["mask_area"])
        safe_area = int(left.get("safe_area", _bbox_area(left["safe_bbox"]))) + int(
            right.get("safe_area", _bbox_area(right["safe_bbox"]))
        )
        union_area = max(_bbox_area(safe_bbox), 1)
        waste_ratio = max(union_area - safe_area, 0) / union_area
        if waste_ratio > float(max_waste_ratio):
            return None

        merged = {
            "id": 0,
            "mask_ids": sorted(left["mask_ids"] + right["mask_ids"]),
            "object_bbox": _union_bbox(left["object_bbox"], right["object_bbox"]),
            "safe_bbox": safe_bbox,
            "preferred_bbox": preferred_bbox,
            "bbox": tile_bbox,
            "mask_area": mask_area,
            "safe_area": safe_area,
            "oversize": False,
        }
        distance_penalty = max(gap - float(merge_distance), 0.0) / max(tile_width, tile_height, 1)
        score = float(waste_ratio + distance_penalty - tile_overlap_ratio)
        return merged, score

    def execute(
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
        tile_width: int,
        tile_height: int,
        context_padding: int,
        edit_grow: int,
        blend_expand: int,
        blend_feather: int,
        multiple_of: int,
        merge_distance: int,
        max_waste_ratio: float,
        threshold: float,
    ):
        image = _normalize_image(image)
        masks = _normalize_mask(masks)
        image_batch, image_height, image_width, _ = image.shape
        if image_batch != 1:
            raise ValueError("Mask Region Planner currently supports a single image at a time.")
        if tuple(masks.shape[1:]) != (image_height, image_width):
            raise ValueError("masks and image must have the same spatial size.")

        requested_tile_width = int(tile_width)
        requested_tile_height = int(tile_height)
        tile_width = min(_ceil_to_multiple(requested_tile_width, int(multiple_of)), image_width)
        tile_height = min(_ceil_to_multiple(requested_tile_height, int(multiple_of)), image_height)
        if tile_width <= 0 or tile_height <= 0:
            raise ValueError("tile_width and tile_height must be positive.")

        items = _mask_items_from_masks(masks, float(threshold))
        required_padding = max(
            int(edit_grow),
            int(blend_expand) + int(blend_feather),
        )
        regions = [
            self._make_region(
                item,
                image_width,
                image_height,
                tile_width,
                tile_height,
                required_padding,
                int(context_padding),
                int(multiple_of),
            )
            for item in items
        ]
        oversized_regions = [region for region in regions if region.get("oversize")]
        if oversized_regions:
            first = oversized_regions[0]
            raise ValueError(
                "At least one mask safe region exceeds tile size. "
                f"Increase tile_width/tile_height; first oversized safe_bbox={first['safe_bbox']}, "
                f"tile_size={tile_width}x{tile_height}."
            )

        while len(regions) > 1:
            best: tuple[int, int, dict[str, Any], float] | None = None
            for left_index in range(len(regions)):
                for right_index in range(left_index + 1, len(regions)):
                    candidate = self._try_merge(
                        regions[left_index],
                        regions[right_index],
                        image_width,
                        image_height,
                        tile_width,
                        tile_height,
                        int(multiple_of),
                        int(merge_distance),
                        float(max_waste_ratio),
                    )
                    if candidate is None:
                        continue
                    merged, score = candidate
                    if best is None or score < best[3]:
                        best = (left_index, right_index, merged, score)
            if best is None:
                break
            left_index, right_index, merged, _ = best
            regions = [
                region
                for index, region in enumerate(regions)
                if index not in (left_index, right_index)
            ]
            regions.append(merged)
            regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))

        regions = _compact_region_tiles(
            regions,
            image_width,
            image_height,
            tile_width,
            tile_height,
        )

        for index, region in enumerate(regions):
            region["id"] = index

        region_info = {
            "version": 1,
            "count": len(regions),
            "orig_width": int(image_width),
            "orig_height": int(image_height),
            "requested_tile_width": int(requested_tile_width),
            "requested_tile_height": int(requested_tile_height),
            "tile_width": int(tile_width),
            "tile_height": int(tile_height),
            "context_padding": int(context_padding),
            "edit_grow": int(edit_grow),
            "blend_expand": int(blend_expand),
            "blend_feather": int(blend_feather),
            "multiple_of": int(multiple_of),
            "merge_distance": int(merge_distance),
            "max_waste_ratio": float(max_waste_ratio),
            "threshold": float(threshold),
            "order": "region_major",
            "regions": regions,
            "masks": items,
        }
        return (region_info, _region_preview(image, regions))


class CropImageByRegions:
    """Crop Image By Regions node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Original image."}),
                "masks": ("MASK", {"tooltip": "Split mask batch used by region_info."}),
                "region_info": ("DICT", {"tooltip": "Region info from Mask Region Planner."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "DICT")
    RETURN_NAMES = ("images", "edit_masks", "blend_masks", "object_masks", "region_info")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Crop Image By Regions"
    DESCRIPTION = "Crop repaint image and mask batches from dynamic mask regions."

    @staticmethod
    def _region_masks(
        masks: torch.Tensor,
        mask_ids: list[int],
        edit_grow: int,
        blend_expand: int,
        blend_feather: int,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = masks[mask_ids] > threshold
        object_mask = selected.any(dim=0, keepdim=True).to(torch.float32)
        edit_mask = _max_pool_mask(object_mask, int(edit_grow))
        blend_core = _max_pool_mask(object_mask, int(blend_expand))
        blend_mask = _gaussian_blur_mask(blend_core, int(blend_feather))
        blend_mask = torch.maximum(blend_core, blend_mask)
        return object_mask[0], edit_mask[0], torch.clamp(blend_mask[0], 0.0, 1.0)

    def execute(self, image: torch.Tensor, masks: torch.Tensor, region_info: dict):
        image = _normalize_image(image)
        masks = _normalize_mask(masks).to(device=image.device)
        if not isinstance(region_info, dict):
            raise ValueError("region_info must be a dict.")
        regions = region_info.get("regions", [])
        if not isinstance(regions, list):
            raise ValueError("region_info.regions must be a list.")

        edit_grow = int(region_info.get("edit_grow", 0))
        blend_expand = int(region_info.get("blend_expand", 0))
        blend_feather = int(region_info.get("blend_feather", 0))
        threshold = float(region_info.get("threshold", 0.5))
        batch, image_height, image_width, _ = image.shape
        if batch != 1:
            raise ValueError("Crop Image By Regions currently supports a single image at a time.")
        if tuple(masks.shape[1:]) != (image_height, image_width):
            raise ValueError("masks and image must have the same spatial size.")

        image_crops: list[torch.Tensor] = []
        edit_crops: list[torch.Tensor] = []
        blend_crops: list[torch.Tensor] = []
        object_crops: list[torch.Tensor] = []

        for region in regions:
            x, y, width, height = [int(value) for value in region["bbox"]]
            if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
                raise ValueError(f"Region bbox is outside the image: {region['bbox']}")

            object_mask, edit_mask, blend_mask = self._region_masks(
                masks[:, y : y + height, x : x + width],
                [int(mask_id) for mask_id in region["mask_ids"]],
                edit_grow,
                blend_expand,
                blend_feather,
                threshold,
            )
            image_crops.append(image[:, y : y + height, x : x + width, :])
            edit_crops.append(edit_mask.unsqueeze(0).expand(batch, -1, -1))
            blend_crops.append(blend_mask.unsqueeze(0).expand(batch, -1, -1))
            object_crops.append(object_mask.unsqueeze(0).expand(batch, -1, -1))

        if not image_crops:
            return (
                image[:1],
                torch.zeros((1, image_height, image_width), device=image.device, dtype=image.dtype),
                torch.zeros((1, image_height, image_width), device=image.device, dtype=image.dtype),
                torch.zeros((1, image_height, image_width), device=image.device, dtype=image.dtype),
                region_info,
            )

        crop_sizes = {(int(crop.shape[1]), int(crop.shape[2])) for crop in image_crops}
        if len(crop_sizes) != 1:
            raise ValueError(
                "Region crop sizes differ. Increase tile_width/tile_height so every safe mask "
                "region fits the fixed tile size, or process oversized masks separately."
            )

        return (
            torch.cat(image_crops, dim=0),
            torch.cat(edit_crops, dim=0).to(device=image.device, dtype=image.dtype),
            torch.cat(blend_crops, dim=0).to(device=image.device, dtype=image.dtype),
            torch.cat(object_crops, dim=0).to(device=image.device, dtype=image.dtype),
            region_info,
        )


class CompositeRepaintRegions:
    """Composite Repaint Regions node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {"tooltip": "Original or VAE-roundtripped base image."}),
                "repainted_regions": ("IMAGE", {"tooltip": "Repainted region image batch."}),
                "blend_masks": ("MASK", {"tooltip": "Blend masks from Crop Image By Regions."}),
                "region_info": ("DICT", {"tooltip": "Region info from Mask Region Planner."}),
                "alpha_normalize": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Composite Repaint Regions"
    DESCRIPTION = "Composite repainted region crops back onto a base image."

    def execute(
        self,
        base_image: torch.Tensor,
        repainted_regions: torch.Tensor,
        blend_masks: torch.Tensor,
        region_info: dict,
        alpha_normalize: bool,
    ):
        base_image = _normalize_image(base_image)
        repainted_regions = _normalize_image(repainted_regions)
        blend_masks = _normalize_mask(blend_masks).to(device=base_image.device)

        regions = region_info.get("regions", []) if isinstance(region_info, dict) else []
        if base_image.shape[0] != 1:
            raise ValueError("Composite Repaint Regions currently supports a single base image at a time.")
        if not regions:
            return (
                base_image,
                torch.zeros(
                    (base_image.shape[0], base_image.shape[1], base_image.shape[2]),
                    device=base_image.device,
                    dtype=base_image.dtype,
                ),
            )

        region_count = len(regions)
        if repainted_regions.shape[0] != region_count:
            raise ValueError("repainted_regions batch size must match region count.")
        image_batch = 1
        if blend_masks.shape[0] != repainted_regions.shape[0]:
            raise ValueError("blend_masks batch must match repainted_regions batch.")

        _, image_height, image_width, channels = base_image.shape
        repaint_accum = torch.zeros_like(base_image, dtype=torch.float32)
        alpha_accum = torch.zeros(
            (image_batch, image_height, image_width, 1),
            device=base_image.device,
            dtype=torch.float32,
        )

        for region_index, region in enumerate(regions):
            x, y, width, height = [int(value) for value in region["bbox"]]
            for batch_index in range(image_batch):
                source_index = region_index * image_batch + batch_index
                crop = repainted_regions[source_index, :height, :width, :channels].to(torch.float32)
                alpha = blend_masks[source_index, :height, :width].to(torch.float32).unsqueeze(-1)
                repaint_accum[
                    batch_index,
                    y : y + height,
                    x : x + width,
                    :,
                ] += crop * alpha
                alpha_accum[
                    batch_index,
                    y : y + height,
                    x : x + width,
                    :,
                ] += alpha

        coverage = torch.clamp(alpha_accum, 0.0, 1.0)
        if bool(alpha_normalize):
            repaint_value = repaint_accum / torch.clamp(alpha_accum, min=1e-6)
            result = base_image.to(torch.float32) * (1.0 - coverage) + repaint_value * coverage
        else:
            result = base_image.to(torch.float32) * (1.0 - coverage) + repaint_accum

        return (
            torch.clamp(result, 0.0, 1.0).to(base_image.dtype),
            torch.clamp(alpha_accum[..., 0], 0.0, 1.0).to(base_image.dtype),
        )


class PreviewRepaintRegions:
    """Preview Repaint Regions node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Original image."}),
                "region_info": ("DICT", {"tooltip": "Region info from Mask Region Planner."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Preview Repaint Regions"
    DESCRIPTION = "Draw planned repaint region boxes over the image."

    def execute(self, image: torch.Tensor, region_info: dict):
        regions = region_info.get("regions", []) if isinstance(region_info, dict) else []
        return (_region_preview(image, regions),)
