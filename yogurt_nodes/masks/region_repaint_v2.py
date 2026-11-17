"""Region repaint V2 nodes: dynamic square tiles, greedy minimal-tile merge,
list I/O, no scaling.

V2 keeps every tile at its native resolution and emits one list element per
tile (``OUTPUT_IS_LIST``); the external workflow handles upscale -> sample ->
downscale per tile and feeds the results back to the V2 compositor
(``INPUT_IS_LIST``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# Reuse V1 pure helpers. Relative import works inside the ComfyUI package; the
# fallback loads the sibling file directly so the module is testable standalone
# via importlib.spec_from_file_location (same pattern the tests use).
try:  # pragma: no cover - exercised via ComfyUI package import
    from .region_repaint import (
        CropImageByRegions as _V1Crop,
        _bbox_area,
        _ceil_to_multiple,
        _expand_bbox,
        _mask_items_from_masks,
        _normalize_image,
        _normalize_mask,
        _region_preview,
        _union_bbox,
    )
except ImportError:  # pragma: no cover - exercised via standalone test import
    import importlib.util
    import pathlib

    _v1_path = pathlib.Path(__file__).with_name("region_repaint.py")
    _spec = importlib.util.spec_from_file_location("yogurt_region_repaint_v1_for_v2", _v1_path)
    _v1 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_v1)
    _V1Crop = _v1.CropImageByRegions
    _bbox_area = _v1._bbox_area
    _ceil_to_multiple = _v1._ceil_to_multiple
    _expand_bbox = _v1._expand_bbox
    _mask_items_from_masks = _v1._mask_items_from_masks
    _normalize_image = _v1._normalize_image
    _normalize_mask = _v1._normalize_mask
    _region_preview = _v1._region_preview
    _union_bbox = _v1._union_bbox

_region_masks = _V1Crop._region_masks


def _place_square_tile(side: int, anchor: list[int], width: int, height: int) -> list[int]:
    """Place a square tile of edge ``side`` covering ``anchor`` and clamped to
    the image. Centered on the anchor, shifted to contain it, then dragged
    inside ``[0, dim-side]``. ``side`` is assumed <= min(width, height)."""
    side = int(side)
    x, y, w, h = (int(v) for v in anchor)
    cx = x + w / 2.0
    cy = y + h / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    # prefer to fully contain the anchor
    x0 = min(x0, x)
    y0 = min(y0, y)
    if x + w > x0 + side:
        x0 = x + w - side
    if y + h > y0 + side:
        y0 = y + h - side
    # clamp inside the image
    x0 = min(max(x0, 0), max(width - side, 0))
    y0 = min(max(y0, 0), max(height - side, 0))
    return [int(x0), int(y0), side, side]


def _merge_regions_min_tiles(
    items: list[dict[str, Any]], s_max: int, min_fill_ratio: float
) -> list[dict[str, Any]]:
    """Greedy agglomerative merge minimizing the number of square tiles, gated
    by an effective-area fill ratio.

    Each item carries ``ids``/``object``/``safe``/``pref`` bboxes and a
    ``content`` area (sum of member object bbox areas). Two groups merge when
    (1) the union of their ``safe`` boxes fits a single ``s_max`` square and
    (2) the merged tile is not mostly empty: ``content / base^2 >=
    min_fill_ratio`` where ``base`` is the larger side of the safe union. This
    stops far-apart small masks from being merged into a big sparse tile that
    would shrink each object after the downstream scale-to-sample_size. Among
    eligible pairs the most compact union (smallest area) is merged first,
    repeated to a fixed point.
    """
    groups: list[dict[str, Any]] = [
        {
            "ids": list(item["ids"]),
            "object": list(item["object"]),
            "safe": list(item["safe"]),
            "pref": list(item["pref"]),
            "content": int(item["content"]),
        }
        for item in items
    ]

    min_fill_ratio = max(float(min_fill_ratio), 0.0)
    while True:
        best: tuple[int, int, int, list[int]] | None = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                union_safe = _union_bbox(groups[a]["safe"], groups[b]["safe"])
                if union_safe[2] > s_max or union_safe[3] > s_max:
                    continue
                base = max(union_safe[2], union_safe[3])
                content = groups[a]["content"] + groups[b]["content"]
                if content / max(base * base, 1) < min_fill_ratio:
                    continue
                area = union_safe[2] * union_safe[3]
                if best is None or area < best[0]:
                    best = (area, a, b, union_safe)
        if best is None:
            break
        _, a, b, union_safe = best
        left, right = groups[a], groups[b]
        merged = {
            "ids": sorted(left["ids"] + right["ids"]),
            "object": _union_bbox(left["object"], right["object"]),
            "safe": union_safe,
            "pref": _union_bbox(left["pref"], right["pref"]),
            "content": int(left["content"]) + int(right["content"]),
        }
        groups = [g for index, g in enumerate(groups) if index not in (a, b)]
        groups.append(merged)

    return groups


class MaskRegionPlannerV2:
    """Mask Region Planner V2 node.

    Plan dynamic square repaint tiles (merged to fewest tiles, native size).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Original image used for size and preview."}),
                "masks": ("MASK", {"tooltip": "Split mask batch."}),
                "min_tile": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 1}),
                "max_tile": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 8,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Max native square tile edge (px) and merge bound; "
                        "clamped by min(image_w, image_h). Native crop size only — "
                        "unrelated to sampling resolution.",
                    },
                ),
                "min_fill_ratio": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Merge gate: only merge masks if the resulting tile's fill "
                        "(object area / safe-union square) stays >= this. Higher = split sparse "
                        "clusters into more, tighter tiles (better per-object detail, more passes); "
                        "0 = merge as much as fits max_tile.",
                    },
                ),
                "context_padding": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
                "edit_grow": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "blend_expand": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "blend_feather": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 1}),
                "boundary_exclude": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "Interior tile-border band (px) zeroed from edit/blend "
                        "masks; also the safe-box margin the object keeps from inner edges.",
                    },
                ),
                "multiple_of": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("DICT", "IMAGE")
    RETURN_NAMES = ("region_info", "preview")
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Mask Region Planner V2"
    DESCRIPTION = "Plan dynamic square repaint tiles (merged to fewest tiles, native size)."

    def execute(
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
        min_tile: int,
        max_tile: int,
        min_fill_ratio: float,
        context_padding: int,
        edit_grow: int,
        blend_expand: int,
        blend_feather: int,
        boundary_exclude: int,
        multiple_of: int,
        threshold: float,
    ):
        image = _normalize_image(image)
        masks = _normalize_mask(masks)
        image_batch, image_height, image_width, _ = image.shape
        if image_batch != 1:
            raise ValueError("Mask Region Planner V2 currently supports a single image at a time.")
        if tuple(masks.shape[1:]) != (image_height, image_width):
            raise ValueError("masks and image must have the same spatial size.")

        mult = max(int(multiple_of), 1)
        s_max = min(int(max_tile), image_width, image_height)
        s_max = max((s_max // mult) * mult, mult)
        min_t = min(int(min_tile), s_max)

        boundary_exclude = max(int(boundary_exclude), 0)
        pad_pref = (
            max(int(edit_grow), int(blend_expand) + int(blend_feather))
            + boundary_exclude
            + max(int(context_padding), 0)
        )

        items = _mask_items_from_masks(masks, float(threshold))
        merge_items: list[dict[str, Any]] = []
        for item in items:
            merge_items.append(
                {
                    "ids": [int(item["id"])],
                    "object": list(item["bbox"]),
                    "safe": _expand_bbox(item["bbox"], boundary_exclude, image_width, image_height),
                    "pref": _expand_bbox(item["bbox"], pad_pref, image_width, image_height),
                    "content": _bbox_area(item["bbox"]),
                    "mask_area": int(item["area"]),
                }
            )

        groups = _merge_regions_min_tiles(merge_items, s_max, float(min_fill_ratio))
        groups.sort(key=lambda g: (g["safe"][1], g["safe"][0]))

        regions: list[dict[str, Any]] = []
        for index, group in enumerate(groups):
            base = max(group["safe"][2], group["safe"][3])
            want = max(group["pref"][2], group["pref"][3])
            side = _ceil_to_multiple(max(base, want, min_t), mult)
            side = min(max(side, min_t), s_max)
            clipped = base > s_max
            if clipped:
                print(
                    f"[MaskRegionPlannerV2] safe region {group['safe']} exceeds max tile "
                    f"{s_max}px; tile clamped to {s_max} and object edge may be clipped."
                )
            tile_bbox = _place_square_tile(side, group["safe"], image_width, image_height)
            regions.append(
                {
                    "id": index,
                    "mask_ids": sorted(int(i) for i in group["ids"]),
                    "object_bbox": group["object"],
                    "safe_bbox": group["safe"],
                    "preferred_bbox": group["pref"],
                    "bbox": tile_bbox,
                    "native_size": int(tile_bbox[2]),
                    "safe_area": _bbox_area(group["safe"]),
                    "clipped": bool(clipped),
                }
            )

        region_info = {
            "version": 2,
            "kind": "dynamic_square_tiles",
            "count": len(regions),
            "orig_width": int(image_width),
            "orig_height": int(image_height),
            "min_tile": int(min_t),
            "max_tile_effective": int(s_max),
            "min_fill_ratio": float(min_fill_ratio),
            "context_padding": int(context_padding),
            "edit_grow": int(edit_grow),
            "blend_expand": int(blend_expand),
            "blend_feather": int(blend_feather),
            "boundary_exclude": int(boundary_exclude),
            "multiple_of": int(mult),
            "threshold": float(threshold),
            "order": "region_major",
            "regions": regions,
            "masks": items,
        }
        return (region_info, _region_preview(image, regions))


def _zero_interior_border(mask: torch.Tensor, region: dict[str, Any], width: int, height: int, band: int) -> None:
    """Zero a ``band``-px strip on every tile edge that does not coincide with
    the true image border. ``mask`` is a native [H, W] tile (modified in place)."""
    band = int(band)
    if band <= 0:
        return
    x, y, side, _ = (int(v) for v in region["bbox"])
    if x > 0:
        mask[:, :band] = 0.0
    if y > 0:
        mask[:band, :] = 0.0
    if x + side < width:
        mask[:, side - band :] = 0.0
    if y + side < height:
        mask[side - band :, :] = 0.0


def _resize_image(crop: torch.Tensor, size: int) -> torch.Tensor:
    """Resize an image crop [B, h, w, C] to [B, size, size, C] (bilinear)."""
    work = crop.permute(0, 3, 1, 2)
    work = F.interpolate(work, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    return work.permute(0, 2, 3, 1).contiguous()


def _resize_mask(crop: torch.Tensor, size: int, hard: bool) -> torch.Tensor:
    """Resize a mask crop [B, h, w] to [B, size, size]. ``hard`` re-binarizes
    (for object/edit masks); soft keeps the feather (for blend masks)."""
    work = crop.unsqueeze(1)
    work = F.interpolate(work, size=(int(size), int(size)), mode="bilinear", align_corners=False)
    work = work.squeeze(1)
    if hard:
        work = (work > 0.5).to(crop.dtype)
    return work


def _per_region_tiles(value: Any, region_count: int, normalize) -> list[torch.Tensor]:
    """Normalize a composite input into one tensor per region, accepting both
    forms produced under INPUT_IS_LIST:

    - native list path: ``region_count`` elements, one tile each;
    - scaled batch path: a single ``[region_count, ...]`` batch tensor wrapped
      in a 1-element list.
    """
    items = list(value) if isinstance(value, list) else [value]
    if len(items) == region_count:
        return [normalize(item) for item in items]
    if len(items) == 1:
        batch = normalize(items[0])
        if batch.shape[0] == region_count:
            return [batch[index : index + 1] for index in range(region_count)]
    raise ValueError(
        f"Expected {region_count} repainted tiles (native list) or one batch of "
        f"{region_count} (scaled), got {len(items)} list item(s)."
    )


class CropImageByRegionsV2:
    """Crop Image By Regions V2 node.

    Crop per-region native square tiles (list) plus a sample_size scaled batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Original image."}),
                "masks": ("MASK", {"tooltip": "Split mask batch used by region_info."}),
                "region_info": ("DICT", {"tooltip": "Region info from Mask Region Planner V2."}),
                "sample_size": (
                    "INT",
                    {
                        "default": 1536,
                        "min": 8,
                        "max": 16384,
                        "step": 8,
                        "tooltip": "Square size of the *_scaled batch outputs (each native "
                        "tile is resized to sample_size x sample_size for batched sampling).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "IMAGE", "MASK", "MASK", "MASK", "DICT")
    RETURN_NAMES = (
        "images",
        "edit_masks",
        "blend_masks",
        "object_masks",
        "images_scaled",
        "edit_masks_scaled",
        "blend_masks_scaled",
        "object_masks_scaled",
        "region_info",
    )
    OUTPUT_IS_LIST = (True, True, True, True, False, False, False, False, False)
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Crop Image By Regions V2"
    DESCRIPTION = "Crop per-region native square tiles (list) plus a sample_size scaled batch."

    def execute(self, image: torch.Tensor, masks: torch.Tensor, region_info: dict, sample_size: int):
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
        boundary_exclude = int(region_info.get("boundary_exclude", 0))
        threshold = float(region_info.get("threshold", 0.5))
        batch, image_height, image_width, _ = image.shape
        if batch != 1:
            raise ValueError("Crop Image By Regions V2 currently supports a single image at a time.")
        if tuple(masks.shape[1:]) != (image_height, image_width):
            raise ValueError("masks and image must have the same spatial size.")

        mask_count = int(masks.shape[0])
        for region in regions:
            for mask_id in region.get("mask_ids", []):
                if not 0 <= int(mask_id) < mask_count:
                    raise ValueError(
                        f"region_info.mask_ids reference mask {int(mask_id)} but only "
                        f"{mask_count} masks were provided (mismatched region_info/masks?)."
                    )

        image_crops: list[torch.Tensor] = []
        edit_crops: list[torch.Tensor] = []
        blend_crops: list[torch.Tensor] = []
        object_crops: list[torch.Tensor] = []

        for region in regions:
            x, y, width, h = (int(value) for value in region["bbox"])
            if x < 0 or y < 0 or x + width > image_width or y + h > image_height:
                raise ValueError(f"Region bbox is outside the image: {region['bbox']}")

            object_mask, edit_mask, blend_mask = _region_masks(
                masks[:, y : y + h, x : x + width],
                [int(mask_id) for mask_id in region["mask_ids"]],
                edit_grow,
                blend_expand,
                blend_feather,
                threshold,
            )
            # _region_masks may return edit/blend aliasing object_mask (e.g.
            # edit_grow=0 -> _max_pool_mask returns the input unchanged). Clone
            # before the in-place border zeroing so it never mutates object_mask
            # or hands out non-owning views of transient buffers.
            edit_mask = edit_mask.clone()
            blend_mask = blend_mask.clone()
            _zero_interior_border(edit_mask, region, image_width, image_height, boundary_exclude)
            _zero_interior_border(blend_mask, region, image_width, image_height, boundary_exclude)

            image_crops.append(image[:, y : y + h, x : x + width, :])
            edit_crops.append(edit_mask.unsqueeze(0).to(device=image.device, dtype=image.dtype))
            blend_crops.append(blend_mask.unsqueeze(0).to(device=image.device, dtype=image.dtype))
            object_crops.append(object_mask.unsqueeze(0).to(device=image.device, dtype=image.dtype))

        channels = int(image.shape[-1])
        size = max(int(sample_size), 1)
        if image_crops:
            images_scaled = torch.cat([_resize_image(c, size) for c in image_crops], dim=0)
            edit_scaled = torch.cat([_resize_mask(c, size, hard=True) for c in edit_crops], dim=0)
            blend_scaled = torch.cat([_resize_mask(c, size, hard=False) for c in blend_crops], dim=0)
            object_scaled = torch.cat([_resize_mask(c, size, hard=True) for c in object_crops], dim=0)
        else:
            images_scaled = torch.zeros((0, size, size, channels), device=image.device, dtype=image.dtype)
            empty_mask = torch.zeros((0, size, size), device=image.device, dtype=image.dtype)
            edit_scaled = empty_mask
            blend_scaled = empty_mask.clone()
            object_scaled = empty_mask.clone()

        if isinstance(region_info, dict):
            region_info["sample_size"] = int(size)

        return (
            image_crops,
            edit_crops,
            blend_crops,
            object_crops,
            images_scaled,
            edit_scaled,
            blend_scaled,
            object_scaled,
            region_info,
        )


class CompositeRepaintRegionsV2:
    """Composite Repaint Regions V2 node.

    Collect repainted native-size tiles (list) and composite back onto the base.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {"tooltip": "Original or VAE-roundtripped base image."}),
                "repainted_regions": (
                    "IMAGE",
                    {"tooltip": "Repainted tiles: native-size list, or a scaled [N,S,S,3] batch."},
                ),
                "blend_masks": ("MASK", {"tooltip": "Blend masks (native list or scaled batch)."}),
                "region_info": ("DICT", {"tooltip": "Region info from Mask Region Planner V2."}),
                "alpha_normalize": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    OUTPUT_NODE = False
    _NODE_NAME = "Composite Repaint Regions V2"
    DESCRIPTION = "Collect repainted native-size tiles (list) and composite back onto the base."

    @staticmethod
    def _first(value):
        return value[0] if isinstance(value, list) else value

    def execute(
        self,
        base_image,
        repainted_regions,
        blend_masks,
        region_info,
        alpha_normalize,
    ):
        base_image = _normalize_image(self._first(base_image))
        region_info = self._first(region_info)
        alpha_normalize = bool(self._first(alpha_normalize))

        regions = region_info.get("regions", []) if isinstance(region_info, dict) else []
        if base_image.shape[0] != 1:
            raise ValueError("Composite Repaint Regions V2 currently supports a single base image at a time.")

        _, image_height, image_width, channels = base_image.shape
        if not regions:
            return (
                base_image,
                torch.zeros((1, image_height, image_width), device=base_image.device, dtype=base_image.dtype),
            )

        repaint_tiles = _per_region_tiles(repainted_regions, len(regions), _normalize_image)
        blend_tiles = _per_region_tiles(blend_masks, len(regions), _normalize_mask)

        repaint_accum = torch.zeros_like(base_image, dtype=torch.float32)
        alpha_accum = torch.zeros(
            (1, image_height, image_width, 1), device=base_image.device, dtype=torch.float32
        )

        for index, region in enumerate(regions):
            x, y, width, height = (int(value) for value in region["bbox"])
            tile = repaint_tiles[index].to(device=base_image.device)
            blend_tile = blend_tiles[index].to(device=base_image.device)
            # Auto-resize back to native when the tile came from the scaled batch.
            if tile.shape[1] != height or tile.shape[2] != width:
                tile = _resize_image(tile[..., :channels], height) if height == width else tile
            if blend_tile.shape[1] != height or blend_tile.shape[2] != width:
                blend_tile = _resize_mask(blend_tile, height, hard=False) if height == width else blend_tile
            crop = tile[0, :height, :width, :channels].to(torch.float32)
            blend = blend_tile[0, :height, :width].to(torch.float32).unsqueeze(-1)
            repaint_accum[0, y : y + height, x : x + width, :] += crop * blend
            alpha_accum[0, y : y + height, x : x + width, :] += blend

        coverage = torch.clamp(alpha_accum, 0.0, 1.0)
        if alpha_normalize:
            repaint_value = repaint_accum / torch.clamp(alpha_accum, min=1e-6)
            result = base_image.to(torch.float32) * (1.0 - coverage) + repaint_value * coverage
        else:
            result = base_image.to(torch.float32) * (1.0 - coverage) + repaint_accum

        return (
            torch.clamp(result, 0.0, 1.0).to(base_image.dtype),
            torch.clamp(alpha_accum[..., 0], 0.0, 1.0).to(base_image.dtype),
        )
