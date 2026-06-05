import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MASKS_PATH = REPO_ROOT / "yogurt_nodes" / "masks" / "region_repaint.py"


def load_module(module_name: str, path: Path):
    if not path.exists():
        raise AssertionError(f"Missing module file: {path.relative_to(REPO_ROOT)}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MaskRegionRepaintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("mask_region_repaint_under_test", MASKS_PATH)

    def assert_region_contains_safe_bbox(self, region):
        x, y, width, height = region["bbox"]
        safe_x, safe_y, safe_width, safe_height = region["safe_bbox"]
        self.assertLessEqual(x, safe_x)
        self.assertLessEqual(y, safe_y)
        self.assertGreaterEqual(x + width, safe_x + safe_width)
        self.assertGreaterEqual(y + height, safe_y + safe_height)

    def test_split_mask_returns_one_mask_per_connected_component(self):
        mask = torch.zeros((1, 8, 10), dtype=torch.float32)
        mask[0, 1:3, 1:3] = 1.0
        mask[0, 4:7, 6:9] = 1.0

        split_masks, mask_info = self.module.SplitMask().execute(
            mask,
            threshold=0.5,
            min_area=1,
            connectivity="8",
            close_radius=0,
            mode="split_components",
        )

        self.assertEqual(tuple(split_masks.shape), (2, 8, 10))
        self.assertEqual(mask_info["count"], 2)
        self.assertEqual(mask_info["items"][0]["bbox"], [1, 1, 2, 2])
        self.assertEqual(mask_info["items"][1]["bbox"], [6, 4, 3, 3])
        self.assertEqual(float(split_masks[0].sum()), 4.0)
        self.assertEqual(float(split_masks[1].sum()), 9.0)

    def test_mask_region_planner_defaults_match_local_repaint_workflow(self):
        inputs = self.module.MaskRegionPlanner.INPUT_TYPES()["required"]

        self.assertEqual(inputs["tile_width"][1]["default"], 512)
        self.assertEqual(inputs["tile_height"][1]["default"], 512)
        self.assertEqual(inputs["context_padding"][1]["default"], 32)
        self.assertEqual(inputs["edit_grow"][1]["default"], 8)
        self.assertEqual(inputs["blend_expand"][1]["default"], 8)
        self.assertEqual(inputs["blend_feather"][1]["default"], 8)
        self.assertEqual(inputs["multiple_of"][1]["default"], 8)
        self.assertEqual(inputs["merge_distance"][1]["default"], 64)
        self.assertEqual(inputs["max_waste_ratio"][1]["default"], 1.0)
        self.assertEqual(inputs["threshold"][1]["default"], 0.5)

    def test_merge_distance_changes_merge_order_score(self):
        planner = self.module.MaskRegionPlanner()
        left = {
            "mask_ids": [0],
            "object_bbox": [0, 0, 2, 2],
            "safe_bbox": [0, 0, 2, 2],
            "preferred_bbox": [0, 0, 2, 2],
            "bbox": [0, 0, 10, 10],
            "mask_area": 4,
            "safe_area": 4,
        }
        right = {
            "mask_ids": [1],
            "object_bbox": [6, 0, 2, 2],
            "safe_bbox": [6, 0, 2, 2],
            "preferred_bbox": [6, 0, 2, 2],
            "bbox": [6, 0, 10, 10],
            "mask_area": 4,
            "safe_area": 4,
        }

        _, score_with_small_distance = planner._try_merge(
            left, right, 32, 16, 10, 10, 1, 0, 1.0
        )
        _, score_with_large_distance = planner._try_merge(
            left, right, 32, 16, 10, 10, 1, 4, 1.0
        )

        self.assertLess(score_with_large_distance, score_with_small_distance)

    def test_region_planner_merges_near_masks_into_one_dynamic_tile(self):
        image = torch.zeros((1, 12, 16, 3), dtype=torch.float32)
        masks = torch.zeros((2, 12, 16), dtype=torch.float32)
        masks[0, 2:4, 2:4] = 1.0
        masks[1, 3:5, 6:8] = 1.0

        region_info, preview = self.module.MaskRegionPlanner().execute(
            image,
            masks,
            tile_width=8,
            tile_height=8,
            context_padding=0,
            edit_grow=0,
            blend_expand=0,
            blend_feather=0,
            multiple_of=1,
            merge_distance=8,
            max_waste_ratio=1.0,
            threshold=0.5,
        )

        self.assertEqual(region_info["count"], 1)
        region = region_info["regions"][0]
        self.assertEqual(region["mask_ids"], [0, 1])
        self.assertEqual(region["bbox"][2:], [8, 8])
        self.assertEqual(tuple(preview.shape), (1, 12, 16, 3))

    def test_region_planner_keeps_far_masks_in_separate_dynamic_tiles(self):
        image = torch.zeros((1, 12, 24, 3), dtype=torch.float32)
        masks = torch.zeros((2, 12, 24), dtype=torch.float32)
        masks[0, 2:4, 1:3] = 1.0
        masks[1, 2:4, 20:22] = 1.0

        region_info, _ = self.module.MaskRegionPlanner().execute(
            image,
            masks,
            tile_width=8,
            tile_height=8,
            context_padding=0,
            edit_grow=0,
            blend_expand=0,
            blend_feather=0,
            multiple_of=1,
            merge_distance=4,
            max_waste_ratio=1.0,
            threshold=0.5,
        )

        self.assertEqual(region_info["count"], 2)
        self.assertEqual(region_info["regions"][0]["mask_ids"], [0])
        self.assertEqual(region_info["regions"][1]["mask_ids"], [1])
        self.assertNotEqual(
            region_info["regions"][0]["bbox"][0],
            region_info["regions"][1]["bbox"][0],
        )

    def test_crop_and_composite_only_changes_blend_mask_area(self):
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        image[..., 1] = 0.25
        masks = torch.zeros((1, 8, 8), dtype=torch.float32)
        masks[0, 2:4, 2:4] = 1.0

        region_info, _ = self.module.MaskRegionPlanner().execute(
            image,
            masks,
            tile_width=6,
            tile_height=6,
            context_padding=0,
            edit_grow=0,
            blend_expand=0,
            blend_feather=0,
            multiple_of=1,
            merge_distance=0,
            max_waste_ratio=0.0,
            threshold=0.5,
        )
        crops, edit_masks, blend_masks, object_masks, returned_info = (
            self.module.CropImageByRegions().execute(image, masks, region_info)
        )

        repainted = torch.ones_like(crops)
        result, alpha_debug = self.module.CompositeRepaintRegions().execute(
            image,
            repainted,
            blend_masks,
            returned_info,
            alpha_normalize=True,
        )

        self.assertEqual(tuple(edit_masks.shape), (1, 6, 6))
        self.assertEqual(tuple(object_masks.shape), (1, 6, 6))
        self.assertEqual(tuple(alpha_debug.shape), (1, 8, 8))
        self.assertTrue(torch.allclose(result[0, 2:4, 2:4], torch.ones((2, 2, 3))))
        self.assertTrue(torch.allclose(result[0, 0, 0], image[0, 0, 0]))

    def test_region_planner_rejects_multi_image_batches(self):
        image = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
        masks = torch.zeros((1, 8, 8), dtype=torch.float32)
        masks[0, 2:4, 2:4] = 1.0

        with self.assertRaisesRegex(ValueError, "single image"):
            self.module.MaskRegionPlanner().execute(
                image,
                masks,
                tile_width=6,
                tile_height=6,
                context_padding=0,
                edit_grow=0,
                blend_expand=0,
                blend_feather=0,
                multiple_of=1,
                merge_distance=0,
                max_waste_ratio=1.0,
                threshold=0.5,
            )

    def test_region_planner_rejects_oversized_safe_region(self):
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        masks = torch.zeros((1, 8, 8), dtype=torch.float32)
        masks[0, 1:7, 1:7] = 1.0

        with self.assertRaisesRegex(ValueError, "exceeds tile size"):
            self.module.MaskRegionPlanner().execute(
                image,
                masks,
                tile_width=4,
                tile_height=4,
                context_padding=0,
                edit_grow=0,
                blend_expand=0,
                blend_feather=0,
                multiple_of=1,
                merge_distance=0,
                max_waste_ratio=1.0,
                threshold=0.5,
            )

    def test_region_planner_rejects_mask_image_size_mismatch(self):
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        masks = torch.zeros((1, 6, 8), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "same spatial size"):
            self.module.MaskRegionPlanner().execute(
                image,
                masks,
                tile_width=6,
                tile_height=6,
                context_padding=0,
                edit_grow=0,
                blend_expand=0,
                blend_feather=0,
                multiple_of=1,
                merge_distance=0,
                max_waste_ratio=1.0,
                threshold=0.5,
            )

    def test_region_planner_uses_multiple_aligned_tile_size_for_merge(self):
        image = torch.zeros((1, 8, 16, 3), dtype=torch.float32)
        masks = torch.zeros((2, 8, 16), dtype=torch.float32)
        masks[0, 2:4, 1:3] = 1.0
        masks[1, 2:4, 6:8] = 1.0

        region_info, _ = self.module.MaskRegionPlanner().execute(
            image,
            masks,
            tile_width=7,
            tile_height=7,
            context_padding=0,
            edit_grow=0,
            blend_expand=0,
            blend_feather=0,
            multiple_of=8,
            merge_distance=8,
            max_waste_ratio=1.0,
            threshold=0.5,
        )

        self.assertEqual(region_info["count"], 1)
        self.assertEqual(region_info["regions"][0]["bbox"][2:], [8, 8])

    def test_region_planner_merges_when_fixed_tiles_overlap_heavily(self):
        image = torch.zeros((1, 10, 20, 3), dtype=torch.float32)
        masks = torch.zeros((2, 10, 20), dtype=torch.float32)
        masks[0, 3:5, 3:5] = 1.0
        masks[1, 3:5, 9:11] = 1.0

        region_info, _ = self.module.MaskRegionPlanner().execute(
            image,
            masks,
            tile_width=10,
            tile_height=10,
            context_padding=0,
            edit_grow=0,
            blend_expand=0,
            blend_feather=0,
            multiple_of=1,
            merge_distance=0,
            max_waste_ratio=1.0,
            threshold=0.5,
        )

        self.assertEqual(region_info["count"], 1)
        self.assertEqual(region_info["regions"][0]["mask_ids"], [0, 1])

    def test_region_planner_treats_context_padding_as_preferred_not_required(self):
        image = torch.zeros((1, 16, 32, 3), dtype=torch.float32)
        masks = torch.zeros((2, 16, 32), dtype=torch.float32)
        masks[0, 6:8, 6:8] = 1.0
        masks[1, 6:8, 18:20] = 1.0

        region_info, _ = self.module.MaskRegionPlanner().execute(
            image,
            masks,
            tile_width=16,
            tile_height=16,
            context_padding=8,
            edit_grow=0,
            blend_expand=0,
            blend_feather=0,
            multiple_of=1,
            merge_distance=0,
            max_waste_ratio=1.0,
            threshold=0.5,
        )

        self.assertEqual(region_info["count"], 1)
        self.assertEqual(region_info["regions"][0]["mask_ids"], [0, 1])

    def test_region_planner_moves_tile_to_common_position_when_merging(self):
        image = torch.zeros((1, 16, 40, 3), dtype=torch.float32)
        masks = torch.zeros((2, 16, 40), dtype=torch.float32)
        masks[0, 6:8, 10:12] = 1.0
        masks[1, 6:8, 24:26] = 1.0

        region_info, _ = self.module.MaskRegionPlanner().execute(
            image,
            masks,
            tile_width=16,
            tile_height=16,
            context_padding=0,
            edit_grow=0,
            blend_expand=0,
            blend_feather=0,
            multiple_of=1,
            merge_distance=0,
            max_waste_ratio=1.0,
            threshold=0.5,
        )

        self.assertEqual(region_info["count"], 1)
        region = region_info["regions"][0]
        self.assertEqual(region["mask_ids"], [0, 1])
        self.assert_region_contains_safe_bbox(region)

    def test_region_planner_spreads_non_mergeable_fixed_tiles(self):
        image = torch.zeros((1, 10, 24, 3), dtype=torch.float32)
        masks = torch.zeros((2, 10, 24), dtype=torch.float32)
        masks[0, 3:5, 2:4] = 1.0
        masks[1, 3:5, 11:13] = 1.0

        region_info, _ = self.module.MaskRegionPlanner().execute(
            image,
            masks,
            tile_width=10,
            tile_height=10,
            context_padding=0,
            edit_grow=0,
            blend_expand=0,
            blend_feather=0,
            multiple_of=1,
            merge_distance=0,
            max_waste_ratio=1.0,
            threshold=0.5,
        )

        self.assertEqual(region_info["count"], 2)
        first, second = region_info["regions"]
        self.assertLessEqual(first["bbox"][0] + first["bbox"][2], second["bbox"][0])
        self.assert_region_contains_safe_bbox(first)
        self.assert_region_contains_safe_bbox(second)

    def test_crop_builds_blend_masks_inside_region_not_full_image(self):
        image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        masks = torch.zeros((1, 64, 64), dtype=torch.float32)
        masks[0, 22:26, 22:26] = 1.0
        region_info = {
            "regions": [{"id": 0, "mask_ids": [0], "bbox": [16, 16, 16, 16]}],
            "edit_grow": 0,
            "blend_expand": 0,
            "blend_feather": 4,
            "threshold": 0.5,
        }
        seen_shapes = []

        def fake_blur(mask, radius):
            seen_shapes.append(tuple(mask.shape))
            return mask

        with mock.patch.object(self.module, "_gaussian_blur_mask", fake_blur):
            self.module.CropImageByRegions().execute(image, masks, region_info)

        self.assertEqual(seen_shapes, [(1, 16, 16)])


if __name__ == "__main__":
    unittest.main()
