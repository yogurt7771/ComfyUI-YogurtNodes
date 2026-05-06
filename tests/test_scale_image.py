import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SCALE_IMAGE_PATH = REPO_ROOT / "yogurt_nodes" / "image" / "scale_image.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"无法加载模块: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prepare_comfy_stub():
    comfy_pkg = types.ModuleType("comfy")
    utils_mod = types.ModuleType("comfy.utils")

    def common_upscale(samples, width, height, upscale_method, crop):
        mode = "nearest-exact" if upscale_method == "nearest-exact" else "bilinear"
        kwargs = {}
        if mode != "nearest-exact":
            kwargs["align_corners"] = False
        return F.interpolate(samples, size=(height, width), mode=mode, **kwargs)

    utils_mod.common_upscale = common_upscale
    comfy_pkg.utils = utils_mod
    sys.modules["comfy"] = comfy_pkg
    sys.modules["comfy.utils"] = utils_mod


class ImageScaleToTotalPixelsAdvancedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_comfy_stub()
        cls.module = load_module("scale_image_under_test", SCALE_IMAGE_PATH)
        cls.node_class = cls.module.ImageScaleToTotalPixelsAdvanced

    def test_declares_scaled_image_metadata_outputs(self):
        self.assertEqual(
            self.node_class.RETURN_TYPES,
            (
                "IMAGE",
                "MASK",
                "INT",
                "INT",
                "INT",
                "INT",
                "INT",
                "FLOAT",
                "INT",
            ),
        )
        self.assertEqual(
            self.node_class.RETURN_NAMES,
            (
                "image",
                "mask",
                "width",
                "height",
                "channels",
                "longest_side",
                "shortest_side",
                "aspect_ratio",
                "pixels",
            ),
        )

    def test_execute_returns_scaled_image_metadata(self):
        image = torch.zeros((1, 3, 5, 3), dtype=torch.float32)

        result = self.node_class().execute(
            image,
            upscale_method="bilinear",
            megapixels=0.0001,
            divide_by=1,
            pad_value="#000000",
        )

        (
            image_out,
            mask_out,
            width,
            height,
            channels,
            longest_side,
            shortest_side,
            aspect_ratio,
            pixels,
        ) = result

        self.assertEqual(tuple(image_out.shape), (1, 8, 13, 3))
        self.assertEqual(tuple(mask_out.shape), (1, 1, 8, 13))
        self.assertEqual(width, 13)
        self.assertEqual(height, 8)
        self.assertEqual(channels, 3)
        self.assertEqual(longest_side, 13)
        self.assertEqual(shortest_side, 8)
        self.assertEqual(aspect_ratio, 1.625)
        self.assertEqual(pixels, 104)


if __name__ == "__main__":
    unittest.main()
