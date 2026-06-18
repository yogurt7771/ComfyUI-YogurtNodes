import importlib.util
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = REPO_ROOT / "yogurt_nodes" / "image" / "hl_frequency_detail_restore_threshold.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load module: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class HLFrequencyDetailRestoreThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("hl_frequency_detail_restore_threshold_under_test", NODE_PATH)
        cls.node_class = cls.module.HLFrequencyDetailRestoreThreshold

    def test_declares_threshold_inputs(self):
        inputs = self.node_class.INPUT_TYPES()["required"]

        self.assertIn("high_threshold", inputs)
        self.assertIn("low_threshold", inputs)
        self.assertEqual(inputs["high_threshold"][0], "FLOAT")
        self.assertEqual(inputs["low_threshold"][0], "FLOAT")

    def test_high_threshold_suppresses_detail_delta(self):
        image = torch.full((1, 7, 7, 3), 0.5, dtype=torch.float32)
        detail = image.clone()
        detail[:, 3, 3, :] = 0.0

        without_threshold = self.node_class().execute(
            image,
            detail,
            keep_high_freq=2,
            erase_low_freq=0,
            mask_blur=0,
            high_threshold=0.0,
            low_threshold=0.0,
        )[0]
        with_threshold = self.node_class().execute(
            image,
            detail,
            keep_high_freq=2,
            erase_low_freq=0,
            mask_blur=0,
            high_threshold=1.0,
            low_threshold=0.0,
        )[0]

        self.assertFalse(torch.allclose(without_threshold, image))
        self.assertTrue(torch.allclose(with_threshold, image, atol=1 / 255))

    def test_low_threshold_suppresses_background_delta(self):
        image = torch.full((1, 7, 7, 3), 0.5, dtype=torch.float32)
        image[:, 3, 3, :] = 0.0
        detail = torch.full((1, 7, 7, 3), 0.5, dtype=torch.float32)

        without_threshold = self.node_class().execute(
            image,
            detail,
            keep_high_freq=2,
            erase_low_freq=2,
            mask_blur=0,
            high_threshold=0.0,
            low_threshold=0.0,
        )[0]
        with_threshold = self.node_class().execute(
            image,
            detail,
            keep_high_freq=2,
            erase_low_freq=2,
            mask_blur=0,
            high_threshold=0.0,
            low_threshold=1.0,
        )[0]

        self.assertFalse(torch.allclose(without_threshold, image))
        self.assertTrue(torch.allclose(with_threshold, image, atol=1 / 255))


if __name__ == "__main__":
    unittest.main()
