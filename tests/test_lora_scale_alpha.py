import unittest
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LORA_OPS_PATH = REPO_ROOT / "yogurt_nodes" / "models" / "lora_ops.py"


class FakeTensor:
    def __init__(self, value, dtype="float32", device="cpu"):
        self.value = value
        self.dtype = dtype
        self.device = device
        self.shape = self._shape(value)
        self.ndim = len(self.shape)

    @staticmethod
    def _shape(value):
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                return (len(value), len(value[0]))
            return (len(value),)
        return ()

    def clone(self):
        if isinstance(self.value, list):
            copied = [row[:] if isinstance(row, list) else row for row in self.value]
        else:
            copied = self.value
        return FakeTensor(copied, dtype=self.dtype, device=self.device)

    def item(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def __mul__(self, scalar):
        return FakeTensor(self._multiply_value(self.value, scalar), dtype=self.dtype, device=self.device)

    __rmul__ = __mul__

    @classmethod
    def _multiply_value(cls, value, scalar):
        if isinstance(value, list):
            return [cls._multiply_value(item, scalar) for item in value]
        return value * scalar


def fake_tensor(value, dtype="float32", device="cpu"):
    return FakeTensor(value, dtype=dtype, device=device)


def fake_ones(shape, dtype="float32", device="cpu"):
    rows, cols = shape
    return FakeTensor([[1.0 for _ in range(cols)] for _ in range(rows)], dtype=dtype, device=device)


def fake_equal(left, right):
    return isinstance(left, FakeTensor) and isinstance(right, FakeTensor) and left.value == right.value


def load_lora_ops_with_stubs():
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = FakeTensor
    torch_module.tensor = fake_tensor
    torch_module.ones = fake_ones
    torch_module.equal = fake_equal
    torch_module.float16 = "float16"
    torch_module.float32 = "float32"
    torch_module.device = lambda name: name
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_module.zeros_like = lambda tensor: FakeTensor(
        FakeTensor._multiply_value(tensor.value, 0),
        dtype=tensor.dtype,
        device=tensor.device,
    )

    comfy_module = types.ModuleType("comfy")
    comfy_module.lora_convert = types.SimpleNamespace(convert_lora=lambda lora: lora)
    comfy_module.sd = types.SimpleNamespace(load_lora_for_models=lambda *args, **kwargs: (args[0], args[1]))
    comfy_module.utils = types.SimpleNamespace(ProgressBar=lambda total: types.SimpleNamespace(update_absolute=lambda value: None))

    folder_paths_module = types.ModuleType("folder_paths")
    folder_paths_module.get_filename_list = lambda category: []
    folder_paths_module.get_full_path_or_raise = lambda category, name: name
    folder_paths_module.get_output_directory = lambda: ""

    safetensors_module = types.ModuleType("safetensors")
    safetensors_torch_module = types.ModuleType("safetensors.torch")
    safetensors_torch_module.save_file = lambda tensors, path: None

    original_modules = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "comfy",
            "comfy.lora_convert",
            "comfy.sd",
            "comfy.utils",
            "folder_paths",
            "safetensors",
            "safetensors.torch",
        )
    }
    sys.modules.update(
        {
            "torch": torch_module,
            "comfy": comfy_module,
            "comfy.lora_convert": comfy_module.lora_convert,
            "comfy.sd": comfy_module.sd,
            "comfy.utils": comfy_module.utils,
            "folder_paths": folder_paths_module,
            "safetensors": safetensors_module,
            "safetensors.torch": safetensors_torch_module,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("lora_ops_under_test", LORA_OPS_PATH)
        if spec is None or spec.loader is None:
            raise AssertionError("无法加载 lora_ops.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["lora_ops_under_test"] = module
        spec.loader.exec_module(module)
        return module, torch_module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


lora_ops, torch = load_lora_ops_with_stubs()


class LoraScaleAlphaTests(unittest.TestCase):
    def _node(self):
        node_cls = getattr(lora_ops, "LoraScaleAlpha", None)
        self.assertIsNotNone(node_cls, "LoraScaleAlpha node is not exported")
        return node_cls()

    def test_scales_alpha_tensor_keys_without_mutating_input(self):
        lora = {
            "block.lora_down.weight": torch.ones((2, 4)),
            "block.lora_up.weight": torch.ones((4, 2)) * 3,
            "block.alpha": torch.tensor(4.0, dtype=torch.float16),
            "block.network_alpha": torch.tensor(8.0, dtype=torch.float32),
        }

        scaled, changed = self._node().scale_lora_alpha(lora, 0.5)

        self.assertEqual(changed, 2)
        self.assertTrue(torch.equal(scaled["block.lora_down.weight"], lora["block.lora_down.weight"]))
        self.assertTrue(torch.equal(scaled["block.lora_up.weight"], lora["block.lora_up.weight"]))
        self.assertEqual(float(scaled["block.alpha"]), 2.0)
        self.assertEqual(scaled["block.alpha"].dtype, torch.float16)
        self.assertEqual(float(scaled["block.network_alpha"]), 4.0)
        self.assertEqual(float(lora["block.alpha"]), 4.0)
        self.assertEqual(float(lora["block.network_alpha"]), 8.0)
        self.assertIsNot(scaled["block.alpha"], lora["block.alpha"])
        self.assertIsNot(scaled["block.lora_down.weight"], lora["block.lora_down.weight"])

    def test_key_pattern_limits_scaled_alpha_keys(self):
        lora = {
            "block0.alpha": torch.tensor(2.0),
            "block1.alpha": torch.tensor(4.0),
        }

        scaled, changed = self._node().scale_lora_alpha(lora, 3.0, key_pattern=r"block1")

        self.assertEqual(changed, 1)
        self.assertEqual(float(scaled["block0.alpha"]), 2.0)
        self.assertEqual(float(scaled["block1.alpha"]), 12.0)

    def test_scalar_alpha_is_converted_to_tensor_for_saving(self):
        lora = {
            "block.lora_down.weight": torch.ones((2, 4)),
            "block.alpha": 2.0,
        }

        scaled, changed = self._node().scale_lora_alpha(lora, 3.0)

        self.assertEqual(changed, 1)
        self.assertIsInstance(scaled["block.alpha"], FakeTensor)
        self.assertEqual(float(scaled["block.alpha"]), 6.0)

    def test_creates_alpha_for_standard_pairs_without_existing_alpha(self):
        lora = {
            "block.lora_A.weight": torch.ones((8, 4)),
            "block.lora_B.weight": torch.ones((4, 8)),
        }

        scaled, changed = self._node().scale_lora_alpha(lora, 0.1)

        self.assertEqual(changed, 1)
        self.assertIn("block.alpha", scaled)
        self.assertIsInstance(scaled["block.alpha"], FakeTensor)
        self.assertEqual(float(scaled["block.alpha"]), 0.8)

    def test_returns_clone_and_zero_count_when_no_alpha_keys_match(self):
        lora = {"block.lora_down.weight": torch.ones((2, 4))}

        scaled, changed = self._node().scale_lora_alpha(lora, 2.0)

        self.assertEqual(changed, 0)
        self.assertTrue(torch.equal(scaled["block.lora_down.weight"], lora["block.lora_down.weight"]))
        self.assertIsNot(scaled["block.lora_down.weight"], lora["block.lora_down.weight"])


if __name__ == "__main__":
    unittest.main()
