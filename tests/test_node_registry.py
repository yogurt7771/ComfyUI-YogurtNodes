import unittest
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_REGISTRY_PATH = REPO_ROOT / "yogurt_nodes" / "node_registry.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"无法加载模块: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


node_registry = load_module("test_node_registry_module", NODE_REGISTRY_PATH)
build_node_mappings = node_registry.build_node_mappings


def no_wrap(_node_id, node_cls):
    return node_cls


class NodeRegistryTests(unittest.TestCase):
    def test_build_node_mappings_assigns_category_from_package_path(self):
        class SampleImage:
            _NODE_NAME = "Sample Image"

        class SampleIO:
            _NODE_NAME = "Sample IO"

        class SampleLLM:
            _NODE_NAME = "Sample LLM"

        SampleImage.__module__ = "yogurt_nodes.image.sample_image"
        SampleIO.__module__ = "yogurt_nodes.io.sample_io"
        SampleLLM.__module__ = "yogurt_nodes.llm.sample_llm"

        mappings, _display_names, _stats = build_node_mappings(
            {
                "SampleImage": SampleImage,
                "SampleIO": SampleIO,
                "SampleLLM": SampleLLM,
            },
            wrap_llm_node_to_v3=no_wrap,
            wrap_node_to_v3=no_wrap,
        )

        self.assertEqual(mappings["YogurtSampleImage"].CATEGORY, "YogurtNodes/Image")
        self.assertEqual(mappings["YogurtSampleIO"].CATEGORY, "YogurtNodes/IO")
        self.assertEqual(mappings["YogurtSampleLLM"].CATEGORY, "YogurtNodes/LLM")

    def test_build_node_mappings_overwrites_explicit_category_with_package_path(self):
        class LoraScaleWeights:
            _NODE_NAME = "LoRA Scale Weights"
            CATEGORY = "YogurtNodes/Models/LoRA"

        LoraScaleWeights.__module__ = "yogurt_nodes.models.lora_ops"

        mappings, _display_names, _stats = build_node_mappings(
            {"LoraScaleWeights": LoraScaleWeights},
            wrap_llm_node_to_v3=no_wrap,
            wrap_node_to_v3=no_wrap,
        )

        self.assertEqual(
            mappings["YogurtLoraScaleWeights"].CATEGORY,
            "YogurtNodes/Models",
        )


if __name__ == "__main__":
    unittest.main()
