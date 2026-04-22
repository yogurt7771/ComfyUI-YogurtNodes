import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
GENERATE_README_PATH = TOOLS_DIR / "generate_readme.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"无法加载模块: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class GenerateReadmeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(TOOLS_DIR))
        cls.module = load_module("tools.generate_readme", GENERATE_README_PATH)

    def test_update_readmes_renders_all_exported_nodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            package_root = repo_root / "yogurt_nodes"
            image_dir = package_root / "image"
            models_dir = package_root / "models"
            image_dir.mkdir(parents=True)
            models_dir.mkdir(parents=True)

            (image_dir / "__init__.py").write_text(
                "from .sample_image import AddTextToImage\n",
                encoding="utf-8",
            )
            (models_dir / "__init__.py").write_text(
                "from .sample_lora import *\n",
                encoding="utf-8",
            )

            (image_dir / "sample_image.py").write_text(
                textwrap.dedent(
                    """
                    class AddTextToImage:
                        \"\"\"Add Text To Image node.

                        Add text overlay to an image.
                        \"\"\"
                        _NODE_NAME = "Add Text To Image"
                        CATEGORY = "YogurtNodes/Image"
                        DESCRIPTION = "Add text overlay to an image."
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (models_dir / "sample_lora.py").write_text(
                textwrap.dedent(
                    """
                    class LoraScaleWeights:
                        \"\"\"LoRA Scale Weights node.

                        Scale LoRA tensor weights globally.
                        \"\"\"
                        _NODE_NAME = "LoRA Scale Weights"
                        CATEGORY = "YogurtNodes/Models/LoRA"
                        DESCRIPTION = "Scale LoRA tensor weights globally."
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            (repo_root / "README.md").write_text(
                textwrap.dedent(
                    """
                    # Demo

                    ## 🔧 Available Nodes
                    old

                    ## 🔑 Gemini API Key Setup
                    tail
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (repo_root / "README_zh.md").write_text(
                textwrap.dedent(
                    """
                    # 演示

                    ## 🔧 可用节点
                    old

                    ## 🔑 Gemini API Key 配置说明
                    tail
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            updated = self.module.update_readmes(repo_root, package_root, write=True)

            self.assertEqual(
                {path.name for path in updated},
                {"README.md", "README_zh.md"},
            )

            readme_en = (repo_root / "README.md").read_text(encoding="utf-8")
            self.assertIn("Total exported nodes: **2**.", readme_en)
            self.assertIn("### Image Processing Nodes", readme_en)
            self.assertIn("### Model Nodes", readme_en)
            self.assertIn("`YogurtAddTextToImage`", readme_en)
            self.assertIn("`YogurtLoraScaleWeights`", readme_en)
            self.assertIn("`YogurtNodes/Models/LoRA`", readme_en)
            self.assertIn("## 🔑 Gemini API Key Setup", readme_en)

            readme_zh = (repo_root / "README_zh.md").read_text(encoding="utf-8")
            self.assertIn("当前导出节点总数：**2**。", readme_zh)
            self.assertIn("### 图像处理节点", readme_zh)
            self.assertIn("### 模型节点", readme_zh)
            self.assertIn("## 🔑 Gemini API Key 配置说明", readme_zh)

    def test_check_mode_reports_changes_without_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            package_root = repo_root / "yogurt_nodes"
            string_dir = package_root / "string"
            string_dir.mkdir(parents=True)

            (string_dir / "__init__.py").write_text(
                "from .sample_string import StringFormat\n",
                encoding="utf-8",
            )
            (string_dir / "sample_string.py").write_text(
                textwrap.dedent(
                    """
                    class StringFormat:
                        _NODE_NAME = "String Format"
                        CATEGORY = "YogurtNodes/String"
                        DESCRIPTION = "Format strings"
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            original_en = textwrap.dedent(
                """
                # Demo

                ## 🔧 Available Nodes
                old

                ## 🔑 Gemini API Key Setup
                tail
                """
            ).lstrip()
            original_zh = textwrap.dedent(
                """
                # 演示

                ## 🔧 可用节点
                old

                ## 🔑 Gemini API Key 配置说明
                tail
                """
            ).lstrip()

            (repo_root / "README.md").write_text(original_en, encoding="utf-8")
            (repo_root / "README_zh.md").write_text(original_zh, encoding="utf-8")

            updated = self.module.update_readmes(repo_root, package_root, write=False)

            self.assertEqual(
                {path.name for path in updated},
                {"README.md", "README_zh.md"},
            )
            self.assertEqual((repo_root / "README.md").read_text(encoding="utf-8"), original_en)
            self.assertEqual((repo_root / "README_zh.md").read_text(encoding="utf-8"), original_zh)


if __name__ == "__main__":
    unittest.main()
