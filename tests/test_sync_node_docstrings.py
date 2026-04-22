import ast
import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_PATH = REPO_ROOT / "tools" / "sync_node_docstrings.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"无法加载模块: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class SyncNodeDocstringsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("tools.sync_node_docstrings", TOOLS_PATH)

    def test_sync_package_updates_explicit_and_star_exports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root = temp_root / "yogurt_nodes"
            image_dir = package_root / "image"
            net_dir = package_root / "net"
            image_dir.mkdir(parents=True)
            net_dir.mkdir(parents=True)

            (image_dir / "__init__.py").write_text(
                "from .sample_image import AddTextToImage\n",
                encoding="utf-8",
            )
            (net_dir / "__init__.py").write_text(
                "from .sample_net import *\n",
                encoding="utf-8",
            )

            (image_dir / "sample_image.py").write_text(
                textwrap.dedent(
                    """
                    class AddTextToImage:
                        @classmethod
                        def INPUT_TYPES(cls):
                            return {}

                        _NODE_NAME = "Add Text To Image"
                        DESCRIPTION = "Add text to image."
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            (net_dir / "sample_net.py").write_text(
                textwrap.dedent(
                    """
                    class ComfyUIClientLoad:

                        @classmethod
                        def INPUT_TYPES(cls):
                            return {}

                        _NODE_NAME = "ComfyUI Client Load"
                        DESCRIPTION = "配置 ComfyUI 客户端实例，供后续节点复用"
                    """
                ).lstrip(),
                encoding="utf-8",
            )

            changes = self.module.sync_package_docstrings(package_root, write=True)

            self.assertEqual(len(changes), 2)

            image_source = (image_dir / "sample_image.py").read_text(encoding="utf-8")
            self.assertIn('class AddTextToImage:\n    """Add Text To Image node."""\n', image_source)
            self.assertIn('    @classmethod\n    def INPUT_TYPES', image_source)

            net_source = (net_dir / "sample_net.py").read_text(encoding="utf-8")
            self.assertIn(
                'class ComfyUIClientLoad:\n\n    """ComfyUI Client Load node.\n',
                net_source,
            )
            self.assertIn("配置 ComfyUI 客户端实例，供后续节点复用", net_source)

            ast.parse(image_source)
            ast.parse(net_source)

    def test_check_mode_reports_changes_without_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root = temp_root / "yogurt_nodes"
            logic_dir = package_root / "logic"
            logic_dir.mkdir(parents=True)

            (logic_dir / "__init__.py").write_text(
                "from .sample_logic import Switch\n",
                encoding="utf-8",
            )
            original_source = textwrap.dedent(
                """
                class Switch:
                    @classmethod
                    def INPUT_TYPES(cls):
                        return {}

                    _NODE_NAME = "Switch"
                    DESCRIPTION = "Switch"
                """
            ).lstrip()
            sample_path = logic_dir / "sample_logic.py"
            sample_path.write_text(original_source, encoding="utf-8")

            changes = self.module.sync_package_docstrings(package_root, write=False)

            self.assertEqual(len(changes), 1)
            self.assertEqual(sample_path.read_text(encoding="utf-8"), original_source)


if __name__ == "__main__":
    unittest.main()
