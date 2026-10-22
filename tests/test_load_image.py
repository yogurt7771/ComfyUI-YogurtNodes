import importlib.util
import mimetypes
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOAD_IMAGE_PATH = REPO_ROOT / "yogurt_nodes" / "io" / "load_image.py"


def load_load_image_with_stubs(input_dir):
    folder_paths_module = types.ModuleType("folder_paths")
    folder_paths_module.get_input_directory = lambda: input_dir
    folder_paths_module.filter_files_content_types = lambda files, content_types: [
        f for f in files if (mimetypes.guess_type(f)[0] or "").split("/")[0] in content_types
    ]

    original_modules = {
        name: sys.modules.get(name) for name in ("torch", "folder_paths", "node_helpers")
    }
    sys.modules.update(
        {
            "torch": types.ModuleType("torch"),
            "folder_paths": folder_paths_module,
            "node_helpers": types.ModuleType("node_helpers"),
        }
    )
    try:
        spec = importlib.util.spec_from_file_location("load_image_under_test", LOAD_IMAGE_PATH)
        if spec is None or spec.loader is None:
            raise AssertionError("无法加载 load_image.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class LoadImageOptionsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.input_dir = self._tmp.name
        for rel in ("top.png", "sub/inner.png", "sub/deeper/x.jpg", "note.txt"):
            path = Path(self.input_dir, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        module = load_load_image_with_stubs(self.input_dir)
        self.options = module.LoadImage.INPUT_TYPES()["required"]["image"][0]

    def tearDown(self):
        self._tmp.cleanup()

    def test_subfolder_options_use_forward_slashes(self):
        self.assertEqual(self.options, ["sub/deeper/x.jpg", "sub/inner.png", "top.png"])

    def test_view_route_resolves_each_option(self):
        # Mirror the preview URL: frontend splits subfolder on the last "/",
        # then /view joins subfolder with os.path.basename(filename).
        for option in self.options:
            subfolder, _, filename = option.rpartition("/")
            resolved = os.path.join(self.input_dir, subfolder, os.path.basename(filename))
            self.assertTrue(os.path.isfile(resolved), option)


if __name__ == "__main__":
    unittest.main()
