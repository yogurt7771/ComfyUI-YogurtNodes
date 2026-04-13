import importlib.util
import sys
import types
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
YOGURT_DIR = REPO_ROOT / "yogurt_nodes"
UTILS_DIR = YOGURT_DIR / "utils"
LLM_DIR = YOGURT_DIR / "llm"


def ensure_package(name: str, path: Path):
    package = sys.modules.get(name)
    if package is None:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = package
    return package


def load_module(module_name: str, path: Path):
    if not path.exists():
        raise AssertionError(f"缺少模块文件: {path.relative_to(REPO_ROOT)}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"无法加载模块: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prepare_import_environment():
    ensure_package("yogurt_nodes", YOGURT_DIR)
    ensure_package("yogurt_nodes.utils", UTILS_DIR)
    ensure_package("yogurt_nodes.llm", LLM_DIR)

    comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is None:
        comfy_pkg = types.ModuleType("comfy")
        comfy_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["comfy"] = comfy_pkg

    model_management = types.ModuleType("comfy.model_management")
    model_management.throw_exception_if_processing_interrupted = lambda: None
    sys.modules["comfy.model_management"] = model_management
    setattr(comfy_pkg, "model_management", model_management)

    load_module("yogurt_nodes.utils.api_keys", UTILS_DIR / "api_keys.py")


def make_test_image(color: str = "white") -> Image.Image:
    return Image.new("RGB", (8, 8), color=color)


class DashScopeImageClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_import_environment()
        cls.qwen_client_module = load_module(
            "yogurt_nodes.utils.qwen_client",
            UTILS_DIR / "qwen_client.py",
        )
        cls.wan_client_module = load_module(
            "yogurt_nodes.utils.wan_client",
            UTILS_DIR / "wan_client.py",
        )

        utils_pkg = sys.modules["yogurt_nodes.utils"]
        setattr(utils_pkg, "QwenClient", cls.qwen_client_module.QwenClient)
        setattr(utils_pkg, "WanClient", cls.wan_client_module.WanClient)

        cls.qwen_node_module = load_module(
            "yogurt_nodes.llm.qwen_generate_image",
            LLM_DIR / "qwen_generate_image.py",
        )
        cls.wan_node_module = load_module(
            "yogurt_nodes.llm.wan_generate_image",
            LLM_DIR / "wan_generate_image.py",
        )

    def test_build_qwen_request_for_text_only_generation(self):
        build_qwen_request = getattr(
            self.qwen_client_module,
            "build_qwen_request",
            None,
        )
        self.assertIsNotNone(build_qwen_request, "缺少 build_qwen_request()")

        request_body = build_qwen_request(
            model_name="qwen-image-2.0-pro",
            prompt="画一只猫",
            system_prompt="高质量插画风格",
            images=[],
            size="1024*1024",
            n=2,
            negative_prompt="低质量",
            prompt_extend=True,
            watermark=False,
            seed=42,
            extra={"custom_flag": True},
        )

        self.assertEqual(request_body["model"], "qwen-image-2.0-pro")
        content = request_body["input"]["messages"][0]["content"]
        self.assertEqual(content, [{"text": "高质量插画风格\n\n画一只猫"}])
        self.assertEqual(request_body["parameters"]["n"], 2)
        self.assertEqual(request_body["parameters"]["size"], "1024*1024")
        self.assertEqual(request_body["parameters"]["seed"], 42)
        self.assertTrue(request_body["parameters"]["prompt_extend"])
        self.assertFalse(request_body["parameters"]["watermark"])
        self.assertEqual(request_body["parameters"]["negative_prompt"], "低质量")
        self.assertTrue(request_body["custom_flag"])

    def test_build_qwen_request_rejects_more_than_three_images(self):
        build_qwen_request = getattr(
            self.qwen_client_module,
            "build_qwen_request",
            None,
        )
        self.assertIsNotNone(build_qwen_request, "缺少 build_qwen_request()")

        with self.assertRaisesRegex(ValueError, "最多支持 3 张输入图像"):
            build_qwen_request(
                model_name="qwen-image-2.0-pro",
                prompt="把图一角色放到图四场景里",
                system_prompt="",
                images=[
                    make_test_image("red"),
                    make_test_image("green"),
                    make_test_image("blue"),
                    make_test_image("black"),
                ],
                size="auto",
                n=1,
                negative_prompt="",
                prompt_extend=True,
                watermark=False,
                seed=-1,
                extra={},
            )

    def test_build_wan_request_merges_known_parameter_overrides(self):
        build_wan_request = getattr(
            self.wan_client_module,
            "build_wan_request",
            None,
        )
        self.assertIsNotNone(build_wan_request, "缺少 build_wan_request()")

        request_body = build_wan_request(
            model_name="wan2.7-image",
            prompt="赛博朋克城市夜景",
            system_prompt="电影级光影",
            images=[make_test_image("purple")],
            size="2K",
            n=3,
            negative_prompt="模糊",
            watermark=False,
            seed=7,
            extra={
                "enable_sequential": False,
                "bbox_list": [[[0, 0, 4, 4]]],
                "thinking_mode": True,
            },
        )

        self.assertEqual(request_body["model"], "wan2.7-image")
        content = request_body["input"]["messages"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertTrue(content[0]["image"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[1]["text"], "电影级光影\n\n赛博朋克城市夜景")
        self.assertEqual(request_body["parameters"]["n"], 3)
        self.assertEqual(request_body["parameters"]["size"], "2K")
        self.assertEqual(request_body["parameters"]["seed"], 7)
        self.assertEqual(request_body["parameters"]["negative_prompt"], "模糊")
        self.assertFalse(request_body["parameters"]["watermark"])
        self.assertFalse(request_body["parameters"]["enable_sequential"])
        self.assertEqual(request_body["parameters"]["bbox_list"], [[[0, 0, 4, 4]]])
        self.assertTrue(request_body["parameters"]["thinking_mode"])

    def test_extract_wan_image_urls_does_not_require_exact_image_type(self):
        extract_wan_image_urls = getattr(
            self.wan_client_module,
            "extract_wan_image_urls",
            None,
        )
        self.assertIsNotNone(
            extract_wan_image_urls,
            "缺少 extract_wan_image_urls()",
        )

        urls = extract_wan_image_urls(
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "output_image", "image": "https://a.example/1.png"},
                                    {"type": "text", "text": "ignored"},
                                ]
                            }
                        }
                    ]
                }
            }
        )

        self.assertEqual(urls, ["https://a.example/1.png"])

    def test_extract_request_prompt_falls_back_when_messages_are_overridden(self):
        extract_request_prompt = getattr(
            self.qwen_client_module,
            "extract_request_prompt",
            None,
        )
        self.assertIsNotNone(
            extract_request_prompt,
            "缺少 extract_request_prompt()",
        )

        prompt = extract_request_prompt(
            {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"image": "data:image/jpeg;base64,abc"}],
                        }
                    ]
                }
            },
            fallback_prompt="fallback prompt",
        )

        self.assertEqual(prompt, "fallback prompt")

    def test_retryable_status_code_only_retries_transient_errors(self):
        is_retryable_status = getattr(
            self.qwen_client_module,
            "is_retryable_status",
            None,
        )
        self.assertIsNotNone(is_retryable_status, "缺少 is_retryable_status()")

        self.assertTrue(is_retryable_status(429))
        self.assertTrue(is_retryable_status(503))
        self.assertFalse(is_retryable_status(400))
        self.assertFalse(is_retryable_status(401))

    def test_qwen_node_shape_matches_existing_llm_image_nodes(self):
        node_class = getattr(self.qwen_node_module, "QwenGenerateImage", None)
        self.assertIsNotNone(node_class, "缺少 QwenGenerateImage 节点类")

        input_types = node_class.INPUT_TYPES()
        self.assertIn("api_key", input_types["required"])
        self.assertIn("model_name", input_types["required"])
        self.assertIn("prompt", input_types["required"])
        self.assertIn("history", input_types["optional"])
        self.assertIn("extra", input_types["optional"])
        self.assertIn("image", input_types["optional"])
        self.assertIn("image1", input_types["optional"])
        self.assertIn("image2", input_types["optional"])
        self.assertIn("image3", input_types["optional"])
        self.assertIn("image4", input_types["optional"])
        self.assertEqual(node_class.RETURN_TYPES, ("IMAGE", "STRING", "HISTORY"))
        self.assertEqual(node_class.FUNCTION, "generate_image")

    def test_wan_node_shape_matches_existing_llm_image_nodes(self):
        node_class = getattr(self.wan_node_module, "WanGenerateImage", None)
        self.assertIsNotNone(node_class, "缺少 WanGenerateImage 节点类")

        input_types = node_class.INPUT_TYPES()
        self.assertIn("api_key", input_types["required"])
        self.assertIn("model_name", input_types["required"])
        self.assertIn("prompt", input_types["required"])
        self.assertIn("history", input_types["optional"])
        self.assertIn("extra", input_types["optional"])
        self.assertIn("image", input_types["optional"])
        self.assertIn("image1", input_types["optional"])
        self.assertIn("image2", input_types["optional"])
        self.assertIn("image3", input_types["optional"])
        self.assertIn("image4", input_types["optional"])
        self.assertEqual(node_class.RETURN_TYPES, ("IMAGE", "STRING", "HISTORY"))
        self.assertEqual(node_class.FUNCTION, "generate_image")


if __name__ == "__main__":
    unittest.main()
