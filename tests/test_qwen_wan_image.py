import base64
import asyncio
import importlib.util
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

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
        cls.openai_client_module = load_module(
            "yogurt_nodes.utils.openai_client",
            UTILS_DIR / "openai_client.py",
        )
        cls.gemini_client_module = load_module(
            "yogurt_nodes.utils.gemini_client",
            UTILS_DIR / "gemini_client.py",
        )

        utils_pkg = sys.modules["yogurt_nodes.utils"]
        setattr(utils_pkg, "QwenClient", cls.qwen_client_module.QwenClient)
        setattr(utils_pkg, "WanClient", cls.wan_client_module.WanClient)
        setattr(utils_pkg, "OpenAIClient", cls.openai_client_module.OpenAIClient)
        setattr(utils_pkg, "GeminiClient", cls.gemini_client_module.GeminiClient)

        cls.qwen_node_module = load_module(
            "yogurt_nodes.llm.qwen_generate_image",
            LLM_DIR / "qwen_generate_image.py",
        )
        cls.wan_node_module = load_module(
            "yogurt_nodes.llm.wan_generate_image",
            LLM_DIR / "wan_generate_image.py",
        )
        cls.openai_node_module = load_module(
            "yogurt_nodes.llm.openai_generate_image",
            LLM_DIR / "openai_generate_image.py",
        )
        cls.gemini_node_module = load_module(
            "yogurt_nodes.llm.gemini_generate_image",
            LLM_DIR / "gemini_generate_image.py",
        )
        cls.gemini_text_node_module = load_module(
            "yogurt_nodes.llm.gemini_generate_text",
            LLM_DIR / "gemini_generate_text.py",
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
        self.assertEqual(
            node_class.RETURN_TYPES,
            ("IMAGE", "IMAGE", "INT", "STRING", "HISTORY"),
        )
        self.assertEqual(
            node_class.RETURN_NAMES,
            ("image", "images", "num_images", "text", "history"),
        )
        self.assertEqual(
            node_class.OUTPUT_IS_LIST,
            (False, True, False, False, False),
        )
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
        self.assertEqual(
            node_class.RETURN_TYPES,
            ("IMAGE", "IMAGE", "INT", "STRING", "HISTORY"),
        )
        self.assertEqual(
            node_class.RETURN_NAMES,
            ("image", "images", "num_images", "text", "history"),
        )
        self.assertEqual(
            node_class.OUTPUT_IS_LIST,
            (False, True, False, False, False),
        )
        self.assertEqual(node_class.FUNCTION, "generate_image")

    def test_openai_node_uses_single_and_list_outputs_with_count(self):
        node_class = getattr(self.openai_node_module, "OpenAIGenerateImage", None)
        self.assertIsNotNone(node_class, "缺少 OpenAIGenerateImage 节点类")
        self.assertEqual(
            node_class.RETURN_TYPES,
            ("IMAGE", "IMAGE", "INT", "STRING", "HISTORY"),
        )
        self.assertEqual(
            node_class.RETURN_NAMES,
            ("image", "images", "num_images", "text", "history"),
        )
        self.assertEqual(
            node_class.OUTPUT_IS_LIST,
            (False, True, False, False, False),
        )

    def test_openai_generate_text_forwards_proxy_mapping(self):
        client_class = getattr(self.openai_client_module, "OpenAIClient", None)
        self.assertIsNotNone(client_class, "缺少 OpenAIClient 类")

        proxy_url = "http://127.0.0.1:7890"
        client = client_class(api_key="test-key", proxy_url=proxy_url)

        class FakeResponse:
            status_code = 200

            text = '{"choices":[{"message":{"content":"ok"}}]}'

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.object(client.http, "post", return_value=FakeResponse()) as mock_post:
            text, _, _ = client.generate_text(
                model_name="gpt-4o-mini",
                prompt="ping",
                retry_count=1,
            )

        self.assertEqual(text, "ok")
        self.assertEqual(
            mock_post.call_args.kwargs["proxies"],
            {"http": proxy_url, "https": proxy_url},
        )

    def test_openai_images_api_uses_httpx_proxy(self):
        client_class = getattr(self.openai_client_module, "OpenAIClient", None)
        self.assertIsNotNone(client_class, "缺少 OpenAIClient 类")

        proxy_url = "http://127.0.0.1:7890"
        image_buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color="red").save(image_buffer, format="PNG")
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")

        class FakeResponse:
            status_code = 200
            text = '{"data":[{"b64_json":"..."}]}'

            @staticmethod
            def json():
                return {"data": [{"b64_json": encoded_image}]}

        client = client_class(api_key="test-key", proxy_url=proxy_url)
        with mock.patch.object(client.http, "post", return_value=FakeResponse()):
            images, _, _ = client.generate_image(
                model_name="gpt-image-1",
                prompt="test",
                api_type="image",
                response_format="b64_json",
                retry_count=1,
            )

        self.assertEqual(len(images), 1)
        self.assertEqual(client.http.proxy_url, proxy_url)

    def test_gemini_node_uses_single_and_list_outputs_with_count(self):
        node_class = getattr(self.gemini_node_module, "GeminiGenerateImage", None)
        self.assertIsNotNone(node_class, "缺少 GeminiGenerateImage 节点类")
        self.assertEqual(
            node_class.RETURN_TYPES,
            ("IMAGE", "IMAGE", "INT", "STRING", "HISTORY", "STRING"),
        )
        self.assertEqual(
            node_class.RETURN_NAMES,
            ("image", "images", "num_images", "text", "history", "thought"),
        )
        self.assertEqual(
            node_class.OUTPUT_IS_LIST,
            (False, True, False, False, False, False),
        )

    def test_gemini_client_sets_httpx_proxy_args(self):
        client_class = getattr(self.gemini_client_module, "GeminiClient", None)
        self.assertIsNotNone(client_class, "缺少 GeminiClient 类")

        proxy_url = "http://127.0.0.1:7890"
        created_clients = []

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                created_clients.append(kwargs)

        with mock.patch.object(
            self.gemini_client_module.genai,
            "Client",
            FakeGenAIClient,
        ):
            client_class(api_key="test-key", proxy_url=proxy_url, timeout=11)

        http_options = created_clients[0]["http_options"]
        self.assertEqual(http_options.client_args["proxy"], proxy_url)
        self.assertEqual(http_options.async_client_args["proxy"], proxy_url)
        self.assertEqual(http_options.timeout, 11000)

    def test_gemini_generate_text_async_uses_sdk_async_generate_content(self):
        client_class = getattr(self.gemini_client_module, "GeminiClient", None)
        self.assertIsNotNone(client_class, "缺少 GeminiClient 类")

        calls = []

        class FakePart:
            text = "ok"

        class FakeContent:
            parts = [FakePart()]

        class FakeCandidate:
            content = FakeContent()

        class FakeResponse:
            candidates = [FakeCandidate()]

        class FakeAsyncModels:
            async def generate_content(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeAioClient:
            def __init__(self):
                self.models = FakeAsyncModels()

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.aio = FakeAioClient()

        async def run():
            with mock.patch.object(
                self.gemini_client_module.genai,
                "Client",
                FakeGenAIClient,
            ):
                client = client_class(api_key="test-key")
            return await client.generate_text_async(
                model_name="gemini-test",
                prompt="ping",
                retry_count=1,
            )

        text, thought, history = asyncio.run(run())

        self.assertEqual(text, "ok")
        self.assertEqual(thought, "")
        self.assertEqual(history, [("user", "ping"), ("assistant", "ok")])
        self.assertEqual(calls[0]["model"], "gemini-test")

    def test_gemini_generate_image_async_uses_sdk_async_generate_content(self):
        client_class = getattr(self.gemini_client_module, "GeminiClient", None)
        self.assertIsNotNone(client_class, "缺少 GeminiClient 类")

        calls = []
        image_buffer = io.BytesIO()
        Image.new("RGB", (6, 4), color="cyan").save(image_buffer, format="PNG")

        class FakeTextPart:
            text = "caption"

        class FakeInlineData:
            data = image_buffer.getvalue()

        class FakeImagePart:
            inline_data = FakeInlineData()

        class FakeContent:
            parts = [FakeTextPart(), FakeImagePart()]

        class FakeCandidate:
            content = FakeContent()

        class FakeResponse:
            candidates = [FakeCandidate()]

        class FakeAsyncModels:
            async def generate_content(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeAioClient:
            def __init__(self):
                self.models = FakeAsyncModels()

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.aio = FakeAioClient()

        async def run():
            with mock.patch.object(
                self.gemini_client_module.genai,
                "Client",
                FakeGenAIClient,
            ):
                client = client_class(api_key="test-key")
            return await client.generate_image_async(
                model_name="gemini-image-test",
                prompt="paint",
                retry_count=1,
            )

        images, text, thought, history = asyncio.run(run())

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (6, 4))
        self.assertEqual(text, "caption")
        self.assertEqual(thought, "")
        self.assertEqual(history, [("user", "paint"), ("assistant", "caption")])
        self.assertEqual(calls[0]["model"], "gemini-image-test")

    def test_gemini_generate_text_async_cancels_and_closes_aio_on_interrupt(self):
        client_class = getattr(self.gemini_client_module, "GeminiClient", None)
        self.assertIsNotNone(client_class, "缺少 GeminiClient 类")

        class TestInterrupt(BaseException):
            pass

        state = {"interrupted": False}

        def processing_interrupted():
            return state["interrupted"]

        def throw_exception_if_processing_interrupted():
            if state["interrupted"]:
                raise TestInterrupt()

        async def run():
            started = asyncio.Event()

            class FakeAsyncModels:
                def __init__(self):
                    self.cancelled = False

                async def generate_content(self, **kwargs):
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled = True
                        raise

            class FakeAioClient:
                def __init__(self):
                    self.models = FakeAsyncModels()
                    self.closed = False

                async def aclose(self):
                    self.closed = True

            class FakeGenAIClient:
                def __init__(self, **kwargs):
                    self.aio = FakeAioClient()

            with mock.patch.object(
                self.gemini_client_module.genai,
                "Client",
                FakeGenAIClient,
            ):
                client = client_class(api_key="test-key")
            client.cancel_check_interval = 0.01

            with (
                mock.patch.object(
                    self.gemini_client_module.model_management,
                    "processing_interrupted",
                    processing_interrupted,
                    create=True,
                ),
                mock.patch.object(
                    self.gemini_client_module.model_management,
                    "throw_exception_if_processing_interrupted",
                    throw_exception_if_processing_interrupted,
                ),
            ):
                task = asyncio.create_task(
                    client.generate_text_async(
                        model_name="gemini-test",
                        prompt="ping",
                        retry_count=1,
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=1)
                state["interrupted"] = True
                with self.assertRaises(TestInterrupt):
                    await asyncio.wait_for(task, timeout=1)

            return client.client.aio.closed, client.client.aio.models.cancelled

        closed, cancelled = asyncio.run(run())

        self.assertTrue(closed)
        self.assertTrue(cancelled)

    def test_gemini_generate_text_async_closes_aio_when_cancel_task_raises_error(self):
        client_class = getattr(self.gemini_client_module, "GeminiClient", None)
        self.assertIsNotNone(client_class, "缺少 GeminiClient 类")

        class TestInterrupt(BaseException):
            pass

        state = {"interrupted": False}

        def processing_interrupted():
            return state["interrupted"]

        def throw_exception_if_processing_interrupted():
            if state["interrupted"]:
                raise TestInterrupt()

        async def run():
            started = asyncio.Event()

            class FakeAsyncModels:
                def __init__(self):
                    self.cancelled = False

                async def generate_content(self, **kwargs):
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError as exception:
                        self.cancelled = True
                        raise RuntimeError("transport failed while cancelling") from exception

            class FakeAioClient:
                def __init__(self):
                    self.models = FakeAsyncModels()
                    self.closed = False

                async def aclose(self):
                    self.closed = True

            class FakeGenAIClient:
                def __init__(self, **kwargs):
                    self.aio = FakeAioClient()

            with mock.patch.object(
                self.gemini_client_module.genai,
                "Client",
                FakeGenAIClient,
            ):
                client = client_class(api_key="test-key")
            client.cancel_check_interval = 0.01

            with (
                mock.patch.object(
                    self.gemini_client_module.model_management,
                    "processing_interrupted",
                    processing_interrupted,
                    create=True,
                ),
                mock.patch.object(
                    self.gemini_client_module.model_management,
                    "throw_exception_if_processing_interrupted",
                    throw_exception_if_processing_interrupted,
                ),
            ):
                task = asyncio.create_task(
                    client.generate_text_async(
                        model_name="gemini-test",
                        prompt="ping",
                        retry_count=1,
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=1)
                state["interrupted"] = True
                with self.assertRaises(TestInterrupt):
                    await asyncio.wait_for(task, timeout=1)

            return client.client.aio.closed, client.client.aio.models.cancelled

        closed, cancelled = asyncio.run(run())

        self.assertTrue(closed)
        self.assertTrue(cancelled)

    def test_gemini_text_node_closes_owned_client_after_success(self):
        node_class = getattr(self.gemini_text_node_module, "GeminiGenerateText", None)
        self.assertIsNotNone(node_class, "缺少 GeminiGenerateText 节点类")

        closed = []

        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def generate_text_async(self, **kwargs):
                return "ok", "", [("assistant", "ok")]

            async def close_async(self):
                closed.append(True)

        async def run():
            with mock.patch.object(
                self.gemini_text_node_module,
                "GeminiClient",
                FakeClient,
            ):
                return await node_class().generate_text(
                    model_name="gemini-test",
                    prompt="ping",
                    retry_count=1,
                    extra="{}",
                )

        result = asyncio.run(run())

        self.assertEqual(result, ("ok", [("assistant", "ok")], ""))
        self.assertEqual(closed, [True])

    def test_gemini_text_node_closes_owned_client_when_cancelled(self):
        node_class = getattr(self.gemini_text_node_module, "GeminiGenerateText", None)
        self.assertIsNotNone(node_class, "缺少 GeminiGenerateText 节点类")

        closed = []

        async def run():
            started = asyncio.Event()

            class FakeClient:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                async def generate_text_async(self, **kwargs):
                    started.set()
                    await asyncio.Event().wait()

                async def close_async(self):
                    closed.append(True)

            with mock.patch.object(
                self.gemini_text_node_module,
                "GeminiClient",
                FakeClient,
            ):
                task = asyncio.create_task(
                    node_class().generate_text(
                        model_name="gemini-test",
                        prompt="ping",
                        retry_count=1,
                        extra="{}",
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)

        asyncio.run(run())

        self.assertEqual(closed, [True])

    def test_gemini_text_node_does_not_create_client_for_invalid_extra_json(self):
        node_class = getattr(self.gemini_text_node_module, "GeminiGenerateText", None)
        self.assertIsNotNone(node_class, "缺少 GeminiGenerateText 节点类")

        created = []

        class FakeClient:
            def __init__(self, **kwargs):
                created.append(True)

        async def run():
            with mock.patch.object(
                self.gemini_text_node_module,
                "GeminiClient",
                FakeClient,
            ):
                with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                    await node_class().generate_text(
                        model_name="gemini-test",
                        prompt="ping",
                        retry_count=1,
                        extra="{invalid",
                    )

        asyncio.run(run())

        self.assertEqual(created, [])

    def test_gemini_image_node_closes_owned_client_after_success(self):
        node_class = getattr(self.gemini_node_module, "GeminiGenerateImage", None)
        self.assertIsNotNone(node_class, "缺少 GeminiGenerateImage 节点类")

        closed = []

        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def generate_image_async(self, **kwargs):
                return [], "caption", "", [("assistant", "caption")]

            async def close_async(self):
                closed.append(True)

        async def run():
            with mock.patch.object(
                self.gemini_node_module,
                "GeminiClient",
                FakeClient,
            ):
                return await node_class().generate_image(
                    model_name="gemini-image-test",
                    prompt="paint",
                    retry_count=1,
                    extra="{}",
                )

        _image, _images, count, text, history, thought = asyncio.run(run())

        self.assertEqual(count, 0)
        self.assertEqual(text, "caption")
        self.assertEqual(history, [("assistant", "caption")])
        self.assertEqual(thought, "")
        self.assertEqual(closed, [True])

    def test_gemini_image_node_closes_owned_client_when_cancelled(self):
        node_class = getattr(self.gemini_node_module, "GeminiGenerateImage", None)
        self.assertIsNotNone(node_class, "缺少 GeminiGenerateImage 节点类")

        closed = []

        async def run():
            started = asyncio.Event()

            class FakeClient:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                async def generate_image_async(self, **kwargs):
                    started.set()
                    await asyncio.Event().wait()

                async def close_async(self):
                    closed.append(True)

            with mock.patch.object(
                self.gemini_node_module,
                "GeminiClient",
                FakeClient,
            ):
                task = asyncio.create_task(
                    node_class().generate_image(
                        model_name="gemini-image-test",
                        prompt="paint",
                        retry_count=1,
                        extra="{}",
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)

        asyncio.run(run())

        self.assertEqual(closed, [True])

    def test_gemini_vertex_credentials_refresh_uses_proxy_env(self):
        client_class = getattr(self.gemini_client_module, "GeminiClient", None)
        self.assertIsNotNone(client_class, "缺少 GeminiClient 类")

        proxy_url = "http://127.0.0.1:7890"
        refresh_proxy_values = []

        class FakeCredentials:
            token = None
            valid = False
            expired = True

            def with_scopes(self, scopes):
                return self

            def refresh(self, request):
                refresh_proxy_values.append(os.environ.get("HTTPS_PROXY"))
                self.token = "token"
                self.valid = True
                self.expired = False

        fake_credentials = FakeCredentials()

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with (
            mock.patch.object(
                self.gemini_client_module.genai,
                "Client",
                FakeGenAIClient,
            ),
            mock.patch(
                "google.oauth2.service_account.Credentials.from_service_account_info",
                return_value=fake_credentials,
            ),
        ):
            client = client_class(
                use_vertex_ai=True,
                vertex_ai_json="{}",
                vertex_ai_project="project",
                vertex_ai_region="us-central1",
                proxy_url=proxy_url,
            )
            client._ensure_vertex_credentials()

        self.assertEqual(refresh_proxy_values, [proxy_url])

    def test_gemini_vertex_adc_credentials_are_loaded_under_proxy_env(self):
        client_class = getattr(self.gemini_client_module, "GeminiClient", None)
        self.assertIsNotNone(client_class, "缺少 GeminiClient 类")

        proxy_url = "http://127.0.0.1:7890"
        token_proxy_values = []

        class FakeCredentials:
            token = "token"
            valid = True
            expired = False

        class FakeAPIClient:
            _credentials = None

            def _access_token(self):
                token_proxy_values.append(os.environ.get("HTTPS_PROXY"))
                self._credentials = FakeCredentials()
                return "token"

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self._api_client = FakeAPIClient()

        with (
            mock.patch.object(
                self.gemini_client_module.genai,
                "Client",
                FakeGenAIClient,
            ),
            mock.patch.object(
                self.gemini_client_module,
                "load_api_keys",
                return_value={},
            ),
        ):
            client = client_class(
                use_vertex_ai=True,
                vertex_ai_json="",
                vertex_ai_project="project",
                vertex_ai_region="us-central1",
                proxy_url=proxy_url,
            )
            client._ensure_vertex_credentials()

        self.assertEqual(token_proxy_values, [proxy_url])


class OpenRouterImageNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_import_environment()
        openai_client_module = load_module(
            "yogurt_nodes.utils.openai_client",
            UTILS_DIR / "openai_client.py",
        )
        openrouter_client_module = load_module(
            "yogurt_nodes.utils.openrouter_client",
            UTILS_DIR / "openrouter_client.py",
        )
        cls.openrouter_client_module = openrouter_client_module

        utils_pkg = sys.modules["yogurt_nodes.utils"]
        setattr(utils_pkg, "OpenAIClient", getattr(openai_client_module, "OpenAIClient"))
        setattr(
            utils_pkg,
            "OpenRouterClient",
            getattr(openrouter_client_module, "OpenRouterClient"),
        )

        cls.openrouter_node_module = load_module(
            "yogurt_nodes.llm.openrouter_generate_image",
            LLM_DIR / "openrouter_generate_image.py",
        )

    def test_openrouter_node_uses_single_and_list_outputs_with_count(self):
        node_class = getattr(
            self.openrouter_node_module,
            "OpenRouterGenerateImage",
            None,
        )
        self.assertIsNotNone(node_class, "缺少 OpenRouterGenerateImage 节点类")
        self.assertEqual(
            node_class.RETURN_TYPES,
            ("IMAGE", "IMAGE", "INT", "STRING", "HISTORY"),
        )
        self.assertEqual(
            node_class.RETURN_NAMES,
            ("image", "images", "num_images", "text", "history"),
        )
        self.assertEqual(
            node_class.OUTPUT_IS_LIST,
            (False, True, False, False, False),
        )
        self.assertTrue(hasattr(node_class, "generate_image"))

    def test_openrouter_formats_multiple_images_as_list_output(self):
        build_image_outputs = getattr(
            self.openrouter_node_module,
            "build_image_outputs",
            None,
        )
        self.assertIsNotNone(
            build_image_outputs,
            "缺少 build_image_outputs()",
        )

        images = [
            Image.new("RGB", (1200, 1200), color="red"),
            Image.new("RGB", (2400, 1200), color="blue"),
        ]
        image_output, image_list, image_count = build_image_outputs(images)

        self.assertEqual(image_count, 2)
        self.assertEqual(tuple(image_output.shape), (1, 1200, 1200, 3))
        self.assertEqual(len(image_list), 2)
        self.assertEqual(tuple(image_list[0].shape), (1, 1200, 1200, 3))
        self.assertEqual(tuple(image_list[1].shape), (1, 1200, 2400, 3))

    def test_openrouter_formats_single_image_as_image_output(self):
        build_image_outputs = getattr(
            self.openrouter_node_module,
            "build_image_outputs",
            None,
        )
        self.assertIsNotNone(
            build_image_outputs,
            "缺少 build_image_outputs()",
        )

        image_output, image_list, image_count = build_image_outputs(
            [Image.new("RGB", (1200, 1200), color="green")]
        )

        self.assertEqual(image_count, 1)
        self.assertEqual(tuple(image_output.shape), (1, 1200, 1200, 3))
        self.assertEqual(len(image_list), 1)
        self.assertEqual(tuple(image_list[0].shape), (1, 1200, 1200, 3))

    def test_openrouter_normalizes_image_size_for_api(self):
        normalize_image_size = getattr(
            self.openrouter_client_module,
            "normalize_image_size",
            None,
        )
        self.assertIsNotNone(
            normalize_image_size,
            "缺少 normalize_image_size()",
        )

        self.assertEqual(normalize_image_size("1k"), "1K")
        self.assertEqual(normalize_image_size("2k"), "2K")
        self.assertEqual(normalize_image_size("4K"), "4K")
        self.assertIsNone(normalize_image_size("auto"))

    def test_openrouter_generate_image_sends_uppercase_image_size(self):
        client_class = getattr(
            self.openrouter_client_module,
            "OpenRouterClient",
            None,
        )
        self.assertIsNotNone(client_class, "缺少 OpenRouterClient 类")

        client = client_class(api_key="test-key")

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                                "images": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": (
                                                "data:image/png;base64,"
                                                "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSnc"
                                                "AAAADElEQVR4nGP4z4AdAAn0AcwV+0U8AAAAAElFTkSuQmCC"
                                            )
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }

        with mock.patch.object(client.http, "post", return_value=FakeResponse()) as mock_post:
            client.generate_image(
                model_name="google/gemini-2.5-flash-image-preview",
                prompt="test",
                image_size="2k",
                retry_count=1,
            )

        sent_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["image_config"]["image_size"], "2K")


if __name__ == "__main__":
    unittest.main()
