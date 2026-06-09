import base64
import importlib.util
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
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
        raise AssertionError(f"Missing module file: {path.relative_to(REPO_ROOT)}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module: {module_name}")
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


def make_image(color: str = "red", size=(4, 3)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def image_bytes(color: str = "green", size=(8, 6)) -> bytes:
    buffer = io.BytesIO()
    make_image(color, size=size).save(buffer, format="PNG")
    return buffer.getvalue()


def image_tensor_batch(count: int = 2) -> torch.Tensor:
    samples = []
    for index in range(count):
        sample = torch.zeros((3, 4, 3), dtype=torch.float32)
        sample[..., index % 3] = 1.0
        samples.append(sample)
    return torch.stack(samples, dim=0)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.content = content
        self.text = "" if payload is None else str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TopazMagnificUpscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_import_environment()
        cls.topaz_module = load_module(
            "yogurt_nodes.utils.topaz_client",
            UTILS_DIR / "topaz_client.py",
        )
        cls.magnific_module = load_module(
            "yogurt_nodes.utils.magnific_client",
            UTILS_DIR / "magnific_client.py",
        )

        utils_pkg = sys.modules["yogurt_nodes.utils"]
        setattr(utils_pkg, "TopazClient", cls.topaz_module.TopazClient)
        setattr(utils_pkg, "MagnificClient", cls.magnific_module.MagnificClient)

        cls.topaz_node_module = load_module(
            "yogurt_nodes.llm.topaz_image_upscale",
            LLM_DIR / "topaz_image_upscale.py",
        )
        cls.magnific_node_module = load_module(
            "yogurt_nodes.llm.magnific_image_upscale",
            LLM_DIR / "magnific_image_upscale.py",
        )

    def test_topaz_client_does_not_require_local_api_key(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(self.topaz_module, "load_api_keys", return_value={}),
        ):
            client = self.topaz_module.TopazClient(api_key="")

        self.assertEqual(client.headers["X-API-Key"], "")

    def test_topaz_builds_official_form_fields_from_scale(self):
        form = self.topaz_module.build_topaz_form_data(
            image=make_image(size=(10, 5)),
            model="Standard V2",
            scale_factor=2.0,
            output_width=0,
            output_height=0,
            crop_to_fill=False,
            output_format="png",
            face_enhancement="true",
            face_enhancement_strength=0.8,
            face_enhancement_creativity=-1,
            subject_detection="All",
            sharpen=0.2,
            denoise=-1,
            fix_compression=-1,
            strength=-1,
            prompt="",
            autoprompt="auto",
            creativity=-1,
            texture=-1,
            detail=-1,
            extra={},
        )

        self.assertEqual(form["model"], "Standard V2")
        self.assertEqual(form["output_width"], "20")
        self.assertEqual(form["output_height"], "10")
        self.assertEqual(form["output_format"], "png")
        self.assertEqual(form["crop_to_fill"], "false")
        self.assertEqual(form["face_enhancement"], "true")
        self.assertEqual(form["face_enhancement_strength"], "0.8")
        self.assertEqual(form["subject_detection"], "All")
        self.assertEqual(form["sharpen"], "0.2")
        self.assertNotIn("denoise", form)

    def test_topaz_waits_for_completion_retries_429_and_downloads_image(self):
        calls = []
        responses = [
            FakeResponse(status_code=429, payload={"message": "slow down"}),
            FakeResponse(
                headers={"X-Process-ID": "proc-1"},
                payload={"process_id": "proc-1"},
            ),
            FakeResponse(payload={"status": "Processing", "progress": 50}),
            FakeResponse(payload={"status": "Completed", "progress": 100}),
            FakeResponse(payload={"download_url": "https://download.example/out.png"}),
            FakeResponse(content=image_bytes("blue")),
        ]

        def fake_request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            return responses.pop(0)

        client = self.topaz_module.TopazClient(api_key="", retry_count=2, timeout=5)
        with (
            mock.patch.object(client.http, "request", fake_request),
            mock.patch.object(client.http, "sleep"),
        ):
            output = client.upscale_image(
                image=make_image(),
                model_type="standard",
                model="Standard V2",
                scale_factor=2,
                task_timeout=30,
                poll_interval=0.01,
            )

        self.assertEqual(output.size, (8, 6))
        self.assertEqual(calls[0][1], "https://api.topazlabs.com/image/v1/enhance/async")
        self.assertEqual(calls[2][1], "https://api.topazlabs.com/image/v1/status/proc-1")
        self.assertEqual(calls[4][1], "https://api.topazlabs.com/image/v1/download/proc-1")
        self.assertEqual(calls[5][1], "https://download.example/out.png")

    def test_magnific_builds_creative_payload(self):
        payload = self.magnific_module.build_magnific_payload(
            image=make_image(),
            mode="creative",
            scale_factor="4x",
            optimized_for="films_n_photography",
            prompt="add crisp texture",
            creativity=2,
            hdr=1,
            resemblance=-1,
            fractality=0,
            engine="magnific_sparkle",
            sharpen=50,
            smart_grain=7,
            ultra_detail=30,
            filter_nsfw=True,
            extra={},
        )

        decoded = base64.b64decode(payload["image"])
        self.assertTrue(decoded.startswith(b"\x89PNG"))
        self.assertEqual(payload["scale_factor"], "4x")
        self.assertEqual(payload["optimized_for"], "films_n_photography")
        self.assertEqual(payload["engine"], "magnific_sparkle")
        self.assertEqual(payload["creativity"], 2)
        self.assertTrue(payload["filter_nsfw"])
        self.assertNotIn("sharpen", payload)

    def test_magnific_waits_for_completed_task_and_downloads_image(self):
        calls = []
        responses = [
            FakeResponse(
                payload={
                    "data": {
                        "task_id": "task-1",
                        "status": "IN_PROGRESS",
                        "generated": [],
                    }
                }
            ),
            FakeResponse(
                payload={
                    "data": {
                        "task_id": "task-1",
                        "status": "COMPLETED",
                        "generated": ["https://download.example/mag.png"],
                    }
                }
            ),
            FakeResponse(content=image_bytes("yellow", size=(16, 12))),
        ]

        def fake_request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            return responses.pop(0)

        client = self.magnific_module.MagnificClient(
            api_key="",
            retry_count=1,
            timeout=5,
        )
        with (
            mock.patch.object(client.http, "request", fake_request),
            mock.patch.object(client.http, "sleep"),
        ):
            output = client.upscale_image(
                image=make_image(),
                mode="creative",
                task_timeout=30,
                poll_interval=0.01,
            )

        self.assertEqual(output.size, (16, 12))
        self.assertEqual(calls[0][1], "https://api.magnific.com/v1/ai/image-upscaler")
        self.assertEqual(calls[1][1], "https://api.magnific.com/v1/ai/image-upscaler/task-1")
        self.assertEqual(calls[2][1], "https://download.example/mag.png")

    def test_nodes_declare_single_image_output_and_batch_images(self):
        topaz_node = self.topaz_node_module.TopazImageUpscaleAPI
        magnific_node = self.magnific_node_module.MagnificImageUpscaleAPI

        self.assertEqual(topaz_node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(magnific_node.RETURN_TYPES, ("IMAGE",))
        self.assertIn("image", topaz_node.INPUT_TYPES()["required"])
        self.assertIn("image", magnific_node.INPUT_TYPES()["required"])

        class FakeTopazClient:
            def __init__(self, **kwargs):
                pass

            def upscale_image(self, image, **kwargs):
                return image.resize((image.width * 2, image.height * 2))

        with mock.patch.object(self.topaz_node_module, "TopazClient", FakeTopazClient):
            (result,) = topaz_node().upscale_image(
                image=image_tensor_batch(2),
                api_key="",
                base_url="",
                model_type="standard",
                model_name="Standard V2",
                scale_factor=2.0,
                output_width=0,
                output_height=0,
                crop_to_fill=False,
                output_format="png",
                face_enhancement="auto",
                face_enhancement_strength=-1,
                face_enhancement_creativity=-1,
                subject_detection="auto",
                sharpen=-1,
                denoise=-1,
                fix_compression=-1,
                strength=-1,
                prompt="",
                autoprompt="auto",
                creativity=-1,
                texture=-1,
                detail=-1,
                timeout=5,
                task_timeout=30,
                poll_interval=0.01,
                retry_count=1,
                proxy_url="",
                extra="{}",
            )

        self.assertEqual(tuple(result.shape), (2, 6, 8, 3))


if __name__ == "__main__":
    unittest.main()
