import importlib.util
import sys
import time
import types
import unittest
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
YOGURT_DIR = REPO_ROOT / "yogurt_nodes"
UTILS_DIR = YOGURT_DIR / "utils"


def ensure_package(name: str, path: Path):
    package = sys.modules.get(name)
    if package is None:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = package
    return package


class InterruptProcessingException(BaseException):
    pass


class InterruptState:
    def __init__(self):
        self.interrupted = False
        self.processing_checks = 0

    def processing_interrupted(self):
        self.processing_checks += 1
        return self.interrupted

    def throw_exception_if_processing_interrupted(self):
        if self.interrupted:
            self.interrupted = False
            raise InterruptProcessingException()


def load_cancellable_http(interrupt_state: InterruptState):
    ensure_package("yogurt_nodes", YOGURT_DIR)
    ensure_package("yogurt_nodes.utils", UTILS_DIR)

    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.__path__ = []  # type: ignore[attr-defined]
    model_management = types.ModuleType("comfy.model_management")
    model_management.InterruptProcessingException = InterruptProcessingException
    model_management.processing_interrupted = interrupt_state.processing_interrupted
    model_management.throw_exception_if_processing_interrupted = (
        interrupt_state.throw_exception_if_processing_interrupted
    )
    comfy_pkg.model_management = model_management
    sys.modules["comfy"] = comfy_pkg
    sys.modules["comfy.model_management"] = model_management

    module_name = "yogurt_nodes.utils.cancellable_http"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        UTILS_DIR / "cancellable_http.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load cancellable_http module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class CancellableHttpClientTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "comfy",
                "comfy.model_management",
                "yogurt_nodes.utils.cancellable_http",
            )
        }

    def tearDown(self):
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_request_returns_response_compatible_json(self):
        state = InterruptState()
        module = load_cancellable_http(state)

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"ok": True})
        )
        client = module.CancellableHttpClient(timeout=5, transport=transport)

        response = client.get("https://example.test/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertIn("ok", response.text)

    def test_request_cancels_when_comfyui_interrupts(self):
        state = InterruptState()
        module = load_cancellable_http(state)

        async def handler(request):
            state.interrupted = True
            try:
                await module.asyncio.sleep(30)
            except module.asyncio.CancelledError:
                raise
            return httpx.Response(200, json={"late": True})

        transport = httpx.MockTransport(handler)
        client = module.CancellableHttpClient(
            timeout=0,
            transport=transport,
            check_interval=0.01,
        )

        started = time.monotonic()
        with self.assertRaises(InterruptProcessingException):
            client.get("https://example.test/slow")

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertGreaterEqual(state.processing_checks, 1)

    def test_sleep_cancels_without_waiting_full_delay(self):
        state = InterruptState()
        module = load_cancellable_http(state)
        client = module.CancellableHttpClient(check_interval=0.01)
        state.interrupted = True

        started = time.monotonic()
        with self.assertRaises(InterruptProcessingException):
            client.sleep(10)

        self.assertLess(time.monotonic() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
