import asyncio
import base64
import importlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
YOGURT_DIR = REPO_ROOT / "yogurt_nodes"


def ensure_package(name: str, path: Path):
    package = sys.modules.get(name)
    if package is None:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = package
    return package


class Interrupted(BaseException):
    """Stands in for comfy.model_management.InterruptProcessingException (also a BaseException)."""


# 新框架依赖真实的 comfy_api（需要 PYTHONPATH 含 ComfyUI 根目录）。在模块导入阶段完成加载，
# 早于其他测试在 setUpClass 中替换 comfy 桩模块。
ensure_package("yogurt_nodes", YOGURT_DIR)
ensure_package("yogurt_nodes.utils", YOGURT_DIR / "utils")
ensure_package("yogurt_nodes.llm", YOGURT_DIR / "llm")
MODULES = {
    name: importlib.import_module(f"yogurt_nodes.llm.{path}")
    for name, path in {
        "framework": "framework",
        "runtime": "framework.runtime",
        "http": "protocols.http",
        "protocols": "protocols",
        "sdcpp": "protocols.sdcpp",
        "extra_nodes": "extra_nodes",
        "image_nodes": "image_nodes",
    }.items()
}


def make_image(color="red", size=(8, 6)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def b64_png(color="green", size=(4, 4)) -> str:
    buffer = io.BytesIO()
    make_image(color, size).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class SDCPPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, module in MODULES.items():
            setattr(cls, name, module)

    def request(self, **overrides):
        fields = {"task": "image", "prompt": "a cat", "params": {}, "seed": 0}
        fields.update(overrides)
        return self.framework.GenerationRequest(**fields)

    def provider(self, **settings):
        return self.framework.ProviderConfig(kind="sdcpp", route_key="sdcpp", settings=settings)

    def route(self):
        return self.framework.Route("sdcpp_img_gen", "")

    def run_protocol(self, responses, provider=None, request=None, sleep=None):
        calls = []

        async def fake_request(conn, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return responses.pop(0)

        async def no_sleep(seconds):
            return None

        handler = self.framework.PROTOCOLS["sdcpp_img_gen"]
        with (
            mock.patch.object(self.sdcpp, "http_request", fake_request),
            mock.patch.object(self.sdcpp, "cancellable_sleep", sleep or no_sleep),
        ):
            result = asyncio.run(handler(provider or self.provider(), self.route(), request or self.request()))
        return result, calls

    # --- request body -------------------------------------------------------

    def test_body_maps_node_fields_and_extra_overrides(self):
        body = self.sdcpp.img_gen_body(
            self.request(
                params={"width": 512, "height": 768, "n": 2,
                        "extra": {"seed": 7, "sample_params": {"sample_steps": 20}}},
                images=[make_image()],
            )
        )
        self.assertEqual(body["prompt"], "a cat")
        self.assertEqual((body["width"], body["height"], body["batch_count"]), (512, 768, 2))
        self.assertEqual(body["seed"], 7)
        self.assertEqual(body["sample_params"], {"sample_steps": 20})
        self.assertTrue(body["ref_images"][0].startswith("data:image/png;base64,"))

    def test_body_zero_seed_is_random_and_size_text_is_parsed(self):
        body = self.sdcpp.img_gen_body(self.request(params={"size": "1024x768"}))
        self.assertEqual(body["seed"], -1)
        self.assertEqual((body["width"], body["height"]), (1024, 768))
        self.assertNotIn("ref_images", body)
        self.assertNotIn("batch_count", body)

    def test_body_follows_init_image_size_when_no_size_given(self):
        init = self.http.data_url(make_image(size=(640, 384)))
        body = self.sdcpp.img_gen_body(self.request(params={"width": 0, "height": 0, "extra": {"init_image": init}}))
        self.assertEqual((body["width"], body["height"]), (640, 384))

        explicit = self.sdcpp.img_gen_body(
            self.request(params={"width": 512, "height": 512, "extra": {"init_image": init}})
        )
        self.assertEqual((explicit["width"], explicit["height"]), (512, 512))

    # --- job flow -----------------------------------------------------------

    def test_submit_poll_and_decode_images_in_index_order(self):
        responses = [
            FakeResponse(202, {"id": "job_1", "status": "queued"}),
            FakeResponse(200, {"id": "job_1", "status": "generating"}),
            FakeResponse(200, {
                "id": "job_1",
                "status": "completed",
                "result": {"images": [
                    {"index": 1, "b64_json": b64_png("blue", (3, 3))},
                    {"index": 0, "b64_json": b64_png("red", (2, 2))},
                ]},
            }),
        ]
        result, calls = self.run_protocol(responses)
        self.assertEqual([image.size for image in result.images], [(2, 2), (3, 3)])
        self.assertEqual(calls[0][0:2], ("POST", "http://127.0.0.1:1234/sdcpp/v1/img_gen"))
        self.assertEqual(calls[0][2]["headers"], {})
        self.assertEqual(calls[1][0:2], ("GET", "http://127.0.0.1:1234/sdcpp/v1/jobs/job_1"))
        self.assertEqual(len(calls), 3)

    def test_bearer_header_only_when_key_given(self):
        responses = [
            FakeResponse(202, {"id": "job_k"}),
            FakeResponse(200, {"id": "job_k", "status": "completed", "result": {"images": [{"b64_json": b64_png()}]}}),
        ]
        _, calls = self.run_protocol(responses, provider=self.provider(api_key="tok", base_url="http://gpu:9000/"))
        self.assertEqual(calls[0][1], "http://gpu:9000/sdcpp/v1/img_gen")
        self.assertEqual(calls[0][2]["headers"], {"Authorization": "Bearer tok"})

    def test_failed_job_raises_server_message(self):
        responses = [
            FakeResponse(202, {"id": "job_f"}),
            FakeResponse(200, {"id": "job_f", "status": "failed",
                               "error": {"code": "generation_failed", "message": "out of memory"}}),
        ]
        with self.assertRaisesRegex(RuntimeError, "job_f failed: out of memory"):
            self.run_protocol(responses)

    def test_interrupt_while_polling_cancels_server_job(self):
        calls = []
        responses = [
            FakeResponse(202, {"id": "job_c"}),
            FakeResponse(200, {"id": "job_c", "status": "generating"}),
            FakeResponse(200, {"id": "job_c", "status": "generating"}),
        ]
        sleeps = []

        async def fake_request(conn, method, url, **kwargs):
            calls.append((method, url))
            return responses.pop(0)

        async def interrupt_on_second_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 2:
                raise Interrupted()

        handler = self.framework.PROTOCOLS["sdcpp_img_gen"]
        with (
            mock.patch.object(self.sdcpp, "http_request", fake_request),
            mock.patch.object(self.sdcpp, "cancellable_sleep", interrupt_on_second_sleep),
            self.assertRaises(Interrupted),
        ):
            asyncio.run(handler(self.provider(), self.route(), self.request()))
        self.assertEqual(calls, [
            ("POST", "http://127.0.0.1:1234/sdcpp/v1/img_gen"),
            ("GET", "http://127.0.0.1:1234/sdcpp/v1/jobs/job_c"),
            ("POST", "http://127.0.0.1:1234/sdcpp/v1/jobs/job_c/cancel"),
        ])

    def test_timeout_cancels_server_job(self):
        calls = []
        responses = [FakeResponse(202, {"id": "job_t"}), FakeResponse(200, {})]

        async def fake_request(conn, method, url, **kwargs):
            calls.append((method, url))
            return responses.pop(0)

        async def no_sleep(seconds):
            return None

        clock = iter([0.0, 99999.0])
        handler = self.framework.PROTOCOLS["sdcpp_img_gen"]
        with (
            mock.patch.object(self.sdcpp, "http_request", fake_request),
            mock.patch.object(self.sdcpp, "cancellable_sleep", no_sleep),
            mock.patch.object(self.sdcpp, "time", types.SimpleNamespace(monotonic=lambda: next(clock))),
            self.assertRaisesRegex(TimeoutError, "job_t did not finish within 30s"),
        ):
            asyncio.run(handler(self.provider(task_timeout=30), self.route(), self.request()))
        self.assertEqual(calls[-1], ("POST", "http://127.0.0.1:1234/sdcpp/v1/jobs/job_t/cancel"))

    # --- connection ---------------------------------------------------------

    def test_sdcpp_needs_no_key_but_compatible_still_does(self):
        conn = self.http.connect(self.provider())
        self.assertEqual((conn.base_url, conn.api_key), ("http://127.0.0.1:1234", ""))
        compatible = self.framework.ProviderConfig(
            kind="openai_compatible", route_key="openai", settings={"base_url": "http://x/v1"}
        )
        with self.assertRaisesRegex(ValueError, "API key is not set"):
            self.http.connect(compatible)

    # --- extra port ---------------------------------------------------------

    def test_merge_extra_deep_merges_and_json_text_wins(self):
        extra = self.framework.ExtraParams(
            source="stable-diffusion.cpp Extra",
            data={"sample_params": {"sample_steps": 20, "guidance": {"txt_cfg": 5.0}}, "negative_prompt": "a"},
            protocol="sdcpp_img_gen",
        )
        merged = self.runtime.merge_extra(
            extra, '{"sample_params": {"guidance": {"txt_cfg": 3.5}}}', self.route(), self.provider()
        )
        self.assertEqual(merged, {
            "sample_params": {"sample_steps": 20, "guidance": {"txt_cfg": 3.5}},
            "negative_prompt": "a",
        })
        self.assertEqual(extra.data["sample_params"]["guidance"]["txt_cfg"], 5.0)

    def test_merge_extra_rejects_other_protocols(self):
        extra = self.framework.ExtraParams(source="stable-diffusion.cpp Extra", data={}, protocol="sdcpp_img_gen")
        openai = self.framework.ProviderConfig(kind="openai_compatible", route_key="openai")
        with self.assertRaisesRegex(ValueError, "only applies to the 'sdcpp_img_gen' API.*OpenAI Compatible"):
            self.runtime.merge_extra(extra, "", self.framework.Route("openai_images", "m"), openai)

    def test_extra_node_defaults_send_nothing(self):
        node = self.extra_nodes.SDCPPExtra
        defaults = {param.id: param.default for param in node.PARAMS if param.kind not in ("image", "mask")}
        self.assertEqual(node.build(**defaults), {})

    def test_extra_node_builds_native_fields(self):
        node = self.extra_nodes.SDCPPExtra
        values = {param.id: param.default for param in node.PARAMS if param.kind not in ("image", "mask")}
        values.update(
            negative_prompt="blurry",
            steps=28,
            cfg=4.5,
            sampler="euler_a",
            scheduler="karras",
            distilled_guidance=3.5,
            loras="detail.safetensors:0.8\nstyle/anime.safetensors\n\n",
            strength=0.6,
            clip_skip=2,
            hires=True,
            hires_steps=10,
            vae_tiling=True,
            cache_mode="easycache",
            init_image=torch.zeros((1, 6, 8, 3)),
            mask=torch.ones((1, 6, 8)),
            ip_adapter_image=torch.ones((1, 4, 5, 3)),
            ip_adapter_strength=0.7,
        )
        data = node.build(**values)
        self.assertEqual(data["sample_params"], {
            "sample_steps": 28,
            "sample_method": "euler_a",
            "scheduler": "karras",
            "guidance": {"txt_cfg": 4.5, "distilled_guidance": 3.5},
        })
        self.assertEqual(data["lora"], [
            {"path": "detail.safetensors", "multiplier": 0.8},
            {"path": "style/anime.safetensors", "multiplier": 1.0},
        ])
        self.assertEqual(data["hires"], {
            "enabled": True, "upscaler": "Latent", "scale": 2.0, "denoising_strength": 0.7, "steps": 10,
        })
        self.assertEqual(data["vae_tiling_params"], {"enabled": True})
        self.assertEqual((data["strength"], data["clip_skip"], data["cache_mode"]), (0.6, 2, "easycache"))
        init = self.http.decode_image(data["init_image"])
        mask = self.http.decode_image(data["mask_image"])
        self.assertEqual((init.size, mask.size, mask.mode), ((8, 6), (8, 6), "L"))
        self.assertEqual(mask.getextrema(), (255, 255))
        ip_adapter = self.http.decode_image(data["ip_adapter_image"])
        self.assertEqual((ip_adapter.size, ip_adapter.mode), ((5, 4), "RGB"))
        self.assertEqual(data["ip_adapter_strength"], 0.7)
        self.assertNotIn("control_image", data)

        output = node.execute(**values)
        extra = output.result[0]
        self.assertEqual((extra.protocol, extra.source), ("sdcpp_img_gen", "stable-diffusion.cpp Extra"))

    # --- node end to end ----------------------------------------------------

    def test_generate_node_merges_extra_port_into_native_request(self):
        sent = {}

        async def fake_request(conn, method, url, **kwargs):
            if method == "POST":
                sent["body"] = kwargs["json"]
                return FakeResponse(202, {"id": "job_n"})
            return FakeResponse(200, {"id": "job_n", "status": "completed",
                                      "result": {"images": [{"b64_json": b64_png(size=(5, 4))}]}})

        async def no_sleep(seconds):
            return None

        extra = self.framework.ExtraParams(
            source="stable-diffusion.cpp Extra",
            data={"negative_prompt": "low quality", "sample_params": {"sample_steps": 20}},
            protocol="sdcpp_img_gen",
        )
        node = self.image_nodes.SDCPPGenerateImage
        with (
            mock.patch.object(self.sdcpp, "http_request", fake_request),
            mock.patch.object(self.sdcpp, "cancellable_sleep", no_sleep),
        ):
            output = asyncio.run(node.execute(
                model={"model": "Loaded model", "width": 512, "height": 768},
                prompt="a lighthouse",
                n=1,
                seed=42,
                extra='{"sample_params": {"sample_steps": 30}}',
                images=None,
                retry_count=1,
                extra_params=extra,
            ))
        self.assertEqual(sent["body"], {
            "prompt": "a lighthouse",
            "seed": 42,
            "width": 512,
            "height": 768,
            "batch_count": 1,
            "negative_prompt": "low quality",
            "sample_params": {"sample_steps": 30},
        })
        self.assertEqual(tuple(output.result[0].shape), (1, 4, 5, 3))


if __name__ == "__main__":
    unittest.main()
