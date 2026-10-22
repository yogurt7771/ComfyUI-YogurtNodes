import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from urllib.parse import urlencode

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEW_SUBFOLDER_PATH = REPO_ROOT / "yogurt_nodes" / "view_subfolder.py"


def load_view_subfolder():
    spec = importlib.util.spec_from_file_location("view_subfolder_under_test", VIEW_SUBFOLDER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("无法加载 view_subfolder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


view_subfolder = load_view_subfolder()


def make_core_view(root):
    # Mirrors the path handling of ComfyUI's /view handler (server.py).
    async def core_view(request):
        query = request.rel_url.query
        filename = query.get("filename", "")
        if not filename or filename[0] == "/" or ".." in filename:
            return web.Response(status=400)
        output_dir = root
        if "subfolder" in query:
            full_output_dir = os.path.join(output_dir, query["subfolder"])
            if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                return web.Response(status=403)
            output_dir = full_output_dir
        file = os.path.join(output_dir, os.path.basename(filename))
        if not os.path.isfile(file):
            return web.Response(status=404)
        return web.FileResponse(file)

    return core_view


class ViewSubfolderMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.abspath(self._tmp.name)
        for rel, data in (("top.png", b"top"), ("ljh/双下巴！.png", b"nested")):
            path = Path(self.root, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def tearDown(self):
        self._tmp.cleanup()

    async def _client(self, middlewares):
        app = web.Application(middlewares=middlewares)
        core_view = make_core_view(self.root)
        app.router.add_get("/view", core_view)
        app.router.add_get("/api/view", core_view)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def _get(self, client, path, **params):
        # Same encoding as the frontend's URLSearchParams ("/" -> %2F).
        response = await client.get(f"{path}?{urlencode(params)}")
        return response.status, await response.read()

    async def test_without_middleware_subfolder_thumbnail_404s(self):
        client = await self._client([])
        status, _ = await self._get(client, "/api/view", filename="ljh/双下巴！.png", type="input")
        self.assertEqual(status, 404)

    async def test_subfolder_filename_is_served(self):
        client = await self._client([view_subfolder.view_subfolder_middleware])
        for path in ("/api/view", "/view"):
            status, body = await self._get(client, path, filename="ljh/双下巴！.png", type="input")
            self.assertEqual((status, body), (200, b"nested"), path)

    async def test_plain_and_explicit_subfolder_requests_unchanged(self):
        client = await self._client([view_subfolder.view_subfolder_middleware])
        self.assertEqual(await self._get(client, "/api/view", filename="top.png"), (200, b"top"))
        self.assertEqual(
            await self._get(client, "/api/view", filename="双下巴！.png", subfolder="ljh"),
            (200, b"nested"),
        )

    async def test_traversal_still_rejected(self):
        client = await self._client([view_subfolder.view_subfolder_middleware])
        for filename in ("../top.png", "ljh/../top.png", "/ljh/双下巴！.png"):
            status, _ = await self._get(client, "/api/view", filename=filename)
            self.assertEqual(status, 400, filename)

    def test_install_appends_once(self):
        app = web.Application()
        server_module = types.ModuleType("server")
        server_module.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(app=app))
        original = sys.modules.get("server")
        sys.modules["server"] = server_module
        try:
            view_subfolder.install_view_subfolder_middleware()
            view_subfolder.install_view_subfolder_middleware()
        finally:
            if original is None:
                sys.modules.pop("server", None)
            else:
                sys.modules["server"] = original
        self.assertEqual(list(app.middlewares).count(view_subfolder.view_subfolder_middleware), 1)


if __name__ == "__main__":
    unittest.main()
