"""Let /view serve thumbnails requested as filename="sub/name.png".

The Vue select dropdown builds option thumbnails as
/api/view?filename=<option>&type=input with no subfolder parameter, and
ComfyUI's /view keeps only os.path.basename(filename), so options that live in
input subfolders 404. This middleware moves the directory part of such
filenames into the subfolder parameter before the core handler runs; the core
handler still applies its own ".." and subfolder containment checks.
"""

from aiohttp import web

VIEW_PATHS = frozenset({"/view", "/api/view"})


def split_view_query(query):
    """Return a copy of query with the filename's directory moved to subfolder, or None."""
    filename = query.get("filename", "")
    if query.get("subfolder") or "/" not in filename:
        return None
    if filename.startswith(("/", "blake3:")) or ".." in filename:
        return None
    subfolder, _, name = filename.rpartition("/")
    rewritten = query.copy()
    rewritten["filename"] = name
    rewritten["subfolder"] = subfolder
    return rewritten


@web.middleware
async def view_subfolder_middleware(request, handler):
    if request.path in VIEW_PATHS:
        query = split_view_query(request.rel_url.query)
        if query is not None:
            request = request.clone(rel_url=request.rel_url.with_query(query))
    return await handler(request)


def install_view_subfolder_middleware():
    from server import PromptServer

    middlewares = PromptServer.instance.app.middlewares
    if view_subfolder_middleware not in middlewares:
        middlewares.append(view_subfolder_middleware)
