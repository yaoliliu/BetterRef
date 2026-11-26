"""Serve the two bundled references for painting before the first queue run."""
from pathlib import Path

from aiohttp import web
from server import PromptServer


@PromptServer.instance.routes.get("/betterref/v1/example/{role}")
async def reference_image(request):
    images = {"baoyu": "jia_baoyu_1987.jpg", "daiyu": "lin_daiyu_1987.jpg"}
    filename = images.get(request.match_info["role"])
    if filename is None:
        raise web.HTTPNotFound()
    directory = Path(__file__).resolve().parents[1] / "workflows" / "h3_ref_examples"
    return web.FileResponse(directory / filename)
