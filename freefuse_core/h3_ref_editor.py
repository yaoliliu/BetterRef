"""Portable PNG mask overrides and lightweight ComfyUI editor previews."""
import base64
import io
import json

import numpy as np
import torch
from PIL import Image


def decode_edits(value):
    edits = json.loads(value or "{}")
    if not isinstance(edits, dict) or set(edits) - {"reference", "target"}:
        raise ValueError("Invalid role mask document.")
    masks = {}
    for side, text in edits.items():
        prefix = "data:image/png;base64,"
        if not isinstance(text, str) or not text.startswith(prefix) or len(text) > 8_000_000:
            raise ValueError(f"Invalid {side} mask PNG.")
        with Image.open(io.BytesIO(base64.b64decode(text[len(prefix):], validate=True))) as image:
            if image.format != "PNG" or max(image.size) > 2048:
                raise ValueError("Role mask PNGs must be at most 2048 pixels per side.")
            gray = np.asarray(image.convert("L"))
            masks[side] = torch.from_numpy(gray >= 128).float().unsqueeze(0)
    return masks


def preview_data(tensor, mask=False):
    if tensor is None:
        return None
    data = (tensor[0].detach().cpu().float().clamp(0, 1) * 255).round().byte().numpy()
    image = Image.fromarray(data)
    stream = io.BytesIO()
    if mask:
        image.thumbnail((1024, 1024), Image.Resampling.NEAREST)
        image.point(lambda pixel: 255 if pixel >= 128 else 0, mode="1").save(stream, format="PNG")
        mime = "png"
    else:
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image.save(stream, format="JPEG", quality=86)
        mime = "jpeg"
    return f"data:image/{mime};base64," + base64.b64encode(stream.getvalue()).decode()


def editor_context(roles, baseline, automatic, seed, grid):
    entries = {}
    for role in roles:
        name = role["name"]
        entries[role.get("editor_id") or name] = {
            "name": name,
            **{side + "External": preview_data(mask, True)
               for side, mask in role.get("external_masks", {}).items() if mask is not None},
            "reference": preview_data(role["image"]),
            "referenceAuto": preview_data(automatic["refs"][name][name], True) if automatic else None,
            "targetAuto": preview_data(automatic["target"].get(name), True) if automatic else None,
        }
    return {"roles": entries, "baseline": preview_data(baseline), "seed": str(seed),
            "width": grid[2] * 32, "height": grid[1] * 32}
