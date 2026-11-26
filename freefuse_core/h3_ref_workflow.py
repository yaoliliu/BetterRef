"""H3 Image role bindings, native token spans and editable patch ownership."""

import math

import torch
import torch.nn.functional as F

from .token_offsets import native_byte_offsets


def phrase_ranges(text, phrase):
    ranges = []
    raw, term = text.encode(), phrase.encode()
    start = 0
    while term:
        found = raw.find(term, start)
        if found < 0:
            break
        start = found + len(term)
        ranges.append((found, start))
    if not ranges:
        raise ValueError(f"Phrase {phrase!r} must occur verbatim in the scene prompt.")
    return ranges


def bind_prompt(scene, roles, background):
    prompt = scene
    target, reference = {}, {}
    if any(role["target"] for role in roles):
        target["__background__"] = reference["__background__"] = phrase_ranges(scene, background)
    for index, role in enumerate(roles, 1):
        name, phrase, caption = role["name"], role["target"], role["reference"]
        if not phrase:
            continue
        target[name] = phrase_ranges(scene, phrase)
        reference[name] = target[name]
        if caption and caption != phrase:
            prompt += f"\nPicture {index} reference: "
            start = len(prompt.encode())
            prompt += caption
            reference[name] = [(start, len(prompt.encode()))]
            prompt += "."
    return prompt, target, reference


def binding_summary(roles):
    rows = []
    for i, role in enumerate(roles, 1):
        target, reference = role["target"], role["reference"]
        heading = f"Picture {i} · {role['name']}"
        if not target:
            rows.append(f"{heading} — 仅图像参考 / Image only")
        elif not reference or reference == target:
            rows.append(f"{heading} — 沿用场景短语 / From scene\n  {target}")
        else:
            rows.append(f"{heading} — 独立参考描述 / Reference caption\n  {reference}")
    return "\n\n".join(rows)


def resolve_positions(clip, prompt, tokens, positive, bindings):
    if not bindings:
        return {}
    plain = clip.tokenize(prompt)["qwen3vl_32b"]
    presented = tokens["qwen3vl_32b"]
    if len(plain) != 1 or len(presented) != 1 or len(positive) != 1:
        raise ValueError("Expected one native H3 Image prompt and one conditioning entry.")
    ids = [t[0] for t in plain[0]]
    if not ids or [t[0] for t in presented[0][-len(ids) :]] != ids:
        raise ValueError("Cannot align the original prompt with the image-conditioned tokens.")
    context, meta = positive[0]
    prefix = context.shape[1] - len(ids)
    tags = meta["minimax_token_tags"]
    if prefix < 0 or tags.numel() != context.shape[1] or not bool((tags[prefix:] == 1).all()):
        raise ValueError("Unexpected H3 Image text/vision layout.")
    offsets = native_byte_offsets(clip.tokenizer.qwen3vl_32b.tokenizer, ids, prompt)
    result = {}
    for name, ranges in bindings.items():
        indices = tuple(
            prefix + i
            for i, (a, b) in enumerate(offsets)
            if any(a < end and b > start for start, end in ranges)
        )
        if not indices:
            raise ValueError(f"No native tokens for {name!r}.")
        result[name] = indices
    return result


def packed_geometry(text_length, refs, names, height, width):
    """Native packed geometry for the manual-only path; ReferenceBias verifies it at use."""
    cursor = text_length
    segments = [(0, cursor, "text")]

    def image(grid, kind):
        nonlocal cursor
        start = cursor
        cursor += math.prod(grid)
        segments.append((start, cursor, kind))
        return {"grid": grid, "span": (start, cursor)}

    images = {
        name: image((1, ref["latent_h"] // 2, ref["latent_w"] // 2), "ref_img")
        for name, ref in zip(names, refs)
    }
    target = image((1, height // 32, width // 32), "video")
    return {
        "signature": (text_length, 1, height // 16, width // 16, 0),
        "segments": segments,
        "refs": images,
        "target": target,
        "ref_names": names,
    }


def painted(mask, grid):
    mask = mask.detach().float().cpu()
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3 or mask.shape[0] != 1 or not torch.isfinite(mask).all():
        raise ValueError("Each painted mask must describe one image with finite values.")
    return F.interpolate(mask[:, None], size=grid[1:], mode="nearest-exact")[:, 0] >= 0.5


def merge_masks(geometry, roles, automatic):
    """Manual target ownership wins everywhere; preserve conflicts for inspection."""
    target = {}
    refs = {}
    grid = geometry["target"]["grid"]
    taken = torch.zeros(grid, dtype=torch.bool)
    conflict = torch.zeros_like(taken)
    conflicts = set()
    manual = {}
    for role in roles:
        name = role["name"]
        if role["target"] and role["target_mask"] is not None:
            region = painted(role["target_mask"], grid)
            overlap = taken & region
            if overlap.any():
                conflicts.add(name)
                conflicts.update(n for n, m in manual.items() if (m & overlap).any())
            conflict |= overlap
            taken |= region
            manual[name] = region
    applied = []
    for role in roles:
        name = role["name"]
        ref_grid = geometry["refs"][name]["grid"]
        if not role["target"]:
            fg, ref = torch.zeros(grid), torch.zeros(ref_grid)
        else:
            if name in manual:
                fg = manual[name].float()
            else:
                fg = (automatic["target"][name].bool() & ~taken).float()
            mask = role["reference_mask"]
            ref = painted(mask, ref_grid).float() if mask is not None else automatic["refs"][name][name]
            if fg.any() and ref.any():
                applied.append(name)
        refs[name] = {name: ref, "__background__": 1 - ref}
        target[name] = fg
    target["__background__"] = 1 - torch.stack(list(target.values())).amax(0)
    return {"target": target, "refs": refs}, applied, conflict, sorted(conflicts)
