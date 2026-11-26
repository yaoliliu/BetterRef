"""Clone-local H3-Image target-to-reference foreground attention bias."""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType, SimpleNamespace

import torch

from .minimax_h3_image_support import ref_geometry
from .minimax_h3_support import augmented_attention


@dataclass(frozen=True, init=False)
class ReferenceBias:
    signature: tuple
    segments: tuple
    ref_names: tuple
    refs: tuple
    target_span: tuple
    target_grid: tuple
    masks: Mapping
    positions: Mapping
    condition_bias: Mapping
    positive: float
    negative: float

    def __init__(self, capture, masks, positive=2.0, negative=4.0, condition_bias=None, active_subjects=None):
        if any(not isinstance(value, Real) or not math.isfinite(value) or value < 0
               for value in (positive, negative)):
            raise ValueError("H3-Image reference bias magnitudes must be finite and nonnegative.")
        try:
            names = tuple(capture["ref_names"])
            signature = tuple(capture["signature"])
            segments = tuple(tuple(segment) for segment in capture["segments"])
            target = capture["target"]
            references = capture["refs"]
            target_values, ref_values = masks["target"], masks["refs"]
        except (KeyError, TypeError) as error:
            raise ValueError("H3-Image requires capture metadata and target/reference masks.") from error
        if (not names or any(not isinstance(name, str) or not name or name.startswith("__")
                             for name in names) or len(set(names)) != len(names)):
            raise ValueError("H3-Image reference subject ids must be nonempty and unique.")
        if (not isinstance(references, Mapping) or set(references) != set(names)
                or not isinstance(ref_values, Mapping) or set(ref_values) != set(names)):
            raise ValueError("H3-Image reference captures and masks must match explicit ref_names.")
        if (not isinstance(target_values, Mapping)
                or set(target_values) - {"__background__"} != set(names)):
            raise ValueError("H3-Image target masks must contain exactly the reference subjects.")
        target_span, target_grid = _image_snapshot(target)
        images = tuple(_image_snapshot(references[name]) for name in names)
        refs = tuple(MappingProxyType({"kind": "image", "latent_t": 1,
                                      "latent_h": 2 * grid[1], "latent_w": 2 * grid[2]})
                     for _, grid in images)
        if not segments or any(len(segment) != 3 for segment in segments):
            raise ValueError("H3-Image requires packed capture segments.")
        layout = SimpleNamespace(signature=signature, segments=segments, seq_len=segments[-1][1])
        _, actual_refs, actual_target, actual_grid = ref_geometry(layout, refs)
        if images != actual_refs or (target_span, target_grid) != (actual_target, actual_grid):
            raise ValueError("H3-Image captured span/grid must match native packed geometry.")
        active = set(names if active_subjects is None else active_subjects)
        if active - set(names):
            raise ValueError("Active reference roles must have a packed reference slot.")
        target_masks, positions = {}, {}
        for name, value in target_values.items():
            checked = _mask_snapshot(value, target_grid, f"target {name!r}",
                                     foreground=name in active)
            if name in active:
                target_masks[name] = checked
        for name, (span, grid) in zip(names, images):
            values = ref_values[name]
            if (not isinstance(values, Mapping) or name not in values
                    or set(values) - {name, "__background__"}):
                raise ValueError(f"H3-Image reference {name!r} masks require its owner and optional background only.")
            for concept, value in values.items():
                checked = _mask_snapshot(value, grid, f"reference {name!r} {concept!r}",
                                         foreground=concept == name and name in active, binary=concept == name)
                if concept == name and name in active:
                    positions[name] = tuple(span[0] + index for index in
                                            torch.nonzero(checked.reshape(-1), as_tuple=True)[0].tolist())
        extras = {}
        for name, entries in (condition_bias or {}).items():
            if name not in names or any(type(i) != int or not 0 <= i < signature[0]
                                        or not math.isfinite(v) or v < 0 for i, v in entries.items()):
                raise ValueError("Invalid subject condition-token bias.")
            if entries:
                extras[name] = (torch.tensor(list(entries), dtype=torch.long),
                                torch.tensor(list(entries.values()), dtype=torch.float32))
        values = {"signature": signature, "segments": segments,
                  "ref_names": names, "refs": refs, "target_span": target_span,
                  "target_grid": target_grid, "masks": MappingProxyType(target_masks),
                  "positions": MappingProxyType(positions),
                  "condition_bias": MappingProxyType(extras), "positive": float(positive), "negative": float(negative)}
        for name, value in values.items():
            object.__setattr__(self, name, value)


def _image_snapshot(image):
    try:
        span, grid = tuple(image["span"]), tuple(image["grid"])
    except (KeyError, TypeError) as error:
        raise ValueError("H3-Image capture needs each image span/grid.") from error
    if (len(grid) != 3 or grid[0] != 1
            or any(type(value) is not int or value <= 0 for value in grid)):
        raise ValueError("H3-Image reference bias requires positive t=1 image grids.")
    if (len(span) != 2 or any(type(value) is not int or value < 0 for value in span)
            or span[1] - span[0] != math.prod(grid)):
        raise ValueError("H3-Image captured span must match its image grid.")
    return span, grid


def _mask_snapshot(value, grid, label, foreground=False, binary=False):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != grid or value.is_complex():
        raise ValueError(f"H3-Image {label} mask shape must be {grid} with real values.")
    value = value.detach().cpu().clone()
    if binary:
        if not bool((torch.isfinite(value) & ((value == 0) | (value == 1))).all()):
            raise ValueError(f"H3-Image {label} foreground must be finite binary 0/1.")
    elif not bool((torch.isfinite(value) & (value >= 0) & (value <= 1)).all()):
        raise ValueError(f"H3-Image {label} mask must be finite in [0,1].")
    if foreground and not bool(value.any()):
        raise ValueError(f"H3-Image {label} foreground is empty; no fallback is allowed.")
    return value


class H3ImageRefAttention:
    """Omit container_function so ComfyUI unwraps native Q/K/V before delegation."""
    def __init__(self, layout, bias, previous=None):
        self.layout = layout
        self.bias = bias
        self.previous = previous

    def __call__(self, original, q, k, v, heads, *args, **kwargs):
        def call(q, k, v, heads, **options):
            if self.previous is None:
                return original(q, k, v, heads, *args, **options)
            return self.previous(original, q, k, v, heads, *args, **options)

        options = kwargs.get("transformer_options", {})
        if (not self.bias.masks or self.bias.positive == 0 and self.bias.negative == 0 and not self.bias.condition_bias
                or options.get("cond_or_uncond", [0]) != [0]):
            return call(q, k, v, heads, **kwargs)
        if (tuple(self.layout.signature) != self.bias.signature
                or tuple(tuple(segment) for segment in self.layout.segments) != self.bias.segments):
            raise ValueError("H3-Image actual layout must match the captured signature/segments snapshot.")
        _, _, target, grid = ref_geometry(self.layout, self.bias.refs)
        if (target, grid) != (self.bias.target_span, self.bias.target_grid):
            raise ValueError("H3-Image actual target span/grid must match the capture snapshot.")
        if (kwargs.get("skip_reshape") is not True or q.ndim != 4
                or q.shape[0] != 1 or q.shape[1] != heads
                or q.shape[2] != self.layout.seq_len or q.shape[-1] <= 0
                or k.shape != q.shape or v.shape != q.shape):
            raise ValueError("H3-Image reference bias requires native post-RoPE packed Q/K/V [1,H,S,D].")

        return augmented_attention(call, q, k, v, heads, self.bias.masks,
                                   self.bias.positions, target, negative=self.bias.negative,
                                   positive=self.bias.positive, condition_bias=self.bias.condition_bias, **kwargs)


class H3ImageRefBlock:
    def __init__(self, bias, block, previous=None):
        self.bias = bias
        self.block = block
        self.previous = previous

    def __call__(self, args, extra):
        def original_with_bias(block_args):
            options = block_args["transformer_options"]
            if (options.get("block_index") != self.block
                    or options.get("cond_or_uncond", [0]) != [0]
                    or self.bias.positive == 0 and self.bias.negative == 0 and not self.bias.condition_bias):
                return extra["original_block"](block_args)
            options = dict(options)
            options["optimized_attention_override"] = H3ImageRefAttention(
                block_args["layout"], self.bias,
                previous=options.get("optimized_attention_override"))
            return extra["original_block"](dict(block_args, transformer_options=options))

        # Install after prior block patches have chosen their attention override.
        if self.previous is not None:
            return self.previous(args, dict(extra, original_block=original_with_bias))
        return original_with_bias(args)


def _copy_dicts(value):
    return {key: _copy_dicts(item) for key, item in value.items()} if isinstance(value, dict) else value


def patch_reference_model(model, bias, blocks=None):
    count = len(model.model.diffusion_model.blocks)
    indices = tuple(range(count)) if blocks is None else tuple(blocks)
    if any(type(index) is not int or index < 0 or index >= count for index in indices):
        raise ValueError(f"H3-Image reference bias block must be in 0..{count - 1}.")
    clone = model.clone()
    clone.model_options = _copy_dicts(model.model_options)
    if (bias.positive == 0 and bias.negative == 0 and not bias.condition_bias) or not indices:
        return clone
    options = clone.model_options.setdefault("transformer_options", {})
    replacements = options.setdefault("patches_replace", {}).setdefault("dit", {})
    for index in dict.fromkeys(indices):
        previous = replacements.get(("double_block", index))
        clone.set_model_patch_replace(H3ImageRefBlock(bias, index, previous),
                                      "dit", "double_block", index)
    return clone
