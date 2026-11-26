"""Read-only native H3-Image attention collection with expanded DiT token indices."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import torch

from .minimax_h3_support import make_mask_volumes, similarity_logits


def _image_grid(ref, index):
    if not isinstance(ref, Mapping) or ref.get("kind") != "image":
        raise ValueError(f"H3-Image reference {index} must have kind 'image'.")
    h, w = ref.get("latent_h"), ref.get("latent_w")
    if any(type(v) is not int or v <= 0 or v % 2 for v in (h, w)):
        raise ValueError(f"H3-Image reference {index} latent_h/latent_w must be positive even integers.")
    if ref.get("latent_t", 1) != 1:
        raise ValueError(f"H3-Image reference {index} must contain one image (t=1).")
    latent = ref.get("latent")
    if latent is not None:
        shape = tuple(latent.shape)
        if len(shape) != 5 or shape[0] != 1 or shape[2:] != (1, h, w):
            raise ValueError(f"H3-Image reference {index} latent must be [1,C,1,{h},{w}].")
    return (1, h // 2, w // 2)


def ref_geometry(layout, refs):
    """Return native half-open spans and independently validated reference grids."""
    refs = tuple(refs)
    grids = tuple(_image_grid(ref, index) for index, ref in enumerate(refs))
    signature = tuple(layout.signature)
    if len(signature) != 5:
        raise ValueError("H3-Image requires a five-entry PackedLayout.signature.")
    text_len, t, h, w, audio_t = signature
    if t != 1:
        raise ValueError("H3-Image Phase-0 requires a t=1 target.")
    if audio_t != 0:
        raise ValueError("H3-Image Phase-0 does not accept an audio target.")
    if type(text_len) is not int or text_len <= 0:
        raise ValueError("H3-Image requires a nonempty native text span.")
    if any(type(v) is not int or v <= 0 or v % 2 for v in (h, w)):
        raise ValueError("H3-Image target latent dimensions must be positive even integers.")
    segments = tuple(tuple(segment) for segment in layout.segments)
    kinds = tuple(segment[2] for segment in segments)
    expected = ("text",) + ("ref_img",) * len(refs) + ("video",)
    if kinds.count("ref_img") != len(refs):
        raise ValueError("H3-Image reference metadata count does not match real ref_img spans.")
    if kinds != expected:
        raise ValueError("H3-Image requires packed [text | ref_img ... | video] spans only.")
    cursor = 0
    for start, stop, _ in segments:
        if (type(start) is not int or type(stop) is not int
                or start != cursor or stop <= start):
            raise ValueError("H3-Image PackedLayout spans must be nonempty and contiguous.")
        cursor = stop
    if cursor != layout.seq_len:
        raise ValueError("H3-Image PackedLayout spans do not match seq_len.")
    text = segments[0][:2]
    if text[1] - text[0] != text_len:
        raise ValueError("H3-Image native text span does not match layout.signature.")
    references = []
    for index, (segment, grid) in enumerate(zip(segments[1:-1], grids)):
        span = segment[:2]
        if span[1] - span[0] != math.prod(grid):
            raise ValueError(f"H3-Image reference {index} span does not match its patch grid {grid}.")
        references.append((span, grid))
    target, grid = segments[-1][:2], (1, h // 2, w // 2)
    if target[1] - target[0] != math.prod(grid):
        raise ValueError("H3-Image target video span does not match its patch grid.")
    return text, tuple(references), target, grid


@dataclass
class Collection:
    """Zero-based capture state; advance current_step to step + 1 in the sampler callback."""
    positions: dict
    ref_names: tuple
    refs: tuple
    steps: tuple = (0, 1, 2)
    blocks: tuple = (10, 20, 30, 40)
    current_step: int = 0
    top_k_ratio: float = 0.1
    captures: dict = field(default_factory=dict)
    ref_shapes: tuple = field(init=False)
    active_subjects: tuple | None = None
    reference_positions: dict | None = None

    def __post_init__(self):
        self.refs = tuple(self.refs)
        self.ref_names = tuple(self.ref_names)
        grids = tuple(_image_grid(ref, index) for index, ref in enumerate(self.refs))
        self.ref_shapes = tuple((1, grid[1] * 2, grid[2] * 2) for grid in grids)
        if len(self.ref_names) != len(self.refs):
            raise ValueError("H3-Image needs one subject id per reference.")
        if (not self.ref_names or any(not isinstance(name, str) or not name
                                      or name.startswith("__") for name in self.ref_names)
                or len(set(self.ref_names)) != len(self.ref_names)):
            raise ValueError("H3-Image reference subject ids must be nonempty and unique.")
        positions = {}
        for name, indices in self.positions.items():
            if not isinstance(name, str) or not name or (name.startswith("__") and name != "__background__"):
                raise ValueError("H3-Image concept ids must be subjects or __background__.")
            indices = tuple(indices)
            if not indices or any(type(index) is not int or index < 0 for index in indices):
                raise ValueError(f"H3-Image requires native DiT token positions for {name!r}.")
            positions[name] = indices
        self.positions = positions
        active = self.ref_names if self.active_subjects is None else tuple(self.active_subjects)
        if set(active) - set(self.ref_names) or set(active) - set(positions):
            raise ValueError("H3-Image reference names need matching subject positions.")
        if self.reference_positions is None:
            self.reference_positions = self.positions
        if self.reference_positions.keys() != self.positions.keys():
            raise ValueError("Reference and target token maps must describe the same roles.")
        for name in ("steps", "blocks"):
            values = tuple(getattr(self, name))
            if not values or any(type(index) is not int or index < 0 for index in values):
                raise ValueError(f"H3-Image {name} must be nonempty zero-based indices.")
            setattr(self, name, values)
        if not math.isfinite(self.top_k_ratio) or not 0 < self.top_k_ratio <= 1:
            raise ValueError("H3-Image top_k_ratio must be finite and in (0,1].")


def spatial_concept_scores(q, k, positions, span):
    """Spatial-softmax diagnostics, not probabilities from the full attention matrix."""
    start, stop = span
    queries = q[:, :, start:stop].float()
    scores = {}
    for name, indices in positions.items():
        logits = torch.matmul(queries, k[:, :, indices].float().transpose(-1, -2))
        scores[name] = (logits / math.sqrt(q.shape[-1])).softmax(dim=2).mean(1).mean(-1)
    return scores


class H3ImageAttention:
    """Omit container_function so ComfyUI unwraps native Q/K/V before delegation."""
    def __init__(self, layout, collection, block, previous=None):
        self.layout = layout
        self.collection = collection
        self.block = block
        self.previous = previous

    def __call__(self, original, q, k, v, heads, *args, **kwargs):
        state = self.collection
        options = kwargs.get("transformer_options", {})
        step, block = state.current_step, options.get("block_index")
        collecting = (block == self.block and block in state.blocks
                      and step in state.steps and (step, block) not in state.captures
                      and options.get("cond_or_uncond", [0]) == [0])
        if collecting:
            text, references, target, grid = ref_geometry(self.layout, state.refs)
            if (not kwargs.get("skip_reshape") or q.ndim != 4
                    or q.shape[0] != 1 or q.shape[1] != heads
                    or q.shape[2] != self.layout.seq_len
                    or k.shape != q.shape or v.shape != q.shape):
                raise ValueError("H3-Image attention must expose native packed Q/K/V; do not capture token_refiner.")
            for name, indices in (*state.positions.items(), *state.reference_positions.items()):
                if (not indices or any(type(index) is not int or not text[0] <= index < text[1]
                                       for index in indices)):
                    raise ValueError(f"H3-Image concept {name!r} lies outside the native text span {text}.")
        if self.previous is None:
            output = original(q, k, v, heads, *args, **kwargs)
        else:
            output = self.previous(original, q, k, v, heads, *args, **kwargs)
        if collecting:
            if output.shape != (q.shape[0], self.layout.seq_len, heads * q.shape[-1]):
                raise ValueError("H3-Image requires native attention output [B,S,heads*D].")
            with torch.no_grad():
                def image_capture(span, image_grid, positions, keep=None):
                    # Compete before filtering so foreign-subject cores cannot dominate the reference.
                    logits = similarity_logits(q, k, output, positions, span, state.top_k_ratio)
                    scores = spatial_concept_scores(q, k, positions, span)
                    return {"span": span, "grid": image_grid,
                            "logits": {name: values.detach().cpu() for name, values in logits.items()
                                       if keep is None or name in keep},
                            "spatial_scores": {name: values.detach().cpu() for name, values in scores.items()}}

                ref_captures = {}
                for name, (span, image_grid) in zip(state.ref_names, references):
                    ref_captures[name] = image_capture(span, image_grid, state.reference_positions, {name, "__background__"})
                state.captures[(step, block)] = {
                    "step": step, "block": block, "text_span": text,
                    "signature": tuple(self.layout.signature),
                    "segments": tuple(tuple(segment) for segment in self.layout.segments),
                    "positions": dict(state.positions), "ref_names": state.ref_names,
                    "target": image_capture(target, grid, state.positions), "refs": ref_captures,
                }
        return output


class H3ImageBlock:
    """Clone-local collection on a selected double_block, composing prior patches."""
    def __init__(self, collection, block, previous=None):
        self.collection = collection
        self.block = block
        self.previous = previous

    def __call__(self, args, extra):
        def original_with_readout(block_args):
            state = self.collection
            options = block_args["transformer_options"]
            block = options.get("block_index")
            active = (block == self.block and block in state.blocks
                      and state.current_step in state.steps
                      and (state.current_step, block) not in state.captures
                      and options.get("cond_or_uncond", [0]) == [0])
            if not active:
                return extra["original_block"](block_args)
            options = dict(options)
            options["optimized_attention_override"] = H3ImageAttention(
                block_args["layout"], state, self.block,
                previous=options.get("optimized_attention_override"))
            return extra["original_block"](dict(block_args, transformer_options=options))

        # Compose at the delegate boundary after earlier patches set their attention overrides.
        if self.previous is not None:
            return self.previous(args, dict(extra, original_block=original_with_readout))
        return original_with_readout(args)


def install_patches(model, collection):
    """Copy modified option dictionaries so even shallow clones remain isolated."""
    count = len(model.model.diffusion_model.blocks)
    if any(index < 0 or index >= count for index in collection.blocks):
        raise ValueError(f"H3-Image collection block must be in 0..{count - 1}.")
    model_options = dict(model.model_options)
    options = dict(model_options.get("transformer_options", {}))
    replacements = dict(options.get("patches_replace", {}))
    blocks = dict(replacements.get("dit", {}))
    replacements["dit"] = blocks
    options["patches_replace"] = replacements
    model_options["transformer_options"] = options
    model.model_options = model_options
    for index in dict.fromkeys(collection.blocks):
        previous = blocks.get(("double_block", index))
        model.set_model_patch_replace(H3ImageBlock(collection, index, previous),
                                      "dit", "double_block", index)


def patch_model(model, collection):
    """Return a readout-only clone; the original patcher and model are unchanged."""
    clone = model.clone()
    # Preserve callable and manager identity while isolating the dictionaries.
    clone.model_options = dict(model.model_options)
    install_patches(clone, collection)
    return clone


def masks_from_capture(capture, temperature=1.0, include_background=True, **mask_options):
    """Reference masks compete only with their background, not absent subjects."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("H3-Image mask readout requires a positive temperature.")
    if "temporal_window" in mask_options:
        raise ValueError("H3-Image mask readout fixes temporal_window=1.")

    def image_masks(image, owner=None):
        grid, span = tuple(image["grid"]), tuple(image["span"])
        if (len(grid) != 3 or grid[0] != 1
                or any(type(value) is not int or value <= 0 for value in grid)):
            raise ValueError("H3-Image mask readout requires a positive t=1 grid.")
        if len(span) != 2 or span[1] - span[0] != math.prod(grid):
            raise ValueError("H3-Image captured mask grid does not match its span.")
        logits = image["logits"]
        if owner is not None:
            if owner not in logits:
                raise ValueError(f"H3-Image reference capture is missing owner {owner!r} logits.")
            logits = {name: value for name, value in logits.items()
                      if name in {owner, "__background__"}}
        if not logits or any(tuple(values.shape) != (1, math.prod(grid)) for values in logits.values()):
            raise ValueError("H3-Image captured logits must be [1, image patch rows].")
        return make_mask_volumes(logits, grid, temperature=temperature,
                                 temporal_window=1, include_background=include_background,
                                 **mask_options)

    return {"target": image_masks(capture["target"]),
            "refs": {name: image_masks(image, name) for name, image in capture["refs"].items()}}
