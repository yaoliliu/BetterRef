"""Experimental MiniMax H3 Image multi-reference workflow nodes."""

import colorsys
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..freefuse_core.h3_ref_editor import decode_edits, editor_context
from ..freefuse_core.h3_ref_workflow import bind_prompt, binding_summary, resolve_positions, packed_geometry, merge_masks
from ..freefuse_core.minimax_h3_image_support import Collection, patch_model
from ..freefuse_core.minimax_h3_ref_attention import ReferenceBias, patch_reference_model
from ..freefuse_core.minimax_h3_ref_localization import apply_readout

CATEGORY = "BetterRef/H3 Image"
DEFAULT_WEIGHTS = "h3-image-v1.json"


def weight_files():
    import folder_paths

    paths = {DEFAULT_WEIGHTS: Path(__file__).resolve().parents[1] / "readouts" / DEFAULT_WEIGHTS}
    for path in sorted((Path(folder_paths.models_dir) / "betterref" / "readouts").glob("*.json")):
        paths["custom/" + path.name] = path
    return paths


class H3RefExamplePair:
    CATEGORY = CATEGORY
    FUNCTION = "images"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("jia_baoyu", "lin_daiyu")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def images(self):
        from PIL import Image, ImageOps
        import numpy as np

        folder = Path(__file__).resolve().parents[1] / "workflows" / "h3_ref_examples"

        def load(name):
            with Image.open(folder / name) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                return torch.from_numpy(np.array(rgb)).float()[None] / 255

        return load("jia_baoyu_1987.jpg"), load("lin_daiyu_1987.jpg")


class H3RefRole:
    CATEGORY = CATEGORY
    FUNCTION = "attach"
    RETURN_TYPES = ("H3_REF_ROLES",)
    RETURN_NAMES = ("roles",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "name": ("STRING", {"default": "A"}),
                "target_phrase": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False, "placeholder": "目标图 · 场景提示词中的角色描述"}),
                "reference_phrase": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False, "placeholder": "参考图 · 留空时沿用目标描述"}),
            },
            "optional": {
                "previous": ("H3_REF_ROLES",),
                "reference_mask": ("MASK",),
                "target_mask": ("MASK",),
                "mask_edits": ("STRING", {"default": "{}", "dynamicPrompts": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def attach(
        self,
        image,
        name,
        target_phrase,
        reference_phrase="",
        previous=(),
        reference_mask=None,
        target_mask=None,
        mask_edits="{}",
        unique_id=None,
    ):
        name, target, reference = name.strip(), target_phrase.strip(), reference_phrase.strip()
        if not name or name.startswith("__") or any(r["name"] == name for r in previous):
            raise ValueError(
                "Each role needs a unique, nonempty name without a double-underscore prefix."
            )
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Attach one reference image per role.")
        edits = decode_edits(mask_edits)
        external = {"reference": reference_mask, "target": target_mask}
        if reference_mask is None:
            reference_mask = edits.get("reference")
        if target_mask is None:
            target_mask = edits.get("target")
        if not target and (reference or reference_mask is not None or target_mask is not None):
            raise ValueError(f"Set a target phrase for {name} to bind its description or masks.")
        role = dict(
            name=name,
            external_masks=external,
            editor_id=str(unique_id) if unique_id is not None else name,
            image=image,
            target=target,
            reference=reference,
            reference_mask=reference_mask,
            target_mask=target_mask,
        )
        return ((*previous, role),)


class H3RefScene:
    CATEGORY = CATEGORY
    FUNCTION = "encode"
    RETURN_TYPES = ("H3_REF_SCENE", "STRING")
    RETURN_NAMES = ("scene", "encoded_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "roles": ("H3_REF_ROLES",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "background_phrase": ("STRING", {"default": "studio"}),
                "width": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 32}),
            }
        }

    def encode(self, clip, vae, roles, prompt, background_phrase, width, height):
        import node_helpers
        from comfy_extras.nodes_minimax_h3 import (
            _image_refs,
            _size_short_edge,
            IMAGE_REF_SHORT_EDGE,
            IMAGE_REF_VAE_MAX_SHORT_EDGE,
        )

        if width % 32 or height % 32:
            raise ValueError("H3 Image width and height must be multiples of 32.")
        text, target, reference = bind_prompt(prompt, roles, background_phrase.strip())
        images = {f"image_{i}": r["image"] for i, r in enumerate(roles, 1)}
        _, vision, refs = _image_refs(
            images,
            vae,
            lambda w, h: _size_short_edge(w, h, IMAGE_REF_SHORT_EDGE, IMAGE_REF_SHORT_EDGE),
            lambda w, h: _size_short_edge(w, h, IMAGE_REF_SHORT_EDGE, IMAGE_REF_VAE_MAX_SHORT_EDGE),
        )
        tokens = clip.tokenize(text, images=vision, picture_label="Picture %d: ")
        positive = clip.encode_from_tokens_scheduled(tokens)
        # Resolve both scopes in one tokenizer pass. Repeated words in the caption cannot leak into the scene.
        scopes = {
            (side, k): v
            for side, group in (("target", target), ("reference", reference))
            for k, v in group.items()
        }
        indices = resolve_positions(clip, text, tokens, positive, scopes)
        positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": refs})
        names = tuple(r["name"] for r in roles)
        geometry = packed_geometry(positive[0][0].shape[1], refs, names, height, width)
        scene = dict(
            roles=roles,
            vae=vae,
            positive=positive,
            negative=[[torch.zeros_like(x), dict(m)] for x, m in positive],
            refs=refs,
            positions={k: indices["target", k] for k in target},
            reference_positions={k: indices["reference", k] for k in reference},
            geometry=geometry,
            latent=torch.zeros(1, 24, 1, height // 16, width // 16),
        )
        return scene, text


def native_attention(original, q, k, v, heads, *args, **kwargs):
    from comfy.ldm.modules.attention import attention_pytorch

    return attention_pytorch(q, k, v, heads, *args, **kwargs)


def sample(model, scene, seed, *, state=None, stop=None):
    import comfy.sample

    steps = len(model.model.diffusion_model.pdd_sigmas) - 1
    latent = scene["latent"]

    def progress(step, x0, x, total):
        state.current_step = step + 1

    return comfy.sample.sample(
        model,
        comfy.sample.prepare_noise(latent, seed),
        steps,
        1.0,
        "euler",
        "simple",
        scene["positive"],
        scene["negative"],
        latent,
        seed=seed,
        callback=progress if state is not None else None,
        last_step=stop,
        force_full_denoise=stop is None,
    )


def decode(vae, samples):
    image = vae.decode(samples)
    return image.flatten(0, 1) if image.ndim == 5 else image


def overview(scene, masks, baseline, conflict, target_label=None):
    from PIL import Image, ImageDraw
    import numpy as np

    roles = scene["roles"]
    colors = [colorsys.hsv_to_rgb((0.54 + i * 0.61803398875) % 1, 0.79, 0.96) for i in range(len(roles))]

    def picture(img, mask_list, size, warning=None):
        w, h = size
        if img is None:
            base = torch.full((h, w, 3), 0.16)
        else:
            base = (
                F.interpolate(
                    img.detach().cpu().movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False
                )[0]
                .movedim(0, -1)
                .clamp(0, 1)
            )
        for i, mask in mask_list:
            opacity = (
                F.interpolate(mask[None].float(), size=(h, w), mode="nearest")[0].movedim(0, -1) * 0.47
            )
            base = base * (1 - opacity) + torch.tensor(colors[i]) * opacity
        if warning is not None and warning.any():
            bad = F.interpolate(warning[None].float(), size=(h, w), mode="nearest")[0, 0].bool()
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            base[bad & ((x + y) % 10 < 5)] = torch.tensor([1.0, 0.0, 0.85])
        return Image.fromarray((base * 255).byte().numpy())

    def fitted(w, h, wmax, hmax):
        scale = min(wmax / w, hmax / h)
        return max(1, round(w * scale)), max(1, round(h * scale))

    tw, th = scene["geometry"]["target"]["grid"][1:][::-1]
    size = fitted(tw, th, 850, 420)
    cols = min(5, len(roles))
    width = max(size[0] + 24, cols * 166 + 20)
    canvas = Image.new(
        "RGB", (width, 56 + size[1] + ((len(roles) + cols - 1) // cols) * 140 + 12), "#152128"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 9),
        "Target ownership  /  " + (target_label or ("same-seed baseline" if baseline is not None else "layout")),
        fill="white",
    )
    target = picture(
        baseline, [(i, masks["target"][r["name"]]) for i, r in enumerate(roles)], size, conflict
    )
    canvas.paste(target, (12, 30))
    for i, r in enumerate(roles):
        x = 12 + (i % cols) * 166
        y = size[1] + 47 + (i // cols) * 140
        img = r["image"]
        mask = masks["refs"][r["name"]][r["name"]]
        thumb = fitted(img.shape[2], img.shape[1], 76, 100)
        canvas.paste(picture(img, [], thumb), (x, y + 20))
        canvas.paste(picture(img, [(i, mask)], thumb), (x + 80, y + 20))
        draw.rectangle((x, y + 2, x + 10, y + 12), fill=tuple(round(c * 255) for c in colors[i]))
        draw.text((x + 16, y), f"Picture {i + 1}", fill="white")
    return torch.from_numpy(np.asarray(canvas).copy()).float()[None] / 255


class H3RefLocate:
    CATEGORY = CATEGORY
    FUNCTION = "locate"
    RETURN_TYPES = ("H3_REF_PLAN", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("plan", "mask_overview", "baseline", "status")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scene": ("H3_REF_SCENE",),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": False,
                    },
                ),
                "compare": ("BOOLEAN", {"default": True}),
                "readout": (list(weight_files()),),
            }
        }

    def locate(self, model, scene, seed, compare, readout):
        from comfy_execution.graph import ExecutionBlocker

        model = model.clone()
        model.model_options = dict(model.model_options)
        model.model_options["transformer_options"] = {
            **model.model_options.get("transformer_options", {}),
            "optimized_attention_override": native_attention,
        }
        roles = scene["roles"]
        active = [r["name"] for r in roles if r["target"]]
        automatic = None
        state = None
        stop = None
        geometry = scene["geometry"]
        if any(r["target"] and (r["target_mask"] is None or r["reference_mask"] is None) for r in roles):
            weights = json.loads(weight_files()[readout].read_bytes())
            keys = [tuple(key) for key in weights["capture_keys"]]
            steps = tuple(sorted({s for s, b in keys}))
            if not steps or steps[0] < 0 or steps[-1] >= len(model.model.diffusion_model.pdd_sigmas) - 1:
                raise ValueError("Readout capture steps must fit the native H3 Image schedule.")
            state = Collection(
                scene["positions"],
                geometry["ref_names"],
                tuple(scene["refs"]),
                steps=steps,
                blocks=tuple(sorted({b for s, b in keys})),
                active_subjects=active,
                reference_positions=scene["reference_positions"],
            )
            stop = None if compare else steps[-1] + 1
        baseline = None
        if state is not None or compare:
            latent = sample(
                patch_model(model, state) if state is not None else model,
                scene,
                seed,
                state=state,
                stop=stop,
            )
            if compare:
                baseline = decode(scene["vae"], latent)
            del latent
        if state is not None:
            automatic = apply_readout(state.captures, weights, active)
            geometry = state.captures[keys[0]]
        masks, applied, conflict, names = merge_masks(geometry, roles, automatic)
        status = binding_summary(roles)
        if names:
            status += (
                "\nOverlapping painted target masks: "
                + ", ".join(names)
                + ". Correct the striped area and queue again."
            )
        elif set(active) - set(applied):
            status += "\nEmpty foreground, bias skipped: " + ", ".join(
                n for n in active if n not in applied
            )
        # Keep only packed metadata in the cached plan; captures can be large.
        snapshot = {k: geometry[k] for k in ("signature", "segments", "ref_names")}

        def spatial(v):
            return {k: v[k] for k in ("span", "grid")}

        snapshot.update(
            target=spatial(geometry["target"]), refs={n: spatial(v) for n, v in geometry["refs"].items()}
        )
        plan = (model, scene, seed, snapshot, masks, applied, bool(names))
        result = (
            plan,
            overview(scene, masks, baseline, conflict),
            baseline if compare else ExecutionBlocker(None),
            status,
        )
        context = editor_context(roles, baseline, automatic, seed, geometry["target"]["grid"])
        return {"result": result, "ui": {"betterref_editor": [context]}}


class H3RefGenerate:
    CATEGORY = CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("H3_REF_PLAN",),
                "matching": ("FLOAT", {"default": 2.0, "min": 0, "max": 32}),
                "other_penalty": ("FLOAT", {"default": 4.0, "min": 0, "max": 32}),
            }
        }

    def generate(self, plan, matching, other_penalty):
        from comfy_execution.graph import ExecutionBlocker

        model, scene, seed, geometry, masks, active, conflict = plan
        if conflict:
            return (ExecutionBlocker(None),)
        if active and (matching or other_penalty):
            bias = ReferenceBias(geometry, masks, matching, other_penalty, active_subjects=active)
            model = patch_reference_model(model, bias)
        return (decode(scene["vae"], sample(model, scene, seed)),)


NODES = {
    "BetterRefH3Examples": H3RefExamplePair,
    "BetterRefH3Role": H3RefRole,
    "BetterRefH3Scene": H3RefScene,
    "BetterRefH3Locate": H3RefLocate,
    "BetterRefH3Generate": H3RefGenerate,
}
TITLES = dict(
    zip(
        NODES,
        (
            "1987 Hong Lou Meng References",
            "H3 Reference Role",
            "H3 Reference Scene",
            "H3 Role Masks & Baseline",
            "H3 Reference Generation",
        ),
    )
)
