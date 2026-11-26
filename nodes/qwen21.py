"""ComfyUI roles and generation for the Qwen-Image 2.1 Diffusers backend."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .h3_ref import H3RefRole, overview
from ..freefuse_core.h3_ref_editor import editor_context

CATEGORY = "BetterRef/Qwen Image 2.1"
WORKER = Path(__file__).resolve().parents[1] / "qwen21" / "worker.py"
ORDER = ["Picture 1 left, Picture 2 right", "Picture 2 left, Picture 1 right"]


class QwenRole(H3RefRole):
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BETTERREF_QWEN21_ROLES",)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["optional"]["previous"] = cls.RETURN_TYPES
        return inputs


def write_image(path, tensor):
    array = tensor[0].detach().cpu().clamp(0, 1).mul(255).round().byte().numpy()
    Image.fromarray(array).save(path)


def read_image(path, mode):
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert(mode))).float()[None] / 255


def run_worker(folder, interpreter, total_steps):
    from comfy.model_management import throw_exception_if_processing_interrupted
    from comfy.utils import ProgressBar
    progress = ProgressBar(total_steps)
    step_file = folder / "step"
    previous = 0
    # File-backed diagnostics avoid deadlocking on full pipes, including during cancellation.
    with (folder / "worker.log").open("w+b") as log:
        proc = subprocess.Popen([interpreter, "-u", str(WORKER), str(folder)], stdout=log, stderr=log)
        try:
            while proc.poll() is None:
                throw_exception_if_processing_interrupted()
                if step_file.is_file():
                    step = int(step_file.read_text())
                    if step != previous:
                        progress.update_absolute(step)
                        previous = step
                try:
                    proc.wait(timeout=.4)
                except subprocess.TimeoutExpired:
                    continue
            if proc.returncode:
                log.seek(max(0, log.tell() - 4800))
                raise RuntimeError("Qwen-Image 2.1 worker failed.\n" + log.read().decode(errors="replace"))
            progress.update_absolute(total_steps)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


class QwenGenerate:
    CATEGORY = CATEGORY
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("enhanced", "mask_overview", "baseline")
    DESCRIPTION = ("Generate two characters from two reference photos. SAM3 finds missing masks. "
                   "Paint both target masks and disable compare to skip the baseline pass.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "roles": (QwenRole.RETURN_TYPES[0],),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
            "width": ("INT", {"default": 1152, "min": 128, "max": 2048, "step": 32}),
            "height": ("INT", {"default": 768, "min": 128, "max": 2048, "step": 32}),
            "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
            "target_order": (ORDER,),
            "compare": ("BOOLEAN", {"default": True}),
            "matching": ("FLOAT", {"default": 2.0, "min": 0, "max": 32}),
            "other_penalty": ("FLOAT", {"default": 4.0, "min": 0, "max": 32}),
        }}

    def generate(self, roles, prompt, seed, width, height, steps, target_order, compare, matching, other_penalty):
        import comfy.model_management as mm
        import folder_paths
        from comfy_execution.graph import ExecutionBlocker
        if len(roles) != 2:
            raise ValueError("Connect two Qwen role cards in reference order.")
        if width % 32 or height % 32:
            raise ValueError("Canvas dimensions must be multiples of 32.")
        for role in roles:
            if not role["target"] or role["target"] not in prompt:
                raise ValueError(f"The target phrase for {role['name']} must occur in the scene prompt.")
        for i, role in enumerate(roles, 1):
            if role["reference"] and role["reference"] != role["target"]:
                prompt += f"\nPicture {i} reference: {role['reference']}."
        root = Path(folder_paths.models_dir)
        model_dirs = [Path(p) / "Qwen-Image-2.1" for p in folder_paths.get_folder_paths("diffusers")]
        model = next((p for p in model_dirs if (p / "model_index.json").is_file()), None)
        if model is None:
            raise ValueError(f"Place the complete Diffusers Qwen-Image-2.1 folder under {root / 'diffusers'}.")
        sam3 = root / "sam3"
        if any(r[key] is None for r in roles for key in ("reference_mask", "target_mask")) and not (sam3 / "config.json").is_file():
            raise ValueError(f"Place SAM3's Transformers files under {sam3}, or supply all four role masks.")
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise ValueError("The Qwen-Image 2.1 BetterRef worker requires a CUDA device with BF16 support.")
        with tempfile.TemporaryDirectory(prefix="betterref-qwen21-") as tmp:
            folder = Path(tmp)
            for i, role in enumerate(roles):
                write_image(folder / f"role{i}.png", role["image"][..., :3])
                for side in ("reference", "target"):
                    mask = role[side + "_mask"]
                    if mask is not None:
                        if mask.ndim != 3 or mask.shape[0] != 1:
                            raise ValueError(f"Supply a single {side} mask per role.")
                        write_image(folder / f"{side}{i}.png", mask)
            job = dict(prompt=prompt, seed=seed, width=width, height=height, steps=steps,
                       reverse=target_order == ORDER[1], compare=compare, matching=matching,
                       other_penalty=other_penalty, model=str(model), sam3=str(sam3),
                       device=f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}")
            (folder / "job.json").write_text(json.dumps(job, ensure_ascii=False))
            mm.unload_all_models()
            mm.soft_empty_cache()
            needs_baseline = compare or any(role["target_mask"] is None for role in roles)
            run_worker(folder, os.environ.get("BETTERREF_QWEN_PYTHON") or sys.executable,
                       steps * (2 if needs_baseline else 1))
            enhanced = read_image(folder / "enhanced.png", "RGB")
            baseline_path = folder / "baseline.png"
            baseline = read_image(baseline_path, "RGB") if baseline_path.is_file() else None
            masks = {"refs": {}, "target": {}}
            for i, role in enumerate(roles):
                name = role["name"]
                masks["refs"][name] = {name: read_image(folder / f"reference{i}.png", "L")}
                masks["target"][name] = read_image(folder / f"target{i}.png", "L")
            # The overview displays overlap on the same 16-pixel grid used by the processor.
            target_grid = (height//16, width//16)
            targets = [F.interpolate(mask[:, None], size=target_grid, mode="area")[:, 0] >= .5
                       for mask in masks["target"].values()]
            scene = {"roles": roles, "geometry": {"target": {"grid": (1, *target_grid)}}}
            context = editor_context(roles, baseline, masks, seed, (1, height//32, width//32))
            preview = overview(scene, masks, baseline, targets[0] & targets[1])
            return {"result": (enhanced, preview, baseline if compare else ExecutionBlocker(None)),
                    "ui": {"betterref_editor": [context]}}
