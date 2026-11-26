"""Run one ComfyUI request in a disposable Diffusers process."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from diffusers import QwenImage21Pipeline
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from attention import RoleAttention
from masks import person_masks


def grid(path, size):
    with Image.open(path) as mask:
        values = np.asarray(mask.convert("L").resize(size, Image.Resampling.BOX)) >= 128
    if not values.any():
        raise ValueError(f"Empty role foreground in {path.name}. Paint a larger mask.")
    return torch.from_numpy(values)


@torch.inference_mode()
def run(folder):
    job = json.loads((folder / "job.json").read_text())
    device = torch.device(job["device"])
    torch.cuda.set_device(device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    images = [Image.open(folder / f"role{i}.png").convert("RGB") for i in range(2)]
    ref = [folder / f"reference{i}.png" for i in range(2)]
    target = [folder / f"target{i}.png" for i in range(2)]
    auto_target = any(not p.exists() for p in target)
    pipe = QwenImage21Pipeline.from_pretrained(job["model"], torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload(gpu_id=device.index)
    width, height = job["width"], job["height"]

    completed = 0
    def on_step(pipe, step, timestep, callback_kwargs):
        nonlocal completed
        completed += 1
        temp = folder / "step.tmp"
        temp.write_text(str(completed))
        temp.replace(folder / "step")
        return callback_kwargs

    def sample():
        rng = torch.Generator(device=device).manual_seed(job["seed"])
        return pipe(prompt=job["prompt"], image=images, generator=rng, width=width, height=height,
                    output_resolution=768, num_inference_steps=job["steps"], true_cfg_scale=1.0,
                    use_kv_cache=True, callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=[]).images[0].convert("RGB")

    if job["compare"] or auto_target:
        baseline = sample()
        baseline.save(folder / "baseline.png")
    if auto_target or any(not p.exists() for p in ref):
        from transformers import Sam3Model, Sam3Processor
        processor = Sam3Processor.from_pretrained(job["sam3"], local_files_only=True)
        model = Sam3Model.from_pretrained(job["sam3"], local_files_only=True).eval()
        for path, img in zip(ref, images):
            if not path.exists():
                mask, = person_masks(img, 1, model, processor)
                Image.fromarray(mask.astype("uint8") * 255).save(path)
        if auto_target:
            masks = person_masks(baseline, 2, model, processor)
            if job["reverse"]:
                masks.reverse()
            for path, mask in zip(target, masks):
                if not path.exists():
                    Image.fromarray(mask.astype("uint8") * 255).save(path)
        del model, processor
    sizes = [calculate_dimensions(768**2, im.width / im.height)[:2] for im in images]
    ref_masks = [grid(p, (w//16, h//16)) for p, (w,h) in zip(ref, sizes)]
    target_masks = [grid(p, (width//16, height//16)) for p in target]
    for block in pipe.transformer.transformer_blocks:
        block.attn.set_processor(RoleAttention(ref_masks, target_masks, job["matching"], job["other_penalty"]))
    sample().save(folder / "enhanced.png")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
