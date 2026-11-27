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
    pipe = QwenImage21Pipeline.from_pretrained(job["model"], torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload(gpu_id=device.index)
    width, height = job["width"], job["height"]
    sizes = [calculate_dimensions(768**2, im.width / im.height)[:2] for im in images]
    ref_sizes = [(w//16, h//16) for w, h in sizes]
    target_size = (width//16, height//16)
    needs_masks = any(not path.exists() for path in ref + target)
    readout = None

    completed = 0
    def on_step(pipe, step, timestep, callback_kwargs):
        nonlocal completed
        if readout is not None:
            readout.advance(step, callback_kwargs["latents"])
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
                    callback_on_step_end_tensor_inputs=["latents"] if readout is not None else []).images[0].convert("RGB")

    if job["compare"]:
        baseline = sample()
        baseline.save(folder / "baseline.png")
    if needs_masks:
        from readout import LearnedReadout

        def select_masks(ref_auto, target_auto):
            def resolve(paths, predictions, sizes):
                masks = []
                for path, predicted, size in zip(paths, predictions, sizes):
                    if path.exists():
                        masks.append(grid(path, size))
                    else:
                        Image.fromarray(predicted.astype("uint8") * 255).save(path)
                        masks.append(torch.from_numpy(predicted))
                return masks
            ref_masks = resolve(ref, ref_auto, ref_sizes)
            target_masks = resolve(target, target_auto, [target_size] * 2)
            # With no own reference foreground, leave that target role's attention native.
            active_targets = [m if r.any() else torch.zeros_like(m) for r, m in zip(ref_masks, target_masks)]
            return ref_masks, active_targets

        readout = LearnedReadout(pipe, [size[::-1] for size in ref_sizes], target_size[::-1],
                                 select_masks, job["matching"], job["other_penalty"])
        if job["steps"] <= readout.last_step + 1:
            raise ValueError(f"Learned readout needs at least {readout.last_step + 2} sampling steps. "
                             "Increase steps or supply all four role masks.")
        readout.install()
        sample().save(folder / "enhanced.png")
        return
    ref_masks = [grid(path, size) for path, size in zip(ref, ref_sizes)]
    target_masks = [grid(path, target_size) for path in target]
    for block in pipe.transformer.transformer_blocks:
        block.attn.set_processor(RoleAttention(ref_masks, target_masks, job["matching"], job["other_penalty"]))
    sample().save(folder / "enhanced.png")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
