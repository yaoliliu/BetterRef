"""Produce baseline-only person masks, assigning the two subjects by prompted side."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

NAMES = ("jia_baoyu_1987", "lin_daiyu_1987")


def people(path, count, model, processor):
    image = Image.open(path).convert("RGB")
    for query in ("person", "character"):
        inputs = processor(images=image, text=query, return_tensors="pt").to(model.device)
        results = processor.post_process_instance_segmentation(
            model(**inputs), threshold=0.5, mask_threshold=0.5, target_sizes=[image.size[::-1]])
        masks = [m.cpu().numpy().astype(bool) for m in results[0]["masks"] if m.float().mean() >= 0.02]
        masks.sort(key=np.sum, reverse=True)
        if count == 1 and masks and (len(masks) == 1 or masks[0].sum() > 2 * masks[1].sum()):
            return masks[:1]
        if count == 2 and len(masks) == 2 and (masks[0] & masks[1]).sum() < .25 * min(m.sum() for m in masks):
            return sorted(masks, key=lambda m: np.where(m)[1].mean())
    raise ValueError(f"Ambiguous SAM3 person assignment: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("model", "refs", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    torch.set_num_threads(8)
    processor = Sam3Processor.from_pretrained(args.model, local_files_only=True)
    model = Sam3Model.from_pretrained(args.model, local_files_only=True).eval()
    jobs = [(args.refs / (n+".jpg"), args.out, [n]) for n in NAMES]
    jobs.extend((args.out / str(s) / "baseline.png", args.out / str(s), NAMES) for s in args.seeds)
    with torch.inference_mode():
        for src, folder, names in jobs:
            masks = people(src, len(names), model, processor)
            for name, mask in zip(names, masks):
                Image.fromarray(mask.astype("uint8")*255).save(folder/(name+".png"))
            print(src, json.dumps([round(float(m.mean()), 4) for m in masks]), flush=True)
