"""Produce baseline-only person masks, assigning the two subjects by prompted side."""
import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qwen21.masks import person_masks
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

NAMES = ("jia_baoyu_1987", "lin_daiyu_1987")


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
            with Image.open(src) as image:
                masks = person_masks(image.convert("RGB"), len(names), model, processor)
            for name, mask in zip(names, masks):
                Image.fromarray(mask.astype("uint8")*255).save(folder/(name+".png"))
            print(src, json.dumps([round(float(m.mean()), 4) for m in masks]), flush=True)
