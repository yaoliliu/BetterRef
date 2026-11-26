"""Run a matched baseline / oracle-mask pilot on the fixed two-role scene."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
from attention import RoleAttention

WIDTH, HEIGHT, RESOLUTION = 1152, 768, 768
REFS = ["jia_baoyu_1987", "lin_daiyu_1987"]
PROMPT = ("以图一为贾宝玉的长相参考，以图二为林黛玉的长相参考。大观园的花窗前，身穿绛红长袍的贾宝玉在画面左侧，"
          "身着淡紫衣裙的林黛玉在右侧。两人肩并肩坐在同一张长凳上，肩膀和脸庞都正对镜头，目光直视前方，双眼清晰可见，神情自然。"
          "两人共同托着一本摊开的线装书，书停在胸口下方。平视机位，双人半身合影，两张脸大小相近。1987 年古典电视剧剧照，柔和自然光。")


def grid_mask(path, size):
    with Image.open(path) as im:
        foreground = np.asarray(im.convert("L").resize(size, Image.Resampling.BOX)) >= 128
    return torch.from_numpy(foreground)


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "refs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=["baseline", "betterref"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    images = [Image.open(args.refs / (name + ".jpg")).convert("RGB") for name in REFS]
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    base = {"width": WIDTH, "height": HEIGHT, "output_resolution": RESOLUTION, "true_cfg_scale": 1.0,
            "num_inference_steps": 40, "use_kv_cache": True}
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "sampling.json").write_text(json.dumps(dict(**base, prompt=PROMPT, refs=REFS), ensure_ascii=False, indent=2))
    sizes = [calculate_dimensions(RESOLUTION**2, im.width / im.height)[:2] for im in images]
    with torch.inference_mode():
        for seed in args.seeds:
            folder = args.out / str(seed)
            folder.mkdir(exist_ok=True)
            if args.arm == "betterref":
                ref = [grid_mask(args.out / (n+".png"), (w//16, h//16)) for n, (w,h) in zip(REFS,sizes)]
                target = [grid_mask(folder / (n+".png"), (WIDTH//16,HEIGHT//16)) for n in REFS]
            for block in pipe.transformer.transformer_blocks:
                processor = QwenImage21AttnProcessor() if args.arm == "baseline" else RoleAttention(ref, target)
                block.attn.set_processor(processor)
            print("start", args.arm, seed, flush=True)
            result = pipe(prompt=PROMPT, image=images, generator=torch.Generator("cuda").manual_seed(seed), **base)
            result.images[0].convert("RGB").save(folder / (args.arm+".png"))
            print("saved", args.arm, seed, flush=True)


if __name__ == "__main__":
    main()
