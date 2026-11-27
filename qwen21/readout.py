"""Switch from native Qwen attention to role-biased attention after the readout."""
import json
from pathlib import Path

from capture import Capture
from predictor import predict
from attention import RoleAttention

WEIGHTS = Path(__file__).resolve().parents[1] / "readouts" / "qwen21-readout-v1.json"


class LearnedReadout:
    def __init__(self, pipe, ref_shapes, target_shape, select_masks, positive, negative):
        self.weights = json.loads(WEIGHTS.read_text())
        self.ref_shapes = ref_shapes
        self.target_shape = target_shape
        self.select_masks = select_masks
        self.positive = positive
        self.negative = negative
        self.blocks = pipe.transformer.transformer_blocks
        height, width = target_shape
        self.state = {
            "step": 0, "steps": self.weights["steps"], "features": {}, "grids": ref_shapes,
            "target_tokens": height * width,
        }

    @property
    def last_step(self):
        return max(self.weights["steps"])

    def install(self):
        for index in self.weights["blocks"]:
            self.blocks[index].attn.set_processor(Capture(self.state, index))

    def advance(self, step, latents):
        self.state["step"] = step + 1
        if step != self.last_step:
            return
        refs, targets = predict(self.state["features"], self.weights, self.target_shape, self.ref_shapes)
        ref_masks, target_masks = self.select_masks(refs, targets)
        layout = self.state["layout"]
        processor = RoleAttention(ref_masks, target_masks, self.positive, self.negative)
        processor.prepare(layout, layout[-1][1] + latents.shape[1], latents.device, latents.dtype)
        for block in self.blocks:
            block.attn.set_processor(processor)
        self.state["features"].clear()
