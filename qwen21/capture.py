"""Capture early Qwen 2.1 attention features while delegating sampling to the native processor."""
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

BLOCKS = (7, 15, 23, 31)
STEPS = (0, 2, 4)


def pool_regions(tokens, shape):
    height, width = shape
    grid = tokens.reshape(height, width, -1).permute(2, 0, 1)
    pooled = F.adaptive_avg_pool2d(grid.float(), (4, 4))
    return pooled.flatten(1).T.reshape(16, *tokens.shape[1:]).contiguous()


def semantic(tokens):
    # Uniform channel sample of normalized pre-attention activations.
    z = F.layer_norm(tokens.float(), (tokens.shape[-1],))
    return z[:, ::max(1, z.shape[-1] // 64)][:, :64]


class Capture(QwenImage21AttnProcessor):
    def __init__(self, state, block):
        self.state = state
        self.block = block

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        step = self.state["step"]
        if step in self.state.get("steps", STEPS):
            self.capture(attn, hidden_states[0], segments, step)
        return super().__call__(attn, hidden_states, attention_mask, rotary_emb, layer_cache,
                                kv_cache_mode, cache_write_slice, segments, key_valid)

    def capture(self, attn, tokens, segments, step):
        state = self.state
        data = state["features"]
        if step == 0:
            state["layout"] = segments
            refs = [(a, b) for a, b, text in segments if not text]
            assert len(refs) == len(state["grids"]) == 2
            self.anchors = []
            for role, ((a, b), shape) in enumerate(zip(refs, state["grids"])):
                ref = tokens[a:b]
                raw = semantic(ref)
                pooled = pool_regions(raw, shape)
                selfsim = F.normalize(raw) @ F.normalize(pooled).T
                data[f"r{role}_{self.block}"] = torch.cat([raw, selfsim], -1).half().cpu().numpy()
                keys = attn.norm_k(attn.to_k(ref).unflatten(-1, (attn.heads, -1)))
                self.anchors.append((pool_regions(keys, shape), pooled))
            tokens = tokens[segments[-1][1]:]
        assert len(tokens) == state["target_tokens"]
        raw = semantic(tokens)
        queries = attn.norm_q(attn.to_q(tokens).unflatten(-1, (attn.heads, -1))).float()
        affinities = []
        for keys, desc in self.anchors:
            # Preserve per-head responses to the portrait centre and strongest spatial bin.
            logits = torch.einsum("nhd,khd->nhk", queries, keys) / queries.shape[-1] ** .5
            centre = logits[:, :, [5, 6, 9, 10]].mean(-1)
            sim = F.normalize(raw) @ F.normalize(desc).T
            affinities.append(torch.cat((centre, logits.amax(-1), sim), -1))
        for role, own in enumerate(affinities):
            values = torch.cat((raw, own, own - affinities[1-role]), -1)
            data[f"t{role}_{step}_{self.block}"] = values.half().cpu().numpy()
