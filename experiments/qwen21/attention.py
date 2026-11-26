"""Qwen-Image 2.1 full-body oracle experiment, with native prefix KV caching."""
import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor, _qwenimage21_prepare_qkv, dispatch_attention_fn,
)


class RoleAttention(QwenImage21AttnProcessor):
    def __init__(self, reference_masks, target_masks, positive=2.0, negative=4.0):
        self.reference = reference_masks
        self.target = target_masks
        self.positive = positive
        self.negative = negative
        self.groups = None

    def prepare(self, segments, nkeys, device, dtype):
        refs = [(a, b) for a, b, is_text in segments if not is_text]
        if len(refs) != len(self.reference) or len(refs) != len(self.target):
            raise ValueError("Reference segment count does not match role masks")
        prefix = segments[-1][1]
        masks = torch.stack([m.to(device).flatten().bool() for m in self.target])
        if masks.shape[1] != nkeys - prefix:
            raise ValueError("Target mask does not match the actual target grid")
        unique = masks.sum(0) == 1
        valid = masks & unique
        bias_by_role = []
        for i, mask in enumerate(valid):
            bias = torch.zeros(nkeys, device=device, dtype=dtype)
            for j, ((a, b), ref) in enumerate(zip(refs, self.reference)):
                if ref.numel() != b - a:
                    raise ValueError("Reference mask does not match the actual reference grid")
                bias[a:b].masked_fill_(ref.to(device).flatten().bool(), self.positive if i == j else -self.negative)
            bias_by_role.append((torch.where(mask)[0], bias[None, None, None, :]))
        self.groups = [(torch.where(~unique)[0], None), *bias_by_role]
        self.key_count = nkeys

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        args = (attn, hidden_states, attention_mask, rotary_emb, layer_cache,
                kv_cache_mode, cache_write_slice, segments, key_valid)
        if self.positive == self.negative == 0:
            return super().__call__(*args)
        if kv_cache_mode == "extract":
            # Delegate native causal prefix attention and KV storage. Only target outputs are replaced.
            original = super().__call__(*args)
            q, k, v, _ = _qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb, None, None, None)
            prefix = segments[-1][1]
            self.prepare(segments, k.shape[1], q.device, q.dtype)
            q = q[:, prefix:]
            allowed = key_valid[:, None, None, :] if key_valid is not None else None
        elif kv_cache_mode == "cached" and self.groups is not None:
            original = None
            q, k, v, _ = _qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, None)
            allowed = attention_mask
        else:
            raise ValueError("Experiment requires the native extract → cached sampling path")
        if k.shape[1] != self.key_count or sum(len(rows) for rows, _ in self.groups) != q.shape[1]:
            raise ValueError("Cached attention layout changed")
        output = torch.empty_like(q)
        for rows, bias in self.groups:
            if not len(rows):
                continue
            mask = allowed if bias is None else bias
            if bias is not None and allowed is not None:
                mask = bias.masked_fill(~allowed, float("-inf"))
            chunk = dispatch_attention_fn(q.index_select(1, rows), k, v, attn_mask=mask, dropout_p=0.0)
            output.index_copy_(1, rows, chunk)
        target = attn.to_out[1](attn.to_out[0](output.flatten(2)))
        return target if original is None else torch.cat([original[:, :prefix], target], dim=1)
