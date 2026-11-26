# Qwen-Image 2.1 full-body reference pilot

This experiment measures the effect of changing attention when each role already has an instance mask. Reference masks come from SAM3 on the two input photos. For each seed, SAM3 segments the unmodified Qwen output, then the two target masks are bound by the left/right positions specified in the prompt. Overlapping target patches receive the original attention. All 32 layers apply +2 to the same role’s reference foreground and −4 to the other role’s reference foreground. The prefix KV cache remains enabled.

The pilot uses Qwen/Qwen-Image-2.1 revision `d26bb61231c349cf6b7896fa83353113880e1ba3` and Diffusers revision `d42ebf9c6e0b46103b74283f5324357a67acaa04`. Inference used BF16, 40 steps, CFG 1, a 1152 × 768 target and reference area parameter 768. Both arms use the same two photos, prompt, seed, resolution, and cache settings. The seeds are 42, 43 and 44.

| Role | ArcFace baseline → enhanced | LVFace-T baseline → enhanced |
| --- | --- | --- |
| Jia Baoyu, left | 0.843 → 0.852 | 0.836 → 0.863 |
| Lin Daiyu, right | 0.767 → 0.873 | 0.780 → 0.866 |

Values are mean cosine similarities to the corresponding photo across the three seeds. Lin Daiyu gained on both metrics for seeds 42 and 44. At seed 43 her ArcFace gain was 0.003 and her LVFace-T score fell by 0.014. Both versions of every seed produced two detected faces. The readout weights bundled for H3 are unused in this experiment.

Run from the repository root with local model directories. `METRICS` contains the InsightFace `insightface/models/buffalo_l` directory and `LVFace-T_Glint360K.onnx`. The evaluation script accepts a separate Python environment with InsightFace, OpenCV and ONNX Runtime.

```bash
PILOT=experiments/qwen21
REFS=workflows/h3_ref_examples
OUT=results/qwen21
python "$PILOT/check_attention.py"
python "$PILOT/pilot.py" --arm baseline --model "$QWEN" --refs "$REFS" --out "$OUT"
python "$PILOT/oracle.py" --model "$SAM3" --refs "$REFS" --out "$OUT"
python "$PILOT/pilot.py" --arm betterref --model "$QWEN" --refs "$REFS" --out "$OUT"
python "$PILOT/evaluate.py" --models "$METRICS" --refs "$REFS" --out "$OUT"
```

The score matrices store detected faces in left-to-right order as rows and the two references in input order as columns. The comparison uses the diagonal so that swapping the characters counts against the method.
