"""NumPy inference for the shared foreground and reference-conditioned role logits."""
import numpy as np
from scipy.ndimage import gaussian_filter


def matrices(features, config, role, kind):
    keys = ([f"r{role}_{b}" for b in config["blocks"]] if kind == "ref" else
            [f"t{role}_{s}_{b}" for s in config["steps"] for b in config["blocks"]])
    return np.concatenate([features[k] for k in keys], axis=-1).astype(np.float32)


def scores(features, config, role, kind, shape):
    p = config[kind]
    logits = matrices(features, config, role, kind) @ np.asarray(p["weight"], dtype=np.float32) + p["bias"]
    return gaussian_filter(logits.reshape(shape), sigma=0.65, mode="nearest")


def predict(features, weights, target_shape, ref_shapes):
    ref = [scores(features, weights, i, "ref", shape) > 0 for i, shape in enumerate(ref_shapes)]
    logits = [scores(features, weights, i, "target", target_shape) for i in range(2)]
    assignment = np.argmax([np.zeros_like(logits[0]), *logits], axis=0)
    return ref, np.stack([assignment == 1, assignment == 2])
