"""Predict per-subject patch occupancy from standardized early attention maps."""
import torch
import torch.nn.functional as F


def features(captures, name, reference):
    maps = []
    for key in sorted(captures):
        image = captures[key]["refs"][name] if reference else captures[key]["target"]
        logits = image["logits"]
        others = [k for k in logits if k not in (name, "__background__")]
        channels = [logits[name], logits["__background__"]]
        if not reference:
            channels.append(torch.stack([logits[k] for k in others]).amax(0) if others else torch.zeros_like(logits[name]))
        grid = image["grid"]
        maps.append(torch.stack(channels).float().reshape(-1, grid[1], grid[2]))
    raw = torch.cat(maps)
    raw = (raw - raw.mean((1,2), keepdim=True)) / raw.std((1,2), keepdim=True, unbiased=False).clamp_min(1e-6)
    smooth = F.avg_pool2d(F.pad(raw.unsqueeze(0), (1,1,1,1), mode="replicate"), 3, stride=1)[0]
    return torch.cat((raw, smooth)).permute(1,2,0)


def apply_readout(captures, weights, active_subjects=None):
    required = [tuple(k) for k in weights["capture_keys"]]
    selected = {k: captures[k] for k in required}
    names = selected[required[0]]["ref_names"]
    def score(name, reference):
        config = weights["refs" if reference else "target"]
        x = features(selected, name, reference)
        return x @ x.new_tensor(config["coef"]) + config["intercept"]
    active = names if active_subjects is None else tuple(active_subjects)
    zero = torch.zeros(selected[required[0]]["target"]["grid"])
    scores = [zero[0]] + [score(name, False) for name in active]
    owner = torch.stack(scores).argmax(0)
    target_masks = {name: (owner == idx).float()[None] for idx, name in enumerate(["__background__", *active])}
    target_masks.update({name: zero for name in names if name not in active})
    refs = {}
    for name in names:
        mask = ((score(name, True) > 0).float()[None] if name in active else
                torch.zeros(selected[required[0]]["refs"][name]["grid"]))
        refs[name] = {name:mask, "__background__":1-mask}
    return {"target":target_masks, "refs":refs}
