"""Assign unambiguous SAM3 person instances in horizontal order."""
import numpy as np


def person_masks(image, count, model, processor):
    for query in ("person", "character"):
        inputs = processor(images=image, text=query, return_tensors="pt")
        output = processor.post_process_instance_segmentation(
            model(**inputs), threshold=0.5, mask_threshold=0.5, target_sizes=[image.size[::-1]])
        masks = [m.cpu().numpy().astype(bool) for m in output[0]["masks"] if m.float().mean() >= 0.02]
        masks.sort(key=np.sum, reverse=True)
        if count == 1 and masks and (len(masks) == 1 or masks[0].sum() > 2 * masks[1].sum()):
            return masks[:1]
        if count == 2 and len(masks) == 2 and (masks[0] & masks[1]).sum() < .25 * min(m.sum() for m in masks):
            return sorted(masks, key=lambda m: np.where(m)[1].mean())
    raise ValueError("SAM3 could not isolate the requested people. Paint or connect masks for this image.")
