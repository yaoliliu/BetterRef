"""Fixed-side identity scores. Keep detection and role-count failures visible."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

NAMES = ["jia_baoyu_1987", "lin_daiyu_1987"]


def extract(path, detector, lv):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Missing image {path}")
    faces = sorted(detector.get(image), key=lambda f: float(f.bbox[0] + f.bbox[2]))
    features = []
    for f in faces:
        crop = cv2.cvtColor(norm_crop(image, f.kps, image_size=112), cv2.COLOR_BGR2RGB)
        array = (crop.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1)[None]
        vec = lv.run(None, {lv.get_inputs()[0].name:array})[0].ravel()
        features.append((f.normed_embedding, vec / np.linalg.norm(vec)))
    return features


def main():
    parser = argparse.ArgumentParser()
    for key in ("models", "refs", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    kwargs = dict(providers=["CPUExecutionProvider"], sess_options=opts)
    det = FaceAnalysis(name="buffalo_l", root=str(args.models / "insightface"),
                       allowed_modules=["detection","recognition"], **kwargs)
    det.prepare(ctx_id=-1, det_size=(640,640))
    lv = ort.InferenceSession(str(args.models / "LVFace-T_Glint360K.onnx"), **kwargs)
    refs = [extract(args.refs/(name+".jpg"), det, lv) for name in NAMES]
    if [len(fs) for fs in refs] != [1, 1]:
        raise ValueError("References must each contain one detected face")
    results = {}
    for seed in args.seeds:
        result = {}
        for arm in ("baseline", "betterref"):
            faces = extract(args.out / str(seed) / (arm+".png"), det, lv)
            row = {"face_count":len(faces)}
            if len(faces) == 2:
                for j, metric in enumerate(("ArcFace","LVFace-T")):
                    row[metric] = [[float(f[j] @ ref[0][j]) for ref in refs] for f in faces]
            result[arm] = row
        results[seed] = result
    (args.out / "scores.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
