"""
reeval_external.py - re-score every trained run on the external test folder(s) under identical
conditions (current MediaPipe detector for crop runs; plain and mirror-TTA), and update each
runs/<name>/metrics.json so summarize.py compares like with like.
Run: python reeval_external.py
"""
import json
from pathlib import Path

import torch
from torchvision import transforms as T
from transformers import AutoModelForImageClassification

from trainV2 import EXTERNAL_TESTS, DEVICE, IMG_SIZE, MEAN, STD, external_test
from eval_v2 import HandCropper

tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
cropper = HandCropper("hand_landmarker.task")
folders = [f for f in EXTERNAL_TESTS if Path(f).is_dir()]

for mfile in sorted(Path("runs").glob("*/metrics.json")):
    d = json.loads(mfile.read_text())
    model = AutoModelForImageClassification.from_pretrained(mfile.parent / "model").to(DEVICE).eval()
    label2id = {v.upper(): int(k) for k, v in model.config.id2label.items()}
    crop = cropper if d["input"] == "crop" else None
    with torch.no_grad():
        d["external"] = {f: external_test(model, f, tf, label2id, crop) for f in folders}
        d["external_tta"] = {f: external_test(model, f, tf, label2id, crop, tta=True) for f in folders}
    mfile.write_text(json.dumps(d, indent=2))
    for f in folders:
        print(f"{d['run']:20s} {f}: plain {d['external'][f]['accuracy']:.3f} "
              f"| mirror TTA {d['external_tta'][f]['accuracy']:.3f}")
