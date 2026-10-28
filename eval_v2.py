"""
eval_v2.py - evaluate the fine-tuned model (runs/effb0_crop/model) on an external image folder.

Supports two folder layouts automatically:
  1) subfolders per class:   asl_alphabet_test/A/xxx.jpg, asl_alphabet_test/B/...
  2) flat files named by class: asl_alphabet_test/A_test.jpg, B_test.jpg ...  (label = text before first "_" or first character)

Usage:
  python eval_v2.py                                   # default folder + model
  python eval_v2.py --data asl_alphabet_test --crop   # crop to the hand with MediaPipe first
  python eval_v2.py --model runs/effb0_full/model --data some_other_folder

pip install torch torchvision transformers scikit-learn pillow
(--crop also needs: pip install mediapipe, and hand_landmarker.task in this folder)
"""
import argparse, csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModelForImageClassification
from sklearn.metrics import classification_report, accuracy_score

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect(data_dir: Path):
    """Return list of (path, label_string)."""
    items = []
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if subdirs:  # layout 1
        for d in sorted(subdirs):
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in EXTS:
                    items.append((f, d.name))
    else:        # layout 2
        for f in sorted(data_dir.iterdir()):
            if f.suffix.lower() in EXTS:
                stem = f.stem
                label = stem.split("_")[0] if "_" in stem else stem[0]
                items.append((f, label))
    return items


class HandCropper:
    def __init__(self, task_path="hand_landmarker.task", margin=0.2):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            num_hands=1, running_mode=vision.RunningMode.IMAGE,
            min_hand_detection_confidence=0.3)   # webcam photos are often dim / low contrast
        self.det = vision.HandLandmarker.create_from_options(opts)
        self.margin = margin
        self.found = 0
        self.total = 0

    def __call__(self, img: Image.Image) -> Image.Image:
        self.total += 1
        arr = np.array(img.convert("RGB"))
        res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=arr))
        if not res.hand_landmarks:   # retry on a contrast-stretched copy (dark images)
            from PIL import ImageOps
            boosted = np.array(ImageOps.autocontrast(img.convert("RGB"), cutoff=2))
            res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=boosted))
        if not res.hand_landmarks:
            return img
        self.found += 1
        h, w = arr.shape[:2]
        xs = [p.x for p in res.hand_landmarks[0]]
        ys = [p.y for p in res.hand_landmarks[0]]
        x0, x1, y0, y1 = min(xs) * w, max(xs) * w, min(ys) * h, max(ys) * h
        side = max(x1 - x0, y1 - y0) * (1 + 2 * self.margin)   # square crop with margin
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)),
               int(min(cx + side / 2, w)), int(min(cy + side / 2, h)))
        return img.crop(box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/effb0_crop/model")
    ap.add_argument("--data", default="asl_alphabet_test")
    ap.add_argument("--crop", action="store_true", help="crop to hand with MediaPipe first")
    ap.add_argument("--task", default="hand_landmarker.task")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta", action="store_true", help="average predictions with the mirrored image")
    args = ap.parse_args()

    model = AutoModelForImageClassification.from_pretrained(args.model).to(DEVICE).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v.upper(): k for k, v in id2label.items()}
    class_names = [id2label[i] for i in range(len(id2label))]
    print("device:", DEVICE, "| model classes:", len(class_names))

    items = collect(Path(args.data))
    kept, skipped = [], Counter()
    for p, lab in items:
        if lab.upper() in label2id:
            kept.append((p, label2id[lab.upper()]))
        else:
            skipped[lab] += 1
    print(f"images found: {len(items)} | evaluated: {len(kept)}")
    if skipped:
        print("skipped (label not in model, e.g. del/nothing/space):", dict(skipped))
    if not kept:
        print("Nothing to evaluate - check the folder layout.")
        return

    cropper = HandCropper(args.task) if args.crop else None
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

    preds, ys, confs, paths = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(kept), args.batch):
            chunk = kept[i:i + args.batch]
            imgs = []
            for p, _ in chunk:
                im = Image.open(p).convert("RGB")
                if cropper:
                    im = cropper(im)
                imgs.append(tf(im))
            x = torch.stack(imgs).to(DEVICE)
            prob = torch.softmax(model(pixel_values=x).logits, dim=1)
            if args.tta:
                prob = (prob + torch.softmax(model(pixel_values=torch.flip(x, dims=[3])).logits, dim=1)) / 2
            prob = prob.cpu()
            conf, pred = prob.max(1)
            preds += pred.tolist(); confs += conf.tolist()
            ys += [y for _, y in chunk]; paths += [str(p) for p, _ in chunk]

    preds, ys, confs = np.array(preds), np.array(ys), np.array(confs)
    acc = accuracy_score(ys, preds)
    print(f"\nACCURACY on {args.data}: {acc:.4f}  ({int((preds == ys).sum())}/{len(ys)})")
    print(f"mean confidence: {confs.mean():.3f} | when correct: "
          f"{confs[preds == ys].mean() if (preds == ys).any() else float('nan'):.3f} | when wrong: "
          f"{confs[preds != ys].mean() if (preds != ys).any() else float('nan'):.3f}")
    if cropper:
        print(f"hand detected in {cropper.found}/{cropper.total} images (rest used uncropped)")

    present = sorted(set(ys.tolist()) | set(preds.tolist()))
    print(classification_report(ys, preds, labels=present,
                                target_names=[class_names[i] for i in present],
                                digits=4, zero_division=0))

    pairs = Counter((class_names[t], class_names[p]) for t, p in zip(ys, preds) if t != p)
    if pairs:
        print("Most confused (true -> predicted):")
        for (t, p), n in pairs.most_common(10):
            print(f"  {t} -> {p}: {n}")

    out = Path("eval_v2_errors.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "predicted", "confidence"])
        for pa, t, p, c in zip(paths, ys, preds, confs):
            if t != p:
                w.writerow([pa, class_names[t], class_names[p], f"{c:.3f}"])
    print(f"\nmisclassified images listed in {out}")


if __name__ == "__main__":
    main()