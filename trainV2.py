"""
trainV2.py - EfficientNet-B0 ASL fine-tune using the model-card hyperparameters.
Base: google/efficientnet-b0 (HF, ImageNet pretrained)
LR 2e-4 | batch 16 | 5 epochs | AdamW | cosine | wd 1e-4 | warmup 5%
Windows-safe (top-level collate class + main guard).

Two input modes (ablation for the paper):
  --input full  : original dataset images
  --input crop  : MediaPipe hand crops from cache_crops/<row index>.jpg (matches Live_asl.py)
  --robust      : domain-robust augmentation (flip p=0.5 for left/right hands, stronger lighting,
                  blur, grayscale, looser crops) - targets the external-test failure modes

Every run writes to runs/<name>/:
  model/                 fine-tuned weights + image processor (loadable by Live_asl.py / eval_v2.py)
  history.csv            per-epoch train/val loss + accuracy
  curves.png             learning curves
  confusion_matrix.png   normalised confusion matrix on the held-out test split
  test_report.txt        per-class precision / recall / F1
  metrics.json           headline numbers (accuracy, macro-F1, params, latency, train time, external tests)

pip install torch torchvision transformers datasets scikit-learn matplotlib
Run: python trainV2.py --input full
     python trainV2.py --input crop
"""
import argparse, csv, json, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
from datasets import load_dataset
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          get_cosine_schedule_with_warmup)
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATASET = "Marxulia/asl_sign_languages_alphabets_v03"
BASE_MODEL = "google/efficientnet-b0"
CROP_DIR = Path("cache_crops")
EXTERNAL_TESTS = ["asl_alphabet_test", "asl_alphabet_test_1"]
IMG_SIZE = 224
BATCH = 16
EPOCHS = 5
LR = 2e-4
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05
SEED = 42
NUM_WORKERS = 4          # set to 0 if you get multiprocessing errors
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class Collate:
    def __init__(self, tf, img_col, lbl_col, crop_dir=None):
        self.tf, self.img_col, self.lbl_col, self.crop_dir = tf, img_col, lbl_col, crop_dir

    def load(self, b):
        if self.crop_dir is not None:
            return Image.open(self.crop_dir / f"{b['idx']}.jpg").convert("RGB")
        return b[self.img_col].convert("RGB")

    def __call__(self, batch):
        x = torch.stack([self.tf(self.load(b)) for b in batch])
        y = torch.tensor([b[self.lbl_col] for b in batch])
        return x, y


@torch.no_grad()
def evaluate(model, dl, loss_fn):
    model.eval()
    preds, ys, total_loss = [], [], 0.0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(pixel_values=x).logits
        total_loss += loss_fn(logits, y).item() * len(y)
        preds.append(logits.argmax(1).cpu())
        ys.append(y.cpu())
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return (preds == ys).mean(), total_loss / len(ys), preds, ys


@torch.no_grad()
def latency_ms(model, n=100):
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for _ in range(10):
        model(pixel_values=x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(pixel_values=x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


@torch.no_grad()
def external_test(model, folder, tf, label2id, cropper, tta=False):
    """Accuracy on a flat folder of <LETTER>_test.jpg images (Kaggle ASL alphabet test style).
    tta=True averages softmax of the image and its mirror (hand-dominance invariance)."""
    from eval_v2 import collect
    items = [(p, label2id[l.upper()]) for p, l in collect(Path(folder)) if l.upper() in label2id]
    if not items:
        return None
    imgs = []
    for p, _ in items:
        im = Image.open(p).convert("RGB")
        imgs.append(tf(cropper(im) if cropper else im))
    x = torch.stack(imgs).to(DEVICE)
    prob = model(pixel_values=x).logits.softmax(1)
    if tta:
        prob = (prob + model(pixel_values=torch.flip(x, dims=[3])).logits.softmax(1)) / 2
    pred = prob.argmax(1).cpu().numpy()
    ys = np.array([y for _, y in items])
    return {"n": len(ys), "correct": int((pred == ys).sum()), "accuracy": float((pred == ys).mean())}


def plot_curves(hist, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [h["train_loss"] for h in hist], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], "s-", label="validation")
    ax[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    ax[1].plot(ep, [h["train_acc"] for h in hist], "o-", label="train")
    ax[1].plot(ep, [h["val_acc"] for h in hist], "s-", label="validation")
    ax[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for a in ax:
        a.grid(alpha=.3); a.legend(); a.set_xticks(ep)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_confusion(ys, preds, names, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(ys, preds, labels=range(len(names)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="Predicted", ylabel="True", title="Normalised confusion matrix (test split)")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}".lstrip("0") if cm[i, j] < 1 else "1",
                        ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["full", "crop"], default="crop")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--robust", action="store_true", help="domain-robust augmentation")
    ap.add_argument("--name", default=None, help="run folder name (default: effb0_<input>)")
    args = ap.parse_args()

    seed_everything(SEED)
    out = Path("runs") / (args.name or f"effb0_{args.input}" + ("_robust" if args.robust else ""))
    out.mkdir(parents=True, exist_ok=True)
    crop_dir = CROP_DIR if args.input == "crop" else None
    print("device:", DEVICE, "| input:", args.input, "| out:", out)

    ds = load_dataset(DATASET, split="train")
    img_col = next(c for c in ds.column_names if "image" in c.lower())
    lbl_col = next(c for c in ds.column_names if "label" in c.lower())
    class_names = ds.features[lbl_col].names
    ds = ds.add_column("idx", list(range(len(ds))))   # row index -> cache_crops/<idx>.jpg
    print("columns:", ds.column_names, "| classes:", len(class_names))

    # identical split for both input modes (same seed, same stratification)
    tmp = ds.train_test_split(test_size=0.2, seed=SEED, stratify_by_column=lbl_col)
    vt = tmp["test"].train_test_split(test_size=0.5, seed=SEED, stratify_by_column=lbl_col)
    train_ds, val_ds, test_ds = tmp["train"], vt["train"], vt["test"]
    print("split sizes:", len(train_ds), len(val_ds), len(test_ds))

    if args.robust:
        # flip 0.5: signers use either hand; lighting/blur/grayscale: webcam conditions;
        # scale 0.6: tolerate imperfect or missing hand crops
        train_tf = T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])
    else:
        # augmentation exactly as in the model card
        train_tf = T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])
    eval_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

    kw = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
              pin_memory=DEVICE == "cuda")
    train_dl = DataLoader(train_ds, BATCH, shuffle=True,
                          collate_fn=Collate(train_tf, img_col, lbl_col, crop_dir), **kw)
    val_dl = DataLoader(val_ds, BATCH, collate_fn=Collate(eval_tf, img_col, lbl_col, crop_dir), **kw)
    test_dl = DataLoader(test_ds, BATCH, collate_fn=Collate(eval_tf, img_col, lbl_col, crop_dir), **kw)

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=len(class_names), ignore_mismatched_sizes=True,
        id2label={i: n for i, n in enumerate(class_names)},
        label2id={n: i for i, n in enumerate(class_names)},
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = args.epochs * len(train_dl)
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP_RATIO * total_steps), total_steps)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model_dir = out / "model"
    best, hist, t0 = 0.0, [], time.time()
    for ep in range(args.epochs):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(pixel_values=x).logits
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y); tc += (logits.argmax(1) == y).sum().item(); tn += len(y)
        acc, vloss, _, _ = evaluate(model, val_dl, loss_fn)
        hist.append(dict(epoch=ep + 1, train_loss=tl / tn, train_acc=tc / tn,
                         val_loss=vloss, val_acc=float(acc)))
        print(f"epoch {ep+1}/{args.epochs}  train_loss={tl/tn:.4f} train_acc={tc/tn:.4f}  "
              f"val_acc={acc:.4f}  val_loss={vloss:.4f}", flush=True)
        if acc > best:
            best = acc
            model.save_pretrained(model_dir)
    train_min = (time.time() - t0) / 60

    # image processor saved alongside so Live_asl.py can load the folder directly
    proc = AutoImageProcessor.from_pretrained(BASE_MODEL)
    proc.size = {"height": IMG_SIZE, "width": IMG_SIZE}
    proc.do_center_crop = False
    proc.image_mean, proc.image_std = list(MEAN), list(STD)
    proc.save_pretrained(model_dir)

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    plot_curves(hist, out / "curves.png")

    model = AutoModelForImageClassification.from_pretrained(model_dir).to(DEVICE)
    acc, tloss, preds, ys = evaluate(model, test_dl, loss_fn)
    report = classification_report(ys, preds, target_names=class_names, digits=4)
    print(f"\nTEST accuracy: {acc:.4f}\n{report}")
    (out / "test_report.txt").write_text(f"TEST accuracy: {acc:.4f}\n\n{report}")
    plot_confusion(ys, preds, class_names, out / "confusion_matrix.png")

    # external generalisation (different signers / backgrounds); crop model gets MediaPipe crops
    cropper = None
    if args.input == "crop":
        from eval_v2 import HandCropper
        cropper = HandCropper("hand_landmarker.task")
    label2id = {n.upper(): i for i, n in enumerate(class_names)}
    external = {f: external_test(model, f, eval_tf, label2id, cropper)
                for f in EXTERNAL_TESTS if Path(f).is_dir()}
    external_tta = {f: external_test(model, f, eval_tf, label2id, cropper, tta=True)
                    for f in EXTERNAL_TESTS if Path(f).is_dir()}
    for f in external:
        if external[f]:
            print(f"external {f}: {external[f]['correct']}/{external[f]['n']} = "
                  f"{external[f]['accuracy']:.4f} | with mirror TTA: {external_tta[f]['accuracy']:.4f}")

    metrics = {
        "run": out.name, "input": args.input, "robust_aug": args.robust, "base_model": BASE_MODEL, "dataset": DATASET,
        "split": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "hyperparams": dict(img_size=IMG_SIZE, batch=BATCH, epochs=args.epochs, lr=LR,
                            weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
                            optimizer="AdamW", schedule="cosine", seed=SEED, amp=use_amp),
        "best_val_acc": float(best),
        "test_acc": float(acc), "test_loss": float(tloss),
        "test_macro_f1": float(f1_score(ys, preds, average="macro")),
        "test_weighted_f1": float(f1_score(ys, preds, average="weighted")),
        "params_M": n_params / 1e6,
        "latency_ms_bs1": latency_ms(model),
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "train_minutes": train_min,
        "external": external,
        "external_tta": external_tta,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nall outputs in {out}/")


if __name__ == "__main__":
    main()
