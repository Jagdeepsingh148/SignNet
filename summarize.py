"""
summarize.py - collect runs/*/metrics.json into one comparison table for the paper.
Writes runs/results_table.md and runs/results_table.csv.
Run after training: python summarize.py
"""
import csv, json
from pathlib import Path

rows = []
for m in sorted(Path("runs").glob("*/metrics.json")):
    d = json.loads(m.read_text())
    # only report external folders that still exist on disk
    ext = {k: v for k, v in d.get("external", {}).items() if v and Path(k).is_dir()}
    tta = {k: v for k, v in d.get("external_tta", {}).items() if v and Path(k).is_dir()}
    rows.append({
        "Run": d["run"],
        "Input": ("MediaPipe hand crop" if d["input"] == "crop" else "Full image")
                 + (" + robust aug" if d.get("robust_aug") else ""),
        "Val acc (%)": f"{100 * d['best_val_acc']:.2f}",
        "Test acc (%)": f"{100 * d['test_acc']:.2f}",
        "Macro-F1": f"{d['test_macro_f1']:.4f}",
        **{f"{k} (%)": f"{100 * v['accuracy']:.1f} ({v['correct']}/{v['n']})" for k, v in ext.items()},
        **{f"{k} + mirror TTA (%)": f"{100 * v['accuracy']:.1f} ({v['correct']}/{v['n']})" for k, v in tta.items()},
        "Params (M)": f"{d['params_M']:.2f}",
        "Latency (ms)": f"{d['latency_ms_bs1']:.1f}",
        "Train (min)": f"{d['train_minutes']:.1f}",
    })

if not rows:
    raise SystemExit("no runs/*/metrics.json found - train first")

cols = list(dict.fromkeys(k for r in rows for k in r))
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(r.get(c, "-") for c in cols) + " |" for r in rows]
d = json.loads(next(Path("runs").glob("*/metrics.json")).read_text())
md += ["", f"Dataset: {d['dataset']} (split train/val/test = "
       f"{d['split']['train']}/{d['split']['val']}/{d['split']['test']}, stratified, seed {d['hyperparams']['seed']}).",
       f"Model: {d['base_model']} fine-tuned; latency = batch-1 forward pass on {d['device']}."]
Path("runs/results_table.md").write_text("\n".join(md) + "\n")
with open("runs/results_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print("\n".join(md))
