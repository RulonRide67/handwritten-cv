import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

from common import (CLASSES, DATA_DIR, MODELS_DIR, RESULTS_DIR, SYNTH_DIR, eval_transform, load_image,
                    load_split, load_trained)
from gradcam import GradCAM, overlay

FIG = RESULTS_DIR / "figures"
CONFIGS = [("svm_hog", "HOG + SVM (A1)", "#2a78d6"),
           ("real", "ResNet18, real only (A1)", "#eb6834"),
           ("all", "ResNet18 + synthetic, no filter", "#1baf7a"),
           ("filtered", "ResNet18 + synthetic, filtered (A2)", "#eda100")]
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
TEXT = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.dpi": 160, "savefig.bbox": "tight"})


def load_runs():
    runs = {}
    for f in sorted((RESULTS_DIR / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        key = "svm_hog" if r["name"] == "svm_hog" else r["synthetic"].replace("none", "real")
        runs.setdefault(key, []).append(r)
    return runs


def summary_table(runs):
    rows = []
    for key, label, _ in CONFIGS:
        if key not in runs:
            continue
        m = pd.DataFrame([r["test"] for r in runs[key]])[["accuracy", "precision", "recall", "f1"]]
        row = {"model": label, "runs": len(m), "train_synthetic_used": runs[key][0].get("train_synthetic", 0)}
        for c in m:
            row[c] = f"{m[c].mean():.3f} ± {m[c].std(ddof=0):.3f}" if len(m) > 1 else f"{m[c].mean():.3f}"
            row[c + "_mean"] = m[c].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    table.drop(columns=[c for c in table if c.endswith("_mean")]).to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    per_class = pd.DataFrame({label: pd.DataFrame([r["test"]["f1_per_class"] for r in runs[key]]).mean()
                              for key, label, _ in CONFIGS if key in runs}).round(3)
    per_class.index.name = "class"
    per_class.to_csv(RESULTS_DIR / "per_class_f1.csv")
    return table


def comparison_chart(table, runs):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    metrics = ["accuracy", "f1"]
    keys = [k for k, _, _ in CONFIGS if k in runs]
    width = 0.8 / len(keys)
    for i, (key, label, color) in enumerate([c for c in CONFIGS if c[0] in runs]):
        vals = [np.mean([r["test"][m] for r in runs[key]]) for m in metrics]
        errs = [np.std([r["test"][m] for r in runs[key]]) for m in metrics]
        x = np.arange(len(metrics)) + (i - (len(keys) - 1) / 2) * width
        ax.bar(x, vals, width * 0.92, yerr=errs, color=color, label=label, capsize=3,
               error_kw={"ecolor": MUTED, "elinewidth": 1})
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(range(len(metrics)), ["Accuracy", "Macro F1"])
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Test set results (mean ± std over seeds)", loc="left", color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.savefig(FIG / "model_comparison.png")
    plt.close(fig)


def training_curves(runs):
    keys = [c for c in CONFIGS[1:] if c[0] in runs]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, label, color in keys:
        h = pd.DataFrame(next((r for r in runs[key] if r["seed"] == 42), runs[key][0])["history"])
        ax[0].plot(h.epoch, h.train_loss, color=color, lw=2, ls="--", label=f"{label} train")
        ax[0].plot(h.epoch, h.val_loss, color=color, lw=2, label=f"{label} val")
        ax[1].plot(h.epoch, h.train_acc, color=color, lw=2, ls="--")
        ax[1].plot(h.epoch, h.val_acc, color=color, lw=2, label=label)
    ax[0].set_title("Loss (dashed = train, solid = val)", loc="left", color=TEXT)
    ax[1].set_title("Accuracy (dashed = train, solid = val)", loc="left", color=TEXT)
    for a in ax:
        a.set_xlabel("epoch")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Training curves, seed 42", x=0.01, y=1.04, ha="left", color=TEXT)
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)


def confusion(runs):
    for key, label, _ in CONFIGS:
        if key not in runs:
            continue
        cm = np.sum([np.array(r["test"]["confusion_matrix"]) for r in runs[key]], axis=0)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.imshow(cm, cmap=BLUES)
        ax.grid(False)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > cm.max() * 0.5 else TEXT)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{label}\n(summed over {len(runs[key])} run(s))", loc="left", fontsize=9, color=TEXT)
        fig.savefig(FIG / f"confusion_{key}.png")
        plt.close(fig)


def dataset_table():
    split = load_split()
    table = pd.DataFrame(index=CLASSES)
    for part in ["train", "val", "test"]:
        table[part] = pd.Series([CLASSES[y] for _, y in split[part]]).value_counts()
    if (SYNTH_DIR / "synthetic.csv").exists():
        s = pd.read_csv(SYNTH_DIR / "synthetic.csv", dtype=str)
        table["synthetic_generated"] = s.label.value_counts()
        if "keep" in s:
            table["synthetic_kept"] = s[s.keep == "1"].label.value_counts()
    table = table.fillna(0).astype(int)
    table.loc["total"] = table.sum()
    table.index.name = "class"
    table.to_csv(RESULTS_DIR / "dataset_summary.csv")
    if "synthetic_kept" in table:
        t = table.loc["total"]
        share = {"real_train": int(t.train), "synthetic_generated": int(t.synthetic_generated),
                 "synthetic_kept": int(t.synthetic_kept),
                 "synthetic_share_of_train_pool": round(t.synthetic_kept / (t.synthetic_kept + t.train), 3),
                 "synthetic_share_per_epoch": 0.5}
        (RESULTS_DIR / "synthetic_share.json").write_text(json.dumps(share, indent=1))
        s = pd.read_csv(SYNTH_DIR / "synthetic.csv", dtype=str)
        s["keep"] = s.keep.astype(int)
        s.groupby("style").keep.agg(["count", "mean"]).round(3).rename(
            columns={"count": "generated", "mean": "kept_ratio"}).to_csv(RESULTS_DIR / "synthetic_by_style.csv")


def synthetic_gallery():
    if not (SYNTH_DIR / "synthetic.csv").exists():
        return
    s = pd.read_csv(SYNTH_DIR / "synthetic.csv", dtype=str)
    picks = [s[s.label == c].source.iloc[0] for c in ["2", "3", "7", "8", "+", "="] if (s.label == c).any()]
    styles = list(dict.fromkeys(s["style"]))
    fig, ax = plt.subplots(len(picks), len(styles) + 1, figsize=(1.5 * (len(styles) + 1), 1.75 * len(picks)))
    ax = np.atleast_2d(ax)
    for r, src in enumerate(picks):
        ax[r, 0].imshow(load_image(DATA_DIR / src))
        ax[r, 0].set_title("REAL", fontsize=8, color=TEXT)
        rows = s[s.source == src].set_index("style")
        for c, style in enumerate(styles, start=1):
            if style in rows.index:
                row = rows.loc[style]
                ax[r, c].imshow(load_image(SYNTH_DIR / row.file))
                mark = "" if "keep" not in row else (" kept" if row.keep == "1" else " rejected")
                ax[r, c].set_title(f"SYNTHETIC\n{style}{mark}", fontsize=6.5, color="#d03b3b")
        for a in ax[r]:
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
    fig.tight_layout()
    fig.suptitle("Real images and TA-TiTok + MaskGen variants (all variants are SYNTHETIC)", fontsize=10, color=TEXT, y=1.02)
    fig.savefig(FIG / "synthetic_examples.png")
    plt.close(fig)


def gradcam_figure():
    names = [n for n in ["resnet18_real", "resnet18_synthetic"] if (MODELS_DIR / f"{n}.pt").exists()]
    if not names:
        return
    models = {n: load_trained(MODELS_DIR / f"{n}.pt") for n in names}
    cams = {n: GradCAM(m) for n, m in models.items()}
    tf = eval_transform()
    test = load_split()["test"]
    preds = {}
    for n, m in models.items():
        with torch.no_grad():
            preds[n] = [int(m(tf(load_image(DATA_DIR / p)).unsqueeze(0)).argmax()) for p, _ in test]
    wrong = [i for i, (_, y) in enumerate(test) if any(preds[n][i] != y for n in names)]
    right = [i for i, (_, y) in enumerate(test) if all(preds[n][i] == y for n in names)]
    chosen = (wrong[:4] + right[::max(1, len(right) // 4)])[:8]
    fig, ax = plt.subplots(len(chosen), 1 + len(names), figsize=(2.2 * (1 + len(names)), 2.2 * len(chosen)))
    for r, i in enumerate(chosen):
        path, y = test[i]
        img = load_image(DATA_DIR / path)
        ax[r, 0].imshow(img.resize((128, 128)))
        ax[r, 0].set_title(f"true: {CLASSES[y]}", fontsize=8)
        for c, n in enumerate(names, start=1):
            cam, probs, idx = cams[n](tf(img).unsqueeze(0))
            ax[r, c].imshow(overlay(img, cam))
            tag = "real only" if n.endswith("real") else "+ synthetic"
            ax[r, c].set_title(f"{tag}: {CLASSES[idx]} ({probs[idx]:.2f})", fontsize=8,
                               color=TEXT if idx == y else "#d03b3b")
        for a in ax[r]:
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
    fig.tight_layout()
    fig.suptitle("Grad-CAM on test images (red title = wrong prediction)", fontsize=10, color=TEXT, y=1.01)
    fig.savefig(FIG / "gradcam_examples.png")
    plt.close(fig)


def latency():
    rows = []
    x = eval_transform()(load_image(DATA_DIR / load_split()["test"][0][0])).unsqueeze(0)
    for n in ["resnet18_real", "resnet18_synthetic"]:
        if not (MODELS_DIR / f"{n}.pt").exists():
            continue
        m = load_trained(MODELS_DIR / f"{n}.pt")
        with torch.no_grad():
            for _ in range(5):
                m(x)
            t = time.perf_counter()
            for _ in range(50):
                m(x)
        ms = (time.perf_counter() - t) / 50 * 1000
        rows.append({"model": n, "device": "cpu", "threads": torch.get_num_threads(), "batch": 1,
                     "mean_ms": round(ms, 2), "images_per_sec": round(1000 / ms, 1)})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "latency.csv", index=False)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    table = summary_table(runs)
    print(pd.read_csv(RESULTS_DIR / "metrics_summary.csv").to_string(index=False))
    comparison_chart(table, runs)
    training_curves(runs)
    confusion(runs)
    dataset_table()
    synthetic_gallery()
    gradcam_figure()
    latency()


if __name__ == "__main__":
    main()
