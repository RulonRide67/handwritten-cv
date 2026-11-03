import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import ConcatDataset, DataLoader

from common import (CLASSES, MODELS_DIR, RESULTS_DIR, SYNTH_DIR, ImageListDataset, build_resnet18,
                    eval_transform, load_split, real_items, set_seed, train_transform)


def synthetic_items(mode):
    if mode == "none":
        return []
    table = pd.read_csv(SYNTH_DIR / "synthetic.csv")
    if mode == "filtered":
        table = table[table.keep == 1]
    return [(str(SYNTH_DIR / p), CLASSES.index(c)) for p, c in zip(table.file, table.label)]


class MixSampler(torch.utils.data.Sampler):
    def __init__(self, n_real, n_synth, k, generator):
        self.n_real, self.n_synth, self.k, self.g = n_real, n_synth, min(k, n_synth), generator

    def __len__(self):
        return self.n_real + self.k

    def __iter__(self):
        synth = self.n_real + torch.randperm(self.n_synth, generator=self.g)[:self.k]
        idx = torch.cat([torch.arange(self.n_real), synth])
        return iter(idx[torch.randperm(len(idx), generator=self.g)].tolist())


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x).argmax(1).tolist()
        ys += y.tolist()
    return np.array(ys), np.array(ps)


def metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0,
                                                        labels=range(len(CLASSES)))
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1,
            "f1_per_class": dict(zip(CLASSES, f1_class.round(4).tolist())),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--synthetic", choices=["none", "all", "filtered"], default="none")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-per-epoch", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    split = load_split()
    train_parts = [ImageListDataset(real_items(split["train"]), train_transform())]
    synth = synthetic_items(args.synthetic)
    if synth:
        train_parts.append(ImageListDataset(synth, train_transform()))
    train_ds = ConcatDataset(train_parts)
    g = torch.Generator().manual_seed(args.seed)
    if synth:
        sampler = MixSampler(len(split["train"]), len(synth), args.synthetic_per_epoch or len(split["train"]), g)
        train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g)
    val_dl = DataLoader(ImageListDataset(real_items(split["val"]), eval_transform()), batch_size=64)
    test_dl = DataLoader(ImageListDataset(real_items(split["test"]), eval_transform()), batch_size=64)

    model = build_resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    MODELS_DIR.mkdir(exist_ok=True)
    out_dir = RESULTS_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = MODELS_DIR / f"{args.name}.pt"

    history, best = [], -1
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, criterion, optimizer)
        vl, va = run_epoch(model, val_dl, criterion)
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
        if va >= best:
            best = va
            torch.save(model.state_dict(), weights)
        print(f"[{args.name}] epoch {epoch:02d} train {tl:.3f}/{ta:.3f} val {vl:.3f}/{va:.3f}", flush=True)
    train_time = time.time() - start

    model.load_state_dict(torch.load(weights, weights_only=True))
    y_true, y_pred = predict(model, test_dl)
    result = {"name": args.name, "synthetic": args.synthetic, "seed": args.seed,
              "train_real": len(split["train"]), "train_synthetic": len(synth),
              "synthetic_per_epoch": min(args.synthetic_per_epoch or len(split["train"]), len(synth)),
              "best_val_acc": best, "train_time_sec": round(train_time, 1),
              "test": metrics(y_true, y_pred), "history": history,
              "test_predictions": y_pred.tolist()}
    (out_dir / f"{args.name}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: round(v, 4) for k, v in result["test"].items() if isinstance(v, float)}))


if __name__ == "__main__":
    main()
