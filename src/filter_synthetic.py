import argparse

import pandas as pd
import torch

from common import CLASSES, MODELS_DIR, SYNTH_DIR, eval_transform, load_image, load_split, load_trained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18_real_s42")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    table = pd.read_csv(SYNTH_DIR / "synthetic.csv", dtype=str)
    train_sources = {p for p, _ in load_split()["train"]}
    table = table[table.source.isin(train_sources)].reset_index(drop=True)
    model = load_trained(MODELS_DIR / f"{args.model}.pt")
    tf = eval_transform()
    probs = []
    with torch.no_grad():
        for start in range(0, len(table), 64):
            chunk = table.iloc[start:start + 64]
            x = torch.stack([tf(load_image(SYNTH_DIR / f)) for f in chunk.file])
            p = model(x).softmax(1)
            probs += [p[i, CLASSES.index(c)].item() for i, c in enumerate(chunk.label)]
    table["p_true"] = [round(p, 4) for p in probs]
    table["keep"] = (table.p_true >= args.threshold).astype(int)
    table.to_csv(SYNTH_DIR / "synthetic.csv", index=False)
    print(f"kept {table.keep.sum()} of {len(table)}")
    print(table.groupby("style").keep.mean().round(3))


if __name__ == "__main__":
    main()
