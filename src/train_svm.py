import json

import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import RESULTS_DIR, load_image, load_split, real_items
from train import metrics


def features(items):
    out = []
    for path, _ in items:
        img = np.asarray(load_image(path).convert("L").resize((64, 64)), dtype=np.float32) / 255.0
        out.append(hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)))
    return np.array(out), np.array([y for _, y in items])


def main():
    split = load_split()
    x_tr, y_tr = features(real_items(split["train"]))
    x_va, y_va = features(real_items(split["val"]))
    x_te, y_te = features(real_items(split["test"]))
    best_c, best_acc = None, -1
    for c in [0.1, 1, 10, 100]:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma="scale")).fit(x_tr, y_tr)
        acc = accuracy_score(y_va, model.predict(x_va))
        print(f"C={c} val acc={acc:.3f}")
        if acc > best_acc:
            best_c, best_acc = c, acc
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=best_c, gamma="scale")).fit(x_tr, y_tr)
    pred = model.predict(x_te)
    result = {"name": "svm_hog", "C": best_c, "best_val_acc": best_acc, "test": metrics(y_te, pred),
              "test_predictions": pred.tolist()}
    (RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runs" / "svm_hog.json").write_text(json.dumps(result, indent=1))
    print({k: round(v, 4) for k, v in result["test"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
