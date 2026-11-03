import base64
import io
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import CLASSES, DATA_DIR, MODELS_DIR, SYNTH_DIR, eval_transform, load_trained
from gradcam import GradCAM, overlay

MODEL_FILES = {"synthetic": "resnet18_synthetic.pt", "real": "resnet18_real.pt"}
MODEL_NAMES = {"synthetic": "ResNet18 + TA-TiTok/MaskGen synthetic data (A2)",
               "real": "ResNet18 real data only (A1 baseline)"}
TIMING_RUNS = 10

app = Flask(__name__)
torch.set_grad_enabled(True)
transform = eval_transform()
models = {k: load_trained(MODELS_DIR / f) for k, f in MODEL_FILES.items() if (MODELS_DIR / f).exists()}
cams = {k: GradCAM(m) for k, m in models.items()}
latency_log = {k: [] for k in models}
synthetic = pd.read_csv(SYNTH_DIR / "synthetic.csv", dtype=str) if (SYNTH_DIR / "synthetic.csv").exists() else None


def environment():
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({platform.processor() or platform.machine()})"
    return {"device": device, "threads": torch.get_num_threads(), "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "torch": torch.__version__}


def crop_drawing(img):
    gray = np.asarray(img.convert("L"))
    ys, xs = np.where(gray < 200)
    if len(xs) == 0:
        return img
    pad = int(0.15 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 4
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, img.width), min(ys.max() + pad, img.height))
    return paper_look(img.crop(box))


def paper_look(img, background=185, size=56):
    ink = 1 - np.asarray(img, dtype=np.float32).mean(2, keepdims=True) / 255
    out = background * (1 - ink) + np.array([60, 60, 110], dtype=np.float32) * ink
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    scale = size / max(img.size)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html", models=[(k, MODEL_NAMES[k]) for k in models], classes=CLASSES,
                           env=environment(), has_gallery=synthetic is not None)


@app.route("/predict", methods=["POST"])
def predict():
    key = request.form.get("model", next(iter(models)))
    if key not in models or "image" not in request.files:
        abort(400)
    img = Image.open(request.files["image"].stream)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    if request.form.get("drawn") == "1":
        img = crop_drawing(img)
    x = transform(img).unsqueeze(0)
    model = models[key]
    with torch.no_grad():
        model(x)
        start = time.perf_counter()
        for _ in range(TIMING_RUNS):
            model(x)
        ms = (time.perf_counter() - start) / TIMING_RUNS * 1000
    latency_log[key].append(ms)
    cam, probs, idx = cams[key](x)
    top = np.argsort(probs)[::-1][:3]
    return jsonify({
        "label": CLASSES[idx],
        "top": [{"label": CLASSES[i], "prob": round(float(probs[i]), 4)} for i in top],
        "input": to_data_url(img.resize((128, 128))),
        "gradcam": to_data_url(overlay(img, cam)),
        "latency_ms": round(ms, 2),
        "avg_latency_ms": round(float(np.mean(latency_log[key])), 2),
        "requests": len(latency_log[key]),
        "model": MODEL_NAMES[key],
    })


@app.route("/gallery/<label>")
def gallery(label):
    if synthetic is None or label not in CLASSES:
        abort(404)
    rows = synthetic[synthetic.label == label]
    sources = sorted(rows.source.unique())[:4]
    return jsonify([{"source": "/real/" + s,
                     "variants": [{"url": "/synthetic/" + r.file, "style": r.style,
                                   "p_true": r.p_true if "p_true" in rows else "",
                                   "keep": r.keep if "keep" in rows else ""}
                                  for r in rows[rows.source == s].itertuples()]}
                    for s in sources])


@app.route("/real/<path:name>")
def real_image(name):
    return send_from_directory(DATA_DIR, name)


@app.route("/synthetic/<path:name>")
def synthetic_image(name):
    return send_from_directory(SYNTH_DIR, name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
