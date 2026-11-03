# Handwritten Digits and Math Symbols: SOTA Data Synthesis + Demo

Computer Vision, Assignment 2 (continuation of Assignment 1).

The task is to classify 13 handwritten classes: digits `0-9` and the symbols `+`, `-`, `=`.
Assignment 1 gave a HOG + SVM baseline and a ResNet18 model trained on 669 real images.
Assignment 2 adds a SOTA method: **TA-TiTok + MaskGen** (ByteDance, 2025) is used to create
synthetic "hard" variants of the real training images. The variants are filtered and added to training.
The project also has a small web app: upload or draw an image, get the prediction and a Grad-CAM heatmap.

## How synthetic data is made

1. A real training image is padded to a square and resized to 256x256 (for the `rotated` style it is also rotated by 12-25 degrees).
2. The TA-TiTok tokenizer (`tatitok_bl128_vq`) turns the image into 128 discrete tokens.
3. 35-60% of the tokens are replaced by a mask token.
4. MaskGen (`maskgen_vq_l`) fills the masked tokens again in 12 steps with classifier-free guidance,
   conditioned on a text prompt such as "a handwritten digit seven drawn with a thick black marker, ink bleeding into the paper".
5. TA-TiTok decodes the tokens back to an image (the decoder also uses the text prompt). The image is resized to the original crop size.
6. Six styles per image: `thin_pen`, `marker_bleed`, `pencil_grid`, `blurry_photo`, `faded_ink`, `rotated`.
7. Label filter ("pick best"): the A1-style ResNet18 (trained on real data only) scores every variant.
   A variant is kept only if the probability of its intended class is at least 0.5.

Synthetic images are made only from the **training** split, so validation and test sets stay 100% real.
During training each epoch uses all real training images plus the same number of randomly sampled synthetic images (1:1).

## Project structure

```
app/                     web app (Flask)
  app.py
  templates/index.html
colab/
  generate_synthetic.ipynb   synthetic data generation (Google Colab, T4 GPU)
  train_images.zip           training images to upload to Colab
data/
  dataset/                   real images, one folder per class
  synthetic/                 generated images + synthetic.csv (style, source image, filter score)
  split.json                 train / val / test split (70/15/15, seed 42, same as A1)
models/
  resnet18_real.pt           ResNet18, real data only (A1 baseline)
  resnet18_synthetic.pt      ResNet18, real + filtered synthetic data (A2)
results/
  metrics_summary.csv        main results table
  per_class_f1.csv
  dataset_summary.csv
  synthetic_by_style.csv
  synthetic_share.json
  latency.csv
  figures/                   plots for the report
  runs/                      raw results of every training run (json)
src/
  common.py                  data loading, split, transforms, model
  train_svm.py               HOG + SVM baseline
  train.py                   ResNet18 training (real only / + synthetic)
  filter_synthetic.py        label filter for synthetic images
  gradcam.py                 Grad-CAM
  make_figures.py            tables and figures
run_experiments.sh           runs all experiments in order
```

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Run the web app

```
cd app
python app.py
```

Open http://localhost:5000 in a browser. The app runs on CPU and uses the weights in `models/`.
On a phone in the same Wi-Fi network, open `http://<computer IP>:5000`; the upload button can use the camera.

## Reproduce the experiments

1. Synthetic data (already included in `data/synthetic/`): open `colab/generate_synthetic.ipynb` in Google Colab,
   select a T4 GPU, Run all, upload `colab/train_images.zip`, and unzip the downloaded `synthetic.zip` into `data/`.
2. Training and evaluation (CPU is enough but takes a few hours; a GPU is much faster):

```
bash run_experiments.sh
```

ImageNet weights for ResNet18 are downloaded automatically (timm `resnet18.a1_in1k` from GitHub releases).

## Results

Test set: 101 real images. ResNet18 rows are mean ± std over 3 seeds (42, 43, 44). Full tables are in `results/`.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| HOG + SVM (A1 baseline) | 0.802 | 0.812 | 0.803 | 0.795 |
| ResNet18, real data only (A1 baseline) | 0.927 ± 0.017 | 0.935 ± 0.015 | 0.932 ± 0.016 | 0.930 ± 0.015 |
| ResNet18 + synthetic, no filter (ablation) | 0.937 ± 0.020 | 0.944 ± 0.017 | 0.940 ± 0.020 | 0.938 ± 0.020 |
| ResNet18 + synthetic, filtered (A2) | 0.937 ± 0.012 | 0.946 ± 0.010 | 0.940 ± 0.011 | 0.938 ± 0.011 |

- Synthetic data: 2808 generated (6 per training image), 1176 kept by the filter (42%).
- Synthetic data gives +1.0 point accuracy and +0.8 point macro F1 on average. The filter does not change the mean but makes results more stable (std 0.012 vs 0.020).
- The test set is small (one image = about 1%), so the gain is inside the seed-to-seed spread.
- Inference: about 13 ms per image on CPU (2 threads), batch 1.

Figures: `results/figures/` (model comparison, training curves, confusion matrices, Grad-CAM examples, synthetic examples).

## Ethics and disclosure

- All synthetic images are produced by TA-TiTok + MaskGen and are marked as SYNTHETIC in the app and in the figures.
  They are stored separately in `data/synthetic/` and are used only for training, never for validation or testing.
- Some raw generations do not show a symbol at all (pens, paper, sometimes face-like shapes). The label filter removes most of them, and no faces of real people are in the data.
- The real handwriting was collected from 6 people (see the A1 report). Images contain only single symbols, no names or personal data.
- Uploaded and drawn images in the app are processed in memory and are not saved.
- TA-TiTok and MaskGen checkpoints are released for research use only.

## References

- Kim et al., "Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens" (TA-TiTok, MaskGen), 2025. https://arxiv.org/abs/2501.07730, code: https://github.com/bytedance/1d-tokenizer
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", 2017.
- He et al., "Deep Residual Learning for Image Recognition", 2016.
