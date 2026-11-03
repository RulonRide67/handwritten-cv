import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "dataset"
SYNTH_DIR = ROOT / "data" / "synthetic"
SPLIT_FILE = ROOT / "data" / "split.json"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

CLASSES = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "="]
SEED = 42
IMG_SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMAGENET_WEIGHTS_URL = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def list_samples(data_dir=DATA_DIR):
    samples = []
    for label, name in enumerate(CLASSES):
        for f in sorted((Path(data_dir) / name).iterdir()):
            if f.is_file():
                samples.append((f.relative_to(data_dir).as_posix(), label))
    return samples


def make_split():
    samples = list_samples()
    targets = np.array([s[1] for s in samples])
    idx = np.arange(len(samples))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=targets, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=targets[temp_idx], random_state=SEED)
    split = {name: [samples[i] for i in ids] for name, ids in
             [("train", train_idx), ("val", val_idx), ("test", test_idx)]}
    SPLIT_FILE.write_text(json.dumps(split, indent=1))
    return split


def load_split():
    if not SPLIT_FILE.exists():
        return make_split()
    return {k: [tuple(x) for x in v] for k, v in json.loads(SPLIT_FILE.read_text()).items()}


def train_transform(augment=True):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment:
        ops += [transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3)]
    ops += [transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def eval_transform():
    return train_transform(augment=False)


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        if path not in self.cache:
            self.cache[path] = load_image(path)
        return self.transform(self.cache[path]), label


def real_items(split_part):
    return [(str(DATA_DIR / p), y) for p, y in split_part]


def build_resnet18(pretrained=True):
    model = models.resnet18(weights=None)
    if pretrained:
        state = torch.hub.load_state_dict_from_url(IMAGENET_WEIGHTS_URL, map_location="cpu", progress=False,
                                                   weights_only=True)
        model.load_state_dict(state)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASSES)))
    return model


def load_trained(path, device="cpu"):
    model = build_resnet18(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()
