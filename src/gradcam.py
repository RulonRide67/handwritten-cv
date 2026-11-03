import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = model.layer4[-1]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        probs = logits.softmax(1)[0]
        if class_idx is None:
            class_idx = int(probs.argmax())
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
        return cam.numpy(), probs.detach().numpy(), class_idx


def overlay(image, cam, alpha=0.45):
    base = image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    heat = (colormaps["jet"](cam)[:, :, :3] * 255).astype(np.uint8)
    mixed = (np.asarray(base, dtype=np.float32) * (1 - alpha) + heat * alpha).astype(np.uint8)
    return Image.fromarray(mixed)
