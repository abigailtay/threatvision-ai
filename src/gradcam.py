"""Grad-CAM visualization for the Raw CNN branch.

By default, produces the side-by-side Autorun.K vs Yuner.A figure used in the
ThreatVisionAI paper to discuss the dominant per-class failure mode: Autorun.K
samples are visually similar to Yuner.A and the model attends to overlapping
regions. The script also supports a generic mode that produces overlays for
any (true_class, predicted_class) sample paths the caller provides.

Example:
    python gradcam.py \
        --models_dir ./models \
        --raw_root ./data/raw \
        --figure_path ./figures/gradcam_autorun_yuner.png
"""

import argparse
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from models import build_resnet18, load_checkpoint


MALIMG_CLASSES_25 = [
    "Adialer.C", "Agent.FYI", "Allaple.A", "Allaple.L", "Alueron.gen!J",
    "Autorun.K", "C2LOP.P", "C2LOP.gen!g", "Dialplatform.B", "Dontovo.A",
    "Fakerean", "Instantaccess", "Lolyda.AA1", "Lolyda.AA2", "Lolyda.AA3",
    "Lolyda.AT", "Malex.gen!J", "Obfuscator.AD", "Rbot!gen", "Skintrim.N",
    "Swizzor.gen!E", "Swizzor.gen!I", "VB.AT", "Wintrim.BX", "Yuner.A",
]


def load_image(path, image_size=224):
    img = Image.open(path).convert("L")
    img_resized = img.resize((image_size, image_size))
    img_rgb = Image.fromarray(np.array(img_resized)).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return img_resized, tfm(img_rgb)


def compute_gradcam(model, img_tensor, target_class, device,
                    target_module=None):
    """Compute Grad-CAM for a single image tensor.

    By default, hooks the last conv layer of layer4 (resnet18.layer4[1].conv2).
    """
    if target_module is None:
        target_module = model.layer4[1].conv2

    gradients, activations = [], []

    def forward_hook(_module, _input, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h_f = target_module.register_forward_hook(forward_hook)
    h_b = target_module.register_full_backward_hook(backward_hook)

    t = img_tensor.unsqueeze(0).to(device)
    out = model(t)
    model.zero_grad()
    out[0, target_class].backward()

    h_f.remove()
    h_b.remove()

    grads = gradients[0].squeeze()
    acts = activations[0].squeeze()
    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = torch.clamp(cam, min=0).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_cam_on_image(img_gray, cam, image_size=224, alpha=0.5):
    img_np = np.array(img_gray, dtype=np.float32)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8))
            .resize((image_size, image_size), Image.BILINEAR)
    ) / 255.0
    heatmap = cm.jet(cam_resized)[:, :, :3]
    overlay = alpha * img_np[:, :, None] + (1 - alpha) * heatmap
    return np.clip(overlay, 0, 1)


def make_autorun_vs_yuner_figure(model, device, autorun_path, yuner_path,
                                 save_path):
    autorun_idx = MALIMG_CLASSES_25.index("Autorun.K")
    yuner_idx = MALIMG_CLASSES_25.index("Yuner.A")

    autorun_img, autorun_tensor = load_image(autorun_path)
    yuner_img, yuner_tensor = load_image(yuner_path)

    autorun_cam = compute_gradcam(model, autorun_tensor, autorun_idx, device)
    yuner_cam = compute_gradcam(model, yuner_tensor, yuner_idx, device)

    autorun_overlay = overlay_cam_on_image(autorun_img, autorun_cam)
    yuner_overlay = overlay_cam_on_image(yuner_img, yuner_cam)

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.5))

    axes[0, 0].imshow(autorun_img, cmap="gray")
    axes[0, 0].set_title("Autorun.K", fontsize=8, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(autorun_overlay)
    axes[0, 1].set_title("Grad-CAM", fontsize=8)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(yuner_img, cmap="gray")
    axes[1, 0].set_title("Yuner.A", fontsize=8, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(yuner_overlay)
    axes[1, 1].set_title("Grad-CAM", fontsize=8)
    axes[1, 1].axis("off")

    fig.suptitle("Grad-CAM: Autorun.K vs Yuner.A", fontsize=9, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM figure to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--raw_root", required=True,
                        help="Root of raw dataset (used to find example PNGs).")
    parser.add_argument("--checkpoint_name", default="best_resnet18.pth")
    parser.add_argument("--figure_path",
                        default="./figures/gradcam_autorun_yuner.png")
    parser.add_argument("--autorun_image", default=None,
                        help="Optional explicit path to an Autorun.K test image.")
    parser.add_argument("--yuner_image", default=None,
                        help="Optional explicit path to a Yuner.A test image.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(MALIMG_CLASSES_25)
    model = build_resnet18(num_classes)
    ckpt_path = os.path.join(args.models_dir, args.checkpoint_name)
    load_checkpoint(model, ckpt_path, device)
    model = model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    def first_png_in(family):
        family_dir = os.path.join(args.raw_root, "test", family)
        if not os.path.isdir(family_dir):
            raise FileNotFoundError(family_dir)
        for f in sorted(os.listdir(family_dir)):
            if f.endswith(".png"):
                return os.path.join(family_dir, f)
        raise FileNotFoundError(f"No PNGs in {family_dir}")

    autorun_path = args.autorun_image or first_png_in("Autorun.K")
    yuner_path = args.yuner_image or first_png_in("Yuner.A")
    print(f"Autorun.K sample: {autorun_path}")
    print(f"Yuner.A sample:   {yuner_path}")

    make_autorun_vs_yuner_figure(model, device, autorun_path, yuner_path,
                                 args.figure_path)


if __name__ == "__main__":
    main()
