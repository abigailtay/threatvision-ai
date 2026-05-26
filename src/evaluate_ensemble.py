"""Evaluate the ThreatVisionAI ensemble on the test set.

Uses weighted soft voting across the three trained branches:
    raw_cnn      weight 0.50
    wavelet_cnn  weight 0.40
    vit_tiny     weight 0.10

If the ViT checkpoint is missing, the script falls back to a two-branch
ensemble of raw + wavelet (weights renormalized to 0.50/0.90 and 0.40/0.90).

Example:
    python evaluate_ensemble.py \
        --raw_root ./data/raw \
        --wavelet_root ./data/wavelet \
        --models_dir ./models \
        --results_dir ./results
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_loaders import get_paired_test_loader
from models import build_resnet18, build_vit_tiny, load_checkpoint
from utils import plot_confusion_matrix


W_RAW_DEFAULT = 0.50
W_WAV_DEFAULT = 0.40
W_VIT_DEFAULT = 0.10


@torch.no_grad()
def collect_probs(raw_model, wav_model, vit_model, loader, device):
    """Run all available branches over the paired loader and return per-branch
    softmax probabilities plus ground-truth labels."""
    raw_ps, wav_ps, vit_ps, ys = [], [], [], []
    for x_raw, x_wav, y in loader:
        x_raw = x_raw.to(device)
        x_wav = x_wav.to(device)
        raw_ps.append(torch.softmax(raw_model(x_raw), dim=1).cpu())
        wav_ps.append(torch.softmax(wav_model(x_wav), dim=1).cpu())
        if vit_model is not None:
            vit_ps.append(torch.softmax(vit_model(x_raw), dim=1).cpu())
        ys.extend(y.numpy())
    raw_p = torch.cat(raw_ps)
    wav_p = torch.cat(wav_ps)
    vit_p = torch.cat(vit_ps) if vit_model is not None else None
    return raw_p, wav_p, vit_p, np.array(ys)


def fuse(raw_p, wav_p, vit_p, w_raw, w_wav, w_vit):
    if vit_p is not None:
        return w_raw * raw_p + w_wav * wav_p + w_vit * vit_p
    # Renormalize when ViT is unavailable.
    denom = w_raw + w_wav
    return (w_raw / denom) * raw_p + (w_wav / denom) * wav_p


def acc_f1(y_true, probs):
    preds = probs.argmax(1).numpy()
    return (
        accuracy_score(y_true, preds),
        f1_score(y_true, preds, average="weighted"),
        f1_score(y_true, preds, average="macro"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--wavelet_root", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--w_raw", type=float, default=W_RAW_DEFAULT)
    parser.add_argument("--w_wav", type=float, default=W_WAV_DEFAULT)
    parser.add_argument("--w_vit", type=float, default=W_VIT_DEFAULT)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader, classes = get_paired_test_loader(
        args.raw_root, args.wavelet_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Paired test samples: {len(loader.dataset)}")

    models_dir = Path(args.models_dir)
    raw_model = build_resnet18(num_classes)
    load_checkpoint(raw_model, str(models_dir / "best_resnet18.pth"), device)
    raw_model = raw_model.to(device).eval()

    wav_model = build_resnet18(num_classes)
    load_checkpoint(wav_model, str(models_dir / "best_wavelet_resnet18.pth"), device)
    wav_model = wav_model.to(device).eval()

    vit_model = None
    vit_path = models_dir / "best_vit_tiny_raw.pth"
    if vit_path.exists():
        try:
            vit_model = build_vit_tiny(num_classes)
            load_checkpoint(vit_model, str(vit_path), device)
            vit_model = vit_model.to(device).eval()
            print("ViT branch loaded.")
        except Exception as e:
            print(f"ViT load failed, continuing without it: {e}")
            vit_model = None
    else:
        print(f"No ViT checkpoint at {vit_path}, running two-branch ensemble.")

    raw_p, wav_p, vit_p, y_true = collect_probs(
        raw_model, wav_model, vit_model, loader, device
    )

    print("\n=== Per-branch test performance ===")
    for name, p in [("Raw CNN", raw_p), ("Wavelet CNN", wav_p),
                    ("ViT-Tiny", vit_p)]:
        if p is None:
            continue
        a, fw, fm = acc_f1(y_true, p)
        print(f"{name:<14} Acc={a:.4f} WeightedF1={fw:.4f} MacroF1={fm:.4f}")

    fused = fuse(raw_p, wav_p, vit_p,
                 args.w_raw, args.w_wav, args.w_vit)
    a, fw, fm = acc_f1(y_true, fused)
    print(f"\n=== Ensemble (weights raw={args.w_raw}, "
          f"wav={args.w_wav}, vit={args.w_vit}) ===")
    print(f"Accuracy:    {a:.4f}")
    print(f"Weighted F1: {fw:.4f}")
    print(f"Macro F1:    {fm:.4f}")

    print("\n=== Per-class classification report (ensemble) ===")
    fused_preds = fused.argmax(1).numpy()
    print(classification_report(y_true, fused_preds,
                                target_names=classes, digits=4))

    csv_path = os.path.join(args.results_dir, "ensemble_test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_accuracy",
                    "test_weighted_f1", "test_macro_f1"])
        for name, p in [("raw_cnn", raw_p), ("wavelet_cnn", wav_p),
                        ("vit_tiny", vit_p)]:
            if p is None:
                continue
            a, fw, fm = acc_f1(y_true, p)
            w.writerow([name, a, fw, fm])
        a, fw, fm = acc_f1(y_true, fused)
        tag = "ensemble_3branch" if vit_p is not None else "ensemble_2branch"
        w.writerow([tag, a, fw, fm])
    print(f"\nSaved metrics CSV to: {csv_path}")

    cm_path = os.path.join(args.results_dir,
                           "ensemble_test_confusion_matrix.png")
    plot_confusion_matrix(y_true, fused_preds, classes, cm_path,
                          title="ThreatVisionAI Ensemble Test Confusion Matrix")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
