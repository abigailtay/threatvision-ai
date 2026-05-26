"""Evaluate ThreatVisionAI under FGSM adversarial perturbation.

Raw and wavelet branches are each attacked through their own input. The ViT
branch (when present) is attacked via the raw image input it consumes.
The ensemble is then computed from the post-attack softmax probabilities.

Example:
    python fgsm_robustness.py \
        --raw_root ./data/raw \
        --wavelet_root ./data/wavelet \
        --models_dir ./models \
        --results_dir ./results \
        --epsilons 0.0 0.01 0.03 0.05
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score

from data_loaders import get_paired_test_loader
from models import build_resnet18, build_vit_tiny, load_checkpoint


W_RAW = 0.50
W_WAV = 0.40
W_VIT = 0.10


def fgsm_attack(model, x, y, epsilon, criterion, device):
    x_adv = x.clone().detach().requires_grad_(True)
    loss = criterion(model(x_adv), y.to(device))
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv + epsilon * x_adv.grad.sign()
        x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()


def evaluate_at_epsilon(eps, loader, raw_model, wav_model, vit_model,
                        criterion, device):
    raw_ps, wav_ps, vit_ps, ys = [], [], [], []

    for x_raw, x_wav, y in loader:
        x_raw = x_raw.to(device)
        x_wav = x_wav.to(device)
        y_dev = y.to(device)

        if eps == 0.0:
            x_raw_adv = x_raw
            x_wav_adv = x_wav
        else:
            x_raw_adv = fgsm_attack(raw_model, x_raw, y_dev, eps,
                                    criterion, device)
            x_wav_adv = fgsm_attack(wav_model, x_wav, y_dev, eps,
                                    criterion, device)

        with torch.no_grad():
            raw_ps.append(torch.softmax(raw_model(x_raw_adv), dim=1).cpu())
            wav_ps.append(torch.softmax(wav_model(x_wav_adv), dim=1).cpu())
            if vit_model is not None:
                vit_ps.append(torch.softmax(vit_model(x_raw_adv), dim=1).cpu())
        ys.extend(y.cpu().numpy())

    raw_p = torch.cat(raw_ps)
    wav_p = torch.cat(wav_ps)
    vit_p = torch.cat(vit_ps) if vit_model is not None else None
    y_true = np.array(ys)

    if vit_p is not None:
        ens_p = W_RAW * raw_p + W_WAV * wav_p + W_VIT * vit_p
    else:
        denom = W_RAW + W_WAV
        ens_p = (W_RAW / denom) * raw_p + (W_WAV / denom) * wav_p

    def acc(p):
        return round(accuracy_score(y_true, p.argmax(1).numpy()), 4)

    return {
        "epsilon": eps,
        "raw_acc": acc(raw_p),
        "wav_acc": acc(wav_p),
        "vit_acc": acc(vit_p) if vit_p is not None else "N/A",
        "ens_acc": acc(ens_p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--wavelet_root", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.0, 0.01, 0.03, 0.05])
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

    criterion = nn.CrossEntropyLoss()

    results = []
    for eps in args.epsilons:
        print(f"\n=== epsilon = {eps} ===")
        r = evaluate_at_epsilon(eps, loader, raw_model, wav_model, vit_model,
                                criterion, device)
        results.append(r)
        print(f"  Raw CNN     : {r['raw_acc']}")
        print(f"  Wavelet CNN : {r['wav_acc']}")
        print(f"  ViT-Tiny    : {r['vit_acc']}")
        print(f"  Full Hybrid : {r['ens_acc']}")

    out_path = os.path.join(args.results_dir, "fgsm_robustness.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {out_path}")

    print()
    header = f'{"Model":<14}'
    for r in results:
        header += f' eps={r["epsilon"]:>5}'
    print(header)
    print("-" * len(header))
    for branch, key in [("Raw CNN", "raw_acc"), ("Wavelet CNN", "wav_acc"),
                        ("ViT-Tiny", "vit_acc"), ("Full Hybrid", "ens_acc")]:
        line = f"{branch:<14}"
        for r in results:
            line += f" {str(r[key]):>9}"
        print(line)


if __name__ == "__main__":
    main()
