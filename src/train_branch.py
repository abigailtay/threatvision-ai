"""Train a single branch of the ThreatVisionAI ensemble.

Supported branches:
    raw      -> ResNet-18 on raw bytecode images
    wavelet  -> ResNet-18 on Haar wavelet images
    vit      -> ViT-Tiny on raw bytecode images

Example:
    python train_branch.py --branch raw      --data_root ./data/raw      --out_dir ./models
    python train_branch.py --branch wavelet  --data_root ./data/wavelet  --out_dir ./models
    python train_branch.py --branch vit      --data_root ./data/raw      --out_dir ./models
"""

import argparse
import os

import torch
from torch import nn, optim

from data_loaders import get_dataloaders
from models import build_resnet18, build_vit_tiny
from utils import (
    plot_confusion_matrix,
    save_metrics_plot,
    train_epoch,
    validate_epoch,
)


CHECKPOINT_NAMES = {
    "raw": "best_resnet18.pth",
    "wavelet": "best_wavelet_resnet18.pth",
    "vit": "best_vit_tiny_raw.pth",
}


def build_model(branch, num_classes):
    if branch in ("raw", "wavelet"):
        return build_resnet18(num_classes)
    if branch == "vit":
        return build_vit_tiny(num_classes)
    raise ValueError(f"Unknown branch: {branch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True,
                        choices=["raw", "wavelet", "vit"])
    parser.add_argument("--data_root", required=True,
                        help="Root with train/, val/, test/ subdirs.")
    parser.add_argument("--out_dir", default="./models")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_weights, classes = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    model = build_model(args.branch, num_classes).to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metrics = {k: [] for k in
               ["train_loss", "val_loss", "train_acc", "val_acc",
                "train_f1", "val_f1"]}
    best_val_f1 = 0.0
    ckpt_path = os.path.join(args.out_dir, CHECKPOINT_NAMES[args.branch])

    last_val_preds, last_val_labels = [], []
    for epoch in range(args.epochs):
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )

        metrics["train_loss"].append(tr_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(tr_acc)
        metrics["val_acc"].append(val_acc)
        metrics["train_f1"].append(tr_f1)
        metrics["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"TrainLoss {tr_loss:.4f} Acc {tr_acc:.4f} F1 {tr_f1:.4f} | "
            f"ValLoss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            last_val_preds, last_val_labels = val_preds, val_labels

    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Saved best checkpoint to: {ckpt_path}")

    model_name = f"{args.branch}_branch"
    save_metrics_plot(metrics, model_name, args.results_dir)
    cm_path = os.path.join(args.results_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(last_val_labels, last_val_preds, classes, cm_path,
                          title=f"{model_name} Validation Confusion Matrix")


if __name__ == "__main__":
    main()
