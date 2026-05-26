"""Shared training, validation, and plotting utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def save_metrics_plot(metrics_dict, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(metrics_dict["train_loss"], label="Train")
    axes[0].plot(metrics_dict["val_loss"], label="Val")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(metrics_dict["train_acc"], label="Train")
    axes[1].plot(metrics_dict["val_acc"], label="Val")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(metrics_dict["train_f1"], label="Train")
    axes[2].plot(metrics_dict["val_f1"], label="Val")
    axes[2].set_title(f"{model_name} Weighted F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    out_path = os.path.join(save_dir, f"{model_name}_metrics.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(all_labels, all_preds, classes, save_path,
                          title="Confusion Matrix"):
    cm = confusion_matrix(all_labels, all_preds)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(6, n * 0.35)))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=classes, yticklabels=classes,
        cmap="Blues", ax=ax, annot_kws={"fontsize": 7},
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(class_counts, class_names, save_path,
                            large_count_cutoff=200):
    """Bar plot of per-class sample counts, colored by majority/minority."""
    fig, ax = plt.subplots(figsize=(14, 7))
    values = [class_counts[c] for c in class_names]
    colors = ["tab:blue" if v > large_count_cutoff else "tab:orange"
              for v in values]
    bars = ax.bar(class_names, values, color=colors)

    ax.set_xlabel("Malware Family", fontsize=13)
    ax.set_ylabel("Number of Images", fontsize=13)
    ax.set_title("Class Distribution in Training Set", fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 3, f"{int(height)}",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
