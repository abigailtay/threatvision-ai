"""Data loaders for the Malimg malware family dataset.

Expects an ImageFolder layout:
    <root>/train/<class_name>/*.png
    <root>/val/<class_name>/*.png
    <root>/test/<class_name>/*.png

Two roots are used in the pipeline:
    - raw bytecode-image data
    - wavelet-transformed data (see wavelet_generation.py)
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_train_val_transforms(image_size=224, augment_train=True):
    if augment_train:
        train_tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return train_tfms, val_tfms


def get_dataloaders(root_dir, batch_size=32, num_workers=4, augment_train=True):
    """Return train_loader, val_loader, class_weights, classes."""
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val directory not found: {val_dir}")

    train_tfms, val_tfms = get_train_val_transforms(augment_train=augment_train)

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_tfms)

    class_indices = [label for _, label in train_ds.samples]
    class_counts = Counter(class_indices)
    class_sample_counts = [class_counts[i] for i in range(len(train_ds.classes))]
    class_weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, class_weights, train_ds.classes


def get_test_loader(root_dir, batch_size=32, num_workers=4):
    """Return test_loader, classes."""
    test_dir = os.path.join(root_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_ds = datasets.ImageFolder(root=test_dir, transform=test_tfms)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_loader, test_ds.classes


class PairedRawWaveletDataset(Dataset):
    """Returns aligned (raw, wavelet, label) triples.

    Used by the hybrid ensemble and the FGSM robustness experiment so that
    raw and wavelet branches see the same underlying samples.
    Alignment is done by filename.
    """

    def __init__(self, raw_root, wav_root, split, transform_raw=None,
                 transform_wav=None):
        raw_dir = os.path.join(raw_root, split)
        wav_dir = os.path.join(wav_root, split)
        self.transform_raw = transform_raw
        self.transform_wav = transform_wav

        dummy = transforms.Compose([transforms.ToTensor()])
        raw_ds = datasets.ImageFolder(raw_dir, transform=dummy)
        self.classes = raw_ds.classes
        self.class_to_idx = raw_ds.class_to_idx
        self.raw_dir = raw_dir
        self.wav_dir = wav_dir
        self.samples = [(path, label) for path, label in raw_ds.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_path, label = self.samples[idx]
        rel = os.path.relpath(raw_path, self.raw_dir)
        wav_path = os.path.join(self.wav_dir, rel)

        raw_img = datasets.folder.default_loader(raw_path)
        wav_img = datasets.folder.default_loader(wav_path)
        if self.transform_raw:
            raw_img = self.transform_raw(raw_img)
        if self.transform_wav:
            wav_img = self.transform_wav(wav_img)
        return raw_img, wav_img, label


def get_paired_test_loader(raw_root, wav_root, batch_size=32, num_workers=4):
    """Paired (raw, wavelet) test loader aligned by filename."""
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = PairedRawWaveletDataset(
        raw_root, wav_root, split="test",
        transform_raw=tfms, transform_wav=tfms,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, ds.classes
