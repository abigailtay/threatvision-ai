"""Generate Haar wavelet transformed images from the raw malware dataset.

Walks the ImageFolder structure under RAW_ROOT and writes wavelet images to
WAVELET_ROOT preserving the same train/val/test/<class>/ layout and filenames.

Run once before training the wavelet CNN branch.
"""

import argparse
import os

import numpy as np
import pywt
from PIL import Image


def wavelet_transform(image_array, wavelet="haar"):
    """Return the approximation coefficient (cA) scaled to 0-255 uint8."""
    coeffs2 = pywt.dwt2(image_array, wavelet)
    cA, (_, _, _) = coeffs2
    cA = np.clip(255 * (cA - cA.min()) / (cA.ptp() + 1e-9), 0, 255).astype(np.uint8)
    return cA


def generate_wavelet_dataset(raw_root, wavelet_root,
                             splits=("train", "val", "test"),
                             wavelet="haar"):
    for split in splits:
        raw_split_dir = os.path.join(raw_root, split)
        wav_split_dir = os.path.join(wavelet_root, split)
        os.makedirs(wav_split_dir, exist_ok=True)
        if not os.path.isdir(raw_split_dir):
            print(f"Skipping missing split: {raw_split_dir}")
            continue

        families = [
            f for f in os.listdir(raw_split_dir)
            if os.path.isdir(os.path.join(raw_split_dir, f))
            and not f.startswith(".")
        ]
        for family in families:
            raw_family_dir = os.path.join(raw_split_dir, family)
            wav_family_dir = os.path.join(wav_split_dir, family)
            os.makedirs(wav_family_dir, exist_ok=True)

            files = [f for f in os.listdir(raw_family_dir) if f.endswith(".png")]
            for filename in files:
                raw_path = os.path.join(raw_family_dir, filename)
                wav_path = os.path.join(wav_family_dir, filename)
                img = Image.open(raw_path).convert("L")
                arr = np.array(img)
                wav_arr = wavelet_transform(arr, wavelet=wavelet)
                Image.fromarray(wav_arr).save(wav_path)
        print(f"Done: {split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True,
                        help="Root of raw ImageFolder dataset.")
    parser.add_argument("--wavelet_root", required=True,
                        help="Destination root for wavelet images.")
    parser.add_argument("--wavelet", default="haar")
    args = parser.parse_args()
    generate_wavelet_dataset(args.raw_root, args.wavelet_root,
                             wavelet=args.wavelet)


if __name__ == "__main__":
    main()
