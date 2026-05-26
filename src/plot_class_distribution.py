"""Plot per-class sample counts from the training split.

Example:
    python plot_class_distribution.py \
        --data_root ./data/raw \
        --save_path ./figures/class_distribution.png
"""

import argparse
import os
from collections import Counter

from torchvision import datasets, transforms

from utils import plot_class_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Root containing train/<class>/*.png")
    parser.add_argument("--save_path",
                        default="./figures/class_distribution.png")
    parser.add_argument("--cutoff", type=int, default=200,
                        help="Sample count above which a class is plotted blue.")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(train_dir)

    dummy = transforms.Compose([transforms.ToTensor()])
    ds = datasets.ImageFolder(train_dir, transform=dummy)
    counts = Counter([y for _, y in ds.samples])

    counts_sorted = {
        ds.classes[i]: counts[i] for i in sorted(counts, key=counts.get,
                                                 reverse=True)
    }
    class_names = list(counts_sorted.keys())
    by_name = {name: counts_sorted[name] for name in class_names}

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plot_class_distribution(by_name, class_names, args.save_path,
                            large_count_cutoff=args.cutoff)
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
