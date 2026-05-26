# ThreatVisionAI

**A Hybrid CNN-ViT Framework for Image-Based Malware Classification**

Multi-branch deep learning framework for static malware family classification on bytecode-image representations. Three branches (two CNNs and one Vision Transformer) are trained independently and fused at inference time with weighted soft voting.

This repository accompanies the paper accepted to the **IEEE World AI IoT Congress (AIIoT) 2026** and is being prepared for inclusion in IEEE Xplore.

## Architecture

Three independently trained branches:

1. **Raw CNN**: ResNet-18 on grayscale bytecode images, 3-channel input via `Grayscale(num_output_channels=3)` for ImageNet-style input.
2. **Wavelet CNN**: ResNet-18 on Haar wavelet approximation coefficients (`pywt.dwt2`, level 1), same input shape.
3. **ViT-Tiny**: `vit_tiny_patch16_224` on raw bytecode images.

At test time, per-branch softmax probabilities are combined with fixed weights:

```
P_ensemble = 0.50 * P_raw + 0.40 * P_wavelet + 0.10 * P_vit
```

If the ViT checkpoint is unavailable, the scripts fall back to a two-branch ensemble with weights renormalized to 0.556 and 0.444.

## Dataset

The model is trained and evaluated on the **Malimg** malware family dataset (25 classes). The repository ships no dataset files; place the data under `data/raw/{train,val,test}/<class_name>/*.png` before running anything. Per-class counts are highly imbalanced; training uses inverse-frequency class weights in the cross-entropy loss.

To produce the wavelet branch's inputs, run `wavelet_generation.py` once over the raw dataset to populate `data/wavelet/{train,val,test}/<class_name>/*.png` mirroring the raw layout.

## Results

Headline test-set numbers from the AIIoT 2026 paper:

| Branch         | Test accuracy | Weighted F1 |
| -------------- | :-----------: | :---------: |
| Raw CNN        |    0.9729     |    n/a      |
| Wavelet CNN    |    0.9791     |    n/a      |
| ViT-Tiny       |    0.9572     |    n/a      |
| **Ensemble**   |  **0.9801**   |    n/a      |

### Known failure mode

The dominant per-class failure is on **Autorun.K** (F1 = 0), which is systematically misclassified as **Yuner.A**. The two families have near-identical visual structure when rendered as bytecode images, and Grad-CAM (`gradcam.py`) shows the model attending to overlapping regions in both. This is a limitation of the bytecode-image representation rather than the ensemble itself.

## Repository layout

```
threatvision-ai/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_loaders.py             # ImageFolder loaders + paired raw/wavelet dataset
│   ├── wavelet_generation.py       # Haar DWT preprocessing
│   ├── models.py                   # ResNet-18 and ViT-Tiny builders + checkpoint loader
│   ├── utils.py                    # Train/validate loops, plotting
│   ├── train_branch.py             # Train raw CNN, wavelet CNN, or ViT-Tiny
│   ├── evaluate_ensemble.py        # Weighted soft voting on test set
│   ├── fgsm_robustness.py          # FGSM adversarial robustness sweep
│   ├── gradcam.py                  # Grad-CAM for Autorun.K vs Yuner.A
│   └── plot_class_distribution.py  # Training-set class balance figure
└── notebooks/
    └── demo.ipynb                  # End-to-end walkthrough: load checkpoints,
                                    # run ensemble, render confusion matrix and Grad-CAM
```

## Reproduction

### Quick look without training

If you just want to inspect the architecture and trained behavior, open `notebooks/demo.ipynb`. It loads the three checkpoints, runs weighted soft voting on the test set, prints the per-class report, and renders the Autorun.K vs Yuner.A Grad-CAM figure inline. It runs on CPU.

### Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Data preparation

Arrange the Malimg dataset under `data/raw/` with the standard ImageFolder layout, then generate the wavelet variant:

```bash
python src/wavelet_generation.py \
  --raw_root ./data/raw \
  --wavelet_root ./data/wavelet
```

### Train the three branches

Each branch is trained independently. The raw CNN and wavelet CNN consume their respective roots; the ViT is trained on raw images.

```bash
python src/train_branch.py --branch raw     --data_root ./data/raw     --out_dir ./models
python src/train_branch.py --branch wavelet --data_root ./data/wavelet --out_dir ./models
python src/train_branch.py --branch vit     --data_root ./data/raw     --out_dir ./models
```

Defaults: 20 epochs, Adam at `lr=1e-4`, batch size 32, weighted cross-entropy. Override with `--epochs`, `--lr`, `--batch_size`.

Checkpoints are saved as `best_resnet18.pth`, `best_wavelet_resnet18.pth`, and `best_vit_tiny_raw.pth` under `--out_dir`.

### Evaluate the ensemble

```bash
python src/evaluate_ensemble.py \
  --raw_root ./data/raw \
  --wavelet_root ./data/wavelet \
  --models_dir ./models \
  --results_dir ./results
```

Writes per-class metrics, the ensemble confusion matrix, and a CSV summary to `./results/`.

### FGSM adversarial robustness

```bash
python src/fgsm_robustness.py \
  --raw_root ./data/raw \
  --wavelet_root ./data/wavelet \
  --models_dir ./models \
  --results_dir ./results \
  --epsilons 0.0 0.01 0.03 0.05
```

Each branch is attacked through its own input; the ensemble is recomputed from the post-attack softmax probabilities.

### Grad-CAM interpretability figure

```bash
python src/gradcam.py \
  --models_dir ./models \
  --raw_root ./data/raw \
  --figure_path ./figures/gradcam_autorun_yuner.png
```

By default this picks the first PNG under `data/raw/test/Autorun.K/` and `data/raw/test/Yuner.A/`. Pass `--autorun_image` and `--yuner_image` to use specific samples.

## Citation

```bibtex
@inproceedings{taylor2026threatvisionai,
  author    = {Taylor, Allyson and BusiReddyGari, Prashanth},
  title     = {ThreatVisionAI: A Hybrid CNN-ViT Framework for Image-Based Malware Classification},
  booktitle = {Proceedings of the IEEE World AI IoT Congress (AIIoT)},
  year      = {2026},
  note      = {To appear in IEEE Xplore}
}
```

## License

MIT. See [LICENSE](LICENSE).
