# DeepFakeImageClassifier

A comprehensive deep learning solution for detecting and classifying deepfake images. This project implements multiple state-of-the-art techniques to achieve high accuracy in distinguishing between real images and various types of AI-generated or manipulated images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training](#training)
- [Prediction](#prediction)
- [Data Augmentation](#data-augmentation)
- [Advanced Techniques](#advanced-techniques)
  - [MixUp and CutMix](#mixup-and-cutmix)
  - [Sharpness-Aware Minimization (SAM)](#sharpness-aware-minimization-sam)
  - [Model Ensembling](#model-ensembling)
- [Results Visualization](#results-visualization)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

DeepFakeImageClassifier is a PyTorch-based solution for identifying deepfake images. The system can classify input images into 5 distinct categories (one real class plus four deepfake classes). It includes custom ResNet architectures, data augmentation, regularization strategies, and utilities for training and inference.

## Features

- Custom ResNet18/34/50 backbones implemented from scratch
- Clean dataset pipeline with CSV-driven splits
- Moderate and strong augmentation pipelines
- Optional MixUp and CutMix regularization
- Optional Sharpness-Aware Minimization (SAM) optimizer wrapper
- Cosine annealing learning-rate schedule and label smoothing
- Early stopping and best-checkpoint saving (intended)
- Prediction script that produces a submission-ready CSV

## Requirements

- Python 3.8+
- PyTorch 1.10+ and TorchVision
- CUDA-capable GPU (recommended)
- Other packages: tqdm, pandas, numpy, pillow, matplotlib, seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mihneave/DeepFakeImageClassifier.git
   cd DeepFakeImageClassifier
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision tqdm pandas numpy pillow matplotlib seaborn
   ```

## Project Structure

```text
DeepFakeImageClassifier/
├─ train.py           # Main training script
├─ predict.py         # Inference script for test predictions
├─ model.py           # Custom ResNet implementations and ImageClassifier wrapper
├─ dataset.py         # CSV-driven Dataset for train/val/test
├─ augment.py         # Offline augmentation and CSV generation
├─ mixup.py           # MixUp and CutMix utilities
├─ sam.py             # SAM optimizer wrapper
├─ ensemble.py        # (Optional) ensemble training/inference utilities
├─ train/             # Training images (image_id.png)
├─ validation/        # Validation images (image_id.png)
├─ test/              # Test images (image_id.png)
├─ models/            # Saved checkpoints and plots
└─ README.md
```

## Dataset

Expected layout:

- Images: PNG files named as `<image_id>.png`
- CSV files:
  - train.csv: columns `image_id,label` (label in [0..4])
  - validation.csv: columns `image_id,label`
  - test.csv: column `image_id`

Example rows:

```csv
# train.csv
image_id,label
00001,3
00002,0

# validation.csv
image_id,label
10001,2
10002,4

# test.csv
image_id
20001
20002
```

Place images under the corresponding directories: `train/`, `validation/`, and `test/`.

## Training

Run the default training workflow:

```bash
python train.py
```

This will:

- Detect GPU availability
- Initialize a ResNet18 classifier
- Train for 80 epochs with AdamW and cosine LR schedule
- Apply moderate augmentation
- Evaluate on validation set each epoch
- Save artifacts (plots, checkpoints) under `models/`

Key defaults (see train.py):

- num_classes: 5
- num_epochs: 80
- batch_size: 32
- learning_rate: 0.0015
- label_smoothing: 0.1

Notes:

- Early stopping/best-checkpoint logic is intended; ensure it matches your needs.
- Reduce batch size if you run out of GPU memory.

## Prediction

Generate predictions for test images and save a submission file:

```bash
python predict.py
```

Behavior:

- Loads the model checkpoint (default path in code: `models/resnet_best.pth`)
- Applies standard center-crop transforms
- Writes `submission.csv` with columns: `image_id,label`

If your checkpoint uses a different path or name, adjust the path in `predict.py`.

## Data Augmentation

Create an offline augmented dataset (and CSV) from your train set:

```bash
python augment.py
```

This will:

- Create multiple augmented variants per image
- Save them under `augmented_train/`
- Generate `augmented_train/augmented_data.csv`

You can then point your training script to the augmented directory/CSV if you integrate it into your pipeline.

## Advanced Techniques

### MixUp and CutMix

Utilities are provided in `mixup.py`:

- MixUp: blends pairs of samples and labels
- CutMix: cuts and pastes a random patch across samples

Typical usage (pseudocode inside your training loop):

```python
from mixup import mixup_data, mixup_criterion

inputs, targets = next(batch)
inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
outputs = model(inputs)
loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
```

### Sharpness-Aware Minimization (SAM)

SAM improves generalization by optimizing for flat minima. Example wiring:

```python
from sam import SAM
import torch.optim as optim

base_optimizer = optim.AdamW
optimizer = SAM(model.parameters(), base_optimizer, lr=0.0015, rho=0.05)

# You must use a closure that does forward+backward twice internally
```

Consult `sam.py` for the full API and required closure behavior.

### Model Ensembling

The repository includes a skeleton `ensemble.py` with training and prediction utilities. A typical flow is:

- Train several models with different architectures/hyperparameters
- Save each best checkpoint to `models/ensemble/`
- Use soft-voting or averaging logits for final predictions

Adjust and complete `ensemble.py` as needed for your use case.

## Results Visualization

Training generates plots under `models/`:

- Loss: `models/loss_plot.png`
- Accuracy: `models/accuracy_plot.png`

If you extend the pipeline or use ensembling, ensemble-specific plots may be saved to `models/ensemble/`.

## Performance Tips

- Use a GPU for training; start with smaller batch sizes if memory is tight
- Keep image size at 224 for speed, or increase if your GPU allows
- Enable mixed precision (AMP) if you integrate it to speed up training
- Monitor overfitting; increase augmentation strength or weight decay if needed
- Try ResNet34/50 by switching the `architecture` in `model.py`/your script

## Troubleshooting

- CUDA OOM: reduce `batch_size`, or lower input size/augmentations
- Checkpoint not found: verify path in `predict.py` (default `models/resnet_best.pth`)
- Slow dataloading: increase `num_workers` if your environment allows
- Class imbalance: consider weighted loss or oversampling

## Contributing

Contributions are welcome! Please open issues/PRs with clear descriptions and minimal reproducible examples.

## License

MIT License. See License file for more.

## Acknowledgements

- PyTorch and TorchVision teams
- Open-source community for research and tooling
