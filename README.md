# DDROP: Dynamic Dropout Pruning Framework

## Overview

DDROP is a dynamic dropout–based pruning framework for inducing structured sparsity in deep neural networks. It adaptively drops neurons or filters during training, determining drop probabilities based on activation statistics and augmented by L1 regularization. DDROP consistently improves the accuracy of pruned models on CIFAR-10, CIFAR-100, and ILSVRC2012 across various architectures (ResNet, VGG, ViT, Swin).

## Requirements

* Python 3.8 or later
* PyTorch 1.10+
* torchvision
* timm (for ViT and Swin)
* numpy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/aftoul/ddrop.git
   cd ddrop
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Train a model with DDROP pruning:

```bash
python main.py \
  --dataset cifar10 \
  --data-dir /path/to/data \
  --model resnet50 \
  --prob 0.0 0.95 \
  --schedule cosine \
  --total_steps 10000 \
  --batch-size 128 \
  --epochs 200 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --before-epochs 100 \
  --amount 0.5 \
  --ft-epochs 50 \
  --ft-lr 0.01 \
  --seed 42
```

### Arguments

| Argument            | Description                                                               | Default    |
| ------------------- | ------------------------------------------------------------------------- | ---------- |
| `--dataset`         | Dataset to use: `cifar10`, `cifar100`, or `imagenet`                      | *required* |
| `--data-dir`        | Path to dataset root directory                                            | *required* |
| `--model`           | Model architecture (`resnet18`, `resnet34`, `resnet50`, `vgg11_bn`, etc.) | *required* |
| `--prob`            | Dropout probability range (min, max)                                      | *required* |
| `--schedule`        | Dropout schedule: `constant`, `cosine`, `linear`                          | constant   |
| `--total_steps`     | Number of steps for schedule progression                                  | 1/3 epochs |
| `--batch-size`      | Batch size for training                                                   | 128        |
| `--test-batch-size` | Batch size for validation/testing                                         | 100        |
| `--epochs`          | Total training epochs                                                     | 200        |
| `--lr`              | Initial learning rate                                                     | 0.1        |
| `--weight-decay`    | Weight decay for optimizer                                                | 1e-4       |
| `--before-epochs`   | Epochs before pruning starts                                              | 100        |
| `--amount`          | Pruning amount (fraction of weights to remove)                            | 0.5        |
| `--ft-epochs`       | Fine-tuning epochs after pruning                                          | 50         |
| `--ft-lr`           | Learning rate for fine-tuning                                             | 0.01       |
| `--seed`            | Random seed for reproducibility                                           | 42         |

## Example

```bash
python main.py --dataset cifar100 --data-dir ~/data --model vgg16_bn --prob 0.1 0.9 \
    --schedule linear --total_steps 5000 --batch-size 64 --epochs 150 --lr 0.01 \
    --weight-decay 5e-4 --before-epochs 75 --amount 0.4 --ft-epochs 30 --ft-lr 0.005 --seed 123
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

