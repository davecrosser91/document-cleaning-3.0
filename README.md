# Document Cleaning 3.0

A modular pipeline for training deep-learning models that remove artifacts from scanned or photographed documents.

## Overview

Document Cleaning 3.0 provides a modular pipeline for training deep-learning models that remove artifacts from scanned or photographed documents. The default baseline is a convolutional autoencoder, but the design allows you to drop-in more sophisticated architectures with minimal changes.

## Core Components

1. **PairedImageDatasetBuilder** – Converts two parallel folders (`raw/` and `clean/`) into a `datasets.Dataset` / `DatasetDict`, handling image loading, preprocessing, and train/val/test splits.
2. **DocumentAutoEncoder** – Simple convolutional autoencoder that maps a noisy input image to its cleaned counterpart.
3. **Trainer** – Orchestrates the training loop in pure PyTorch, taking a `datasets.Dataset` and any model that implements the `CleanModel` protocol.

## Quick Start

### Dataset Preparation

```bash
# Build the HF dataset
python -m document_cleaning_3.data.build_dataset \
  --raw-dir data/raw_images \
  --clean-dir data/clean_images \
  --output-dir data/hf_dataset
```

### Local Training

Train the autoencoder model locally:

```bash
# Train with default settings
python src/train_autoencoder.py

# Train with custom parameters
python src/train_autoencoder.py \
  --data-root src/dataset_maker/data \
  --batch-size 8 \
  --num-epochs 100 \
  --lr 0.0005 \
  --hidden-dims 32 64 128 256 \
  --latent-dim 512 \
  --output-dir results/my_experiment

# Train with Weights & Biases monitoring
python src/train_autoencoder.py --wandb
```

### Remote Training on RunPod

For training on powerful GPUs, we provide integration with RunPod:

#### Setup

1. **Install RunPod CLI**:

```bash
# Download the RunPod CLI binary for macOS
curl -fsSL https://github.com/runpod/runpodctl/releases/download/v1.10.0/runpodctl-darwin-amd64 -o runpodctl

# Make it executable
chmod +x runpodctl

# Move to a directory in your PATH
mv runpodctl /usr/local/bin/

# For Linux, use:
# curl -fsSL https://github.com/runpod/runpodctl/releases/download/v1.10.0/runpodctl-linux-amd64 -o runpodctl

# For Windows, download from:
# https://github.com/runpod/runpodctl/releases
```

1. **Configure API Key**:

```bash
runpodctl config set api-key YOUR_API_KEY
# Or set as environment variable
export RUNPOD_API_KEY=your_api_key_here
```

#### Training Workflow

1. **Create a RunPod GPU instance**:

```bash
python src/runpod_train.py --create --name "document-cleaner-training"
```

1. **Upload your code**:

```bash
python src/runpod_train.py --pod-id YOUR_POD_ID --upload .
```

1. **Install dependencies**:

```bash
python src/runpod_train.py --pod-id YOUR_POD_ID --install
```

1. **Start training**:

```bash
# Basic training
python src/runpod_train.py --pod-id YOUR_POD_ID --train

# With Weights & Biases monitoring
python src/runpod_train.py --pod-id YOUR_POD_ID --train --wandb --wandb-key YOUR_WANDB_API_KEY

# With custom parameters
python src/runpod_train.py --pod-id YOUR_POD_ID --train -- \
  --batch-size 16 --num-epochs 100 --lr 0.0003
```

1. **Download results**:

```bash
runpodctl cp YOUR_POD_ID:/workspace/results ./local_results
```

1. **Stop or terminate the pod**:

```bash
# Stop the pod (keeps data, stops billing)
python src/runpod_train.py --pod-id YOUR_POD_ID --stop

# Terminate the pod (deletes pod completely)
python src/runpod_train.py --pod-id YOUR_POD_ID --terminate
```

## Development Guidelines

* **Python ≥ 3.13**
* **Type hints are mandatory** for all public functions and classes.
* **Configuration objects** (dataset, model, training) **must inherit from `pydantic.BaseModel`** for validation and IDE support.
* **Frameworks**
  * Models: `torch`, `torchvision`
  * Data pipeline: `datasets`
  * Optional advanced models: `transformers`
* **Code style**
  * Follow PEP 8 and the project’s `ruff` configuration: `ruff format . && ruff check . --fix`
  * Validate types with `pyright`.
* **Testing**
  * Add unit tests with `pytest` for all new features.
* **Conventional commits** for commit messages.
