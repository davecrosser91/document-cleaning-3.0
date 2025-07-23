# Document Cleaning 3.0

This repo should help train a model to clean documents

## Overview

Document Cleaning 3.0 provides a modular pipeline for training deep-learning models that remove artifacts from scanned or photographed documents. The default baseline is a convolutional autoencoder, but the design allows you to drop-in more sophisticated architectures with minimal changes.

## Core Components

1. **PairedImageDatasetBuilder** – Converts two parallel folders (`raw/` and `clean/`) into a `datasets.Dataset` / `DatasetDict`, handling image loading, preprocessing, and train/val/test splits.
2. **DocumentAutoEncoder** – Simple convolutional autoencoder that maps a noisy input image to its cleaned counterpart.
3. **Trainer** – Orchestrates the training loop in pure PyTorch, taking a `datasets.Dataset` and any model that implements the `CleanModel` protocol.

## Quick Start

```bash
# Build the HF dataset
python -m document_cleaning_3.data.build_dataset \
  --raw-dir data/raw_images \
  --clean-dir data/clean_images \
  --output-dir data/hf_dataset

# Train the baseline autoencoder
python -m document_cleaning_3.train --config configs/autoencoder.yaml
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
