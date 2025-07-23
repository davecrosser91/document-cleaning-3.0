"""
Script to load the dataset and perform a forward pass through the autoencoder.

This script demonstrates how to:
1. Load the paired PDF dataset
2. Initialize the autoencoder model
3. Process a batch of images through the model
4. Visualize the results
"""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict

# Use relative imports when running as a module
try:
    from src.dataset_maker.PairedPDFBuilder import PairedPDFBuilder
    from src.models.autoencoder import create_autoencoder, Autoencoder
# Use direct imports when running the script directly
except ModuleNotFoundError:
    from dataset_maker.PairedPDFBuilder import PairedPDFBuilder
    from models.autoencoder import create_autoencoder, Autoencoder


def load_dataset(data_root: Path) -> Dataset:
    """Load the paired PDF dataset.
    
    Args:
        data_root: Path to the dataset root directory
        
    Returns:
        Loaded dataset
    """
    # Initialize the builder with the data root
    builder = PairedPDFBuilder(name="default", data_root=data_root)
    
    # Prepare the dataset (skip if already prepared)
    try:
        # Try to load the dataset without rebuilding
        dataset = builder.as_dataset(split="train")
        print(f"Dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Rebuilding dataset...")
        builder.download_and_prepare(download_mode="force_redownload")
        dataset = builder.as_dataset(split="train")
        print(f"Dataset rebuilt and loaded: {len(dataset)} examples")
    
    return dataset


def prepare_dataloader(dataset: Union[Dataset, DatasetDict], batch_size: int = 4, split: str = "train") -> DataLoader:
    """Prepare a DataLoader for the dataset.
    
    Args:
        dataset: The dataset to load, can be either Dataset or DatasetDict
        batch_size: Batch size for the DataLoader
        split: Dataset split to use when dataset is a DatasetDict (default: "train")
        
    Returns:
        DataLoader for the dataset
    """
    # Create a custom collate function to handle the conversion to tensors
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List[Any]]]:
        clean_images = torch.tensor(np.stack([item['clean_image'] for item in batch])).float()
        dirty_images = torch.tensor(np.stack([item['dirty_image'] for item in batch])).float()
        
        # Normalize images to [0, 1] if they aren't already
        if clean_images.max() > 1.0:
            clean_images = clean_images / 255.0
        if dirty_images.max() > 1.0:
            dirty_images = dirty_images / 255.0
            
        return {
            'clean_image': clean_images,
            'dirty_image': dirty_images,
            'dirt_type': torch.tensor([item['dirt_type'] for item in batch]),
            'clean_file': [item['clean_file'] for item in batch],
            'dirty_file': [item['dirty_file'] for item in batch]
        }
    
    # Handle DatasetDict by selecting the appropriate split
    if isinstance(dataset, DatasetDict):
        actual_dataset = dataset[split]
    else:
        actual_dataset = dataset
        
    # Create the DataLoader
    dataloader = DataLoader(  # pyright: ignore[reportGeneralTypeIssues]
        actual_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def initialize_model(
    input_height: int, 
    input_width: int,
    device: str = "cpu"
) -> Autoencoder:
    """Initialize the autoencoder model.
    
    Args:
        input_height: Height of input images
        input_width: Width of input images
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        Initialized autoencoder model
    """
    # Create the autoencoder with default parameters
    model = create_autoencoder(
        input_channels=3,
        output_channels=3,
        hidden_dims=[16, 32, 64],
        latent_dim=128,
        input_height=input_height,
        input_width=input_width
    )
    
    # Move model to the specified device
    model = model.to(device)
    
    return model


def resize_images(
    images: torch.Tensor, 
    target_height: int = 64, 
    target_width: int = 64
) -> torch.Tensor:
    """Resize images to the target dimensions.
    
    Args:
        images: Input images tensor [B, C, H, W]
        target_height: Target height
        target_width: Target width
        
    Returns:
        Resized images tensor [B, C, target_height, target_width]
    """
    # Use torch's interpolate function for resizing
    return torch.nn.functional.interpolate(
        images,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )


def visualize_results(
    input_images: torch.Tensor,
    output_images: torch.Tensor,
    target_images: torch.Tensor,
    num_samples: int = 1,
    save_path: Optional[Path] = None
) -> None:
    """Visualize the input, output, and target images.
    
    Args:
        input_images: Input (dirty) images tensor [B, C, H, W]
        output_images: Output images from the autoencoder [B, C, H, W]
        target_images: Target (clean) images tensor [B, C, H, W]
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
    """
    # Ensure we don't try to visualize more samples than we have
    num_samples = min(num_samples, input_images.shape[0])
    
    # Create a figure with 3 rows (input, output, target) and num_samples columns
    fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
    
    # Helper function to convert tensor to numpy image
    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        # Move to CPU if on GPU
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        # Convert to numpy and transpose from [C, H, W] to [H, W, C]
        image = tensor.detach().numpy().transpose(1, 2, 0)
        
        # Clip values to [0, 1] range
        image = np.clip(image, 0, 1)
        
        return image
    # Plot each sample in a separate figure with high quality
    for i in range(num_samples):
        # Input (dirty) image
        plt.figure(figsize=(15, 5), dpi=150)
        plt.imshow(tensor_to_image(input_images[i]))
        plt.title("Input (Dirty)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        # Output (cleaned by model) image
        plt.figure(figsize=(15, 5), dpi=150)
        plt.imshow(tensor_to_image(output_images[i]))
        plt.title("Output (Cleaned)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        # Target (clean) image
        plt.figure(figsize=(15, 5), dpi=150)
        plt.imshow(tensor_to_image(target_images[i]))
        plt.title("Target (Clean)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figures if a path is provided
        if save_path is not None:
            plt.figure(figsize=(15, 5), dpi=150)
            plt.imshow(tensor_to_image(input_images[i]))
            plt.title("Input (Dirty)", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path.parent / f"input_{i}.png", dpi=300)
            
            plt.figure(figsize=(15, 5), dpi=150)
            plt.imshow(tensor_to_image(output_images[i]))
            plt.title("Output (Cleaned)", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path.parent / f"output_{i}.png", dpi=300)
            
            plt.figure(figsize=(15, 5), dpi=150)
            plt.imshow(tensor_to_image(target_images[i]))
            plt.title("Target (Clean)", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path.parent / f"target_{i}.png", dpi=300)
            
        print(f"Processed sample {i+1}/{num_samples}")


def run_forward_pass() -> None:
    """Run a forward pass through the autoencoder using the dataset."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set paths
    data_root = Path("src/dataset_maker/data")
    
    # Load dataset
    dataset = load_dataset(data_root)
    
    # Prepare DataLoader
    dataloader = prepare_dataloader(dataset, batch_size=4)
    
    # Get a batch to determine image dimensions
    print("Loading a batch from the dataset...")
    batch = next(iter(dataloader))
    
    # Extract images
    dirty_images = batch['dirty_image'].to(device)
    clean_images = batch['clean_image'].to(device)
    
    # Get actual image dimensions
    _, _, height, width = dirty_images.shape
    print(f"Using actual image dimensions: {height}x{width}")
    
    # Initialize model with actual image dimensions
    model = initialize_model(
        input_height=height,
        input_width=width,
        device=device
    )
    
    # Print shapes
    print(f"Dirty images shape: {dirty_images.shape}")
    print(f"Clean images shape: {clean_images.shape}")
    
    # Forward pass
    print("Running forward pass through the autoencoder...")
    with torch.no_grad():
        outputs = model(dirty_images)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Input shape matches output shape: {outputs.shape == dirty_images.shape}")
    
    # Verify dimensions match
    assert outputs.shape == dirty_images.shape, "Output shape must match input shape exactly"
    
    # Calculate loss
    criterion = nn.MSELoss()
    loss = criterion(outputs, clean_images)
    print(f"MSE Loss: {loss.item():.6f}")
    
    # Visualize results
    print("Visualizing results...")
    
    # For visualization, we need to resize to a reasonable size
    #vis_height, vis_width = 256, 256
    #dirty_images_resized = resize_images(dirty_images, vis_height, vis_width)
    #clean_images_resized = resize_images(clean_images, vis_height, vis_width)
    #outputs_resized = resize_images(outputs, vis_height, vis_width)
    
    visualize_results(
        dirty_images.cpu(),
        outputs.cpu(),
        clean_images.cpu(),
        save_path=Path("results/autoencoder_results.png")
    )
    
    print("Forward pass completed successfully!")


if __name__ == "__main__":
    run_forward_pass()
