"""
Script to train the autoencoder model for document cleaning.

This script provides functionality to:
1. Load the paired PDF dataset
2. Initialize the autoencoder model
3. Train the model on the dataset
4. Save checkpoints and visualize results during training
5. Evaluate the model on a validation set
"""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union, Callable
import time
import argparse
import json
import io
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import wandb

# Use relative imports when running as a module
try:
    from src.dataset_maker.PairedPDFBuilder import PairedPDFBuilder
    from src.models.autoencoder import create_autoencoder, Autoencoder, AutoencoderConfig
    from src.models.swinir import create_swinir_model, SwinIR, SwinIRConfig
# Use direct imports when running the script directly
except ModuleNotFoundError:
    from dataset_maker.PairedPDFBuilder import PairedPDFBuilder
    from models.autoencoder import create_autoencoder, Autoencoder, AutoencoderConfig
    from models.swinir import create_swinir_model, SwinIR, SwinIRConfig

# Define a type alias for our model types
ModelType = Union[Autoencoder, SwinIR]


def load_dataset(data_root: Path, val_split: float = 0.1) -> Tuple[Dataset, Dataset]:
    """Load the paired PDF dataset and split into train/val.
    
    Args:
        data_root: Path to the dataset root directory
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
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
    
    # Split into train and validation sets
    dataset = dataset.shuffle(seed=42)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    
    return train_dataset, val_dataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List[Any]]]:
    """Robust collate function that handles both numpy arrays and lists.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Dictionary of batched tensors
    """
    # Handle various input types (numpy arrays or lists) robustly
    clean_images_list = []
    dirty_images_list = []
    
    for item in batch:
        # Handle clean images
        clean_img = item['clean_image']
        if isinstance(clean_img, np.ndarray):
            clean_images_list.append(torch.from_numpy(clean_img).float())
        elif isinstance(clean_img, list):
            clean_images_list.append(torch.tensor(clean_img).float())
        else:
            # Already a tensor or another type
            clean_images_list.append(torch.tensor(clean_img).float())
            
        # Handle dirty images
        dirty_img = item['dirty_image']
        if isinstance(dirty_img, np.ndarray):
            dirty_images_list.append(torch.from_numpy(dirty_img).float())
        elif isinstance(dirty_img, list):
            dirty_images_list.append(torch.tensor(dirty_img).float())
        else:
            # Already a tensor or another type
            dirty_images_list.append(torch.tensor(dirty_img).float())
    
    # Stack tensors
    clean_images = torch.stack(clean_images_list)
    dirty_images = torch.stack(dirty_images_list)
    
    # Resize images to target dimensions for training
    import torch.nn.functional as F
    
    # Target dimensions (half of original 2480x1754)
    target_H, target_W = 1240, 877
    
    # Get current dimensions and check if resizing is needed
    _, _, H, W = clean_images.shape
    if H != target_H or W != target_W:
        print(f"Resizing images from {H}x{W} to {target_H}x{target_W}")
        # Resize both clean and dirty images
        clean_images = F.interpolate(clean_images, size=(target_H, target_W), mode='bilinear', align_corners=False)
        dirty_images = F.interpolate(dirty_images, size=(target_H, target_W), mode='bilinear', align_corners=False)
    
    # Dimension check after resizing
    _, _, final_H, final_W = clean_images.shape
    assert final_H == target_H and final_W == target_W, f"Dimension check failed: expected {target_H}x{target_W}, got {final_H}x{final_W}"
    
    # Normalize images to [0, 1] if they aren't already
    # Only check the first image to avoid expensive max operation
    if clean_images[0].max() > 1.0:
        clean_images = clean_images / 255.0
    if dirty_images[0].max() > 1.0:
        dirty_images = dirty_images / 255.0
        
    # Pre-compute the dirt_type tensor
    dirt_types = torch.tensor([item['dirt_type'] for item in batch])
    
    return {
        'clean_image': clean_images,
        'dirty_image': dirty_images,
        'dirt_type': dirt_types,
        'clean_file': [item['clean_file'] for item in batch],
        'dirty_file': [item['dirty_file'] for item in batch]
    }


def prepare_dataloader(dataset: Dataset, batch_size: int = 4, shuffle: bool = True) -> DataLoader:
    """Prepare an optimized DataLoader for the dataset.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader for the dataset with optimized performance settings
    """
    # Use more worker processes based on available CPU cores
    num_workers = min(8, os.cpu_count() or 4)
    
    # Create the DataLoader with optimized settings
    dataloader = DataLoader(  # pyright: ignore[reportGeneralTypeIssues]
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,  # Increased worker count
        pin_memory=True,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    return dataloader


def initialize_model(
    input_height: int, 
    input_width: int,
    model_type: str = "autoencoder",
    autoencoder_config: Optional[AutoencoderConfig] = None,
    swinir_config: Optional[SwinIRConfig] = None,
    device: str = "cpu"
) -> ModelType:
    """Initialize the model.
    
    Args:
        input_height: Height of input images
        input_width: Width of input images
        model_type: Type of model to use ('autoencoder' or 'swinir')
        autoencoder_config: Optional custom configuration for the autoencoder
        swinir_config: Optional custom configuration for SwinIR
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        Initialized model
    """
    print(f"Initializing {model_type} model...")
    
    # Create model based on type
    if model_type.lower() == "autoencoder":
        model = create_autoencoder(
            input_height=input_height,
            input_width=input_width,
            config=autoencoder_config
        )
    elif model_type.lower() == "swinir":
        # Use 3-channel input to match our dataset
        # Use actual input dimensions instead of square size
        model = create_swinir_model(
            img_size=(input_height, input_width),  # Use actual rectangular dimensions
            in_chans=3,  # RGB images (3 channels)
            out_chans=3,  # RGB output (3 channels)
            window_size=8,  # Standard window size
            embed_dim=96,  # Embedding dimension
            depths=[6, 6, 6, 6],  # Default depths for moderate-sized model
            num_heads=[6, 6, 6, 6],  # Default number of heads
            use_checkpoint=True  # Use checkpointing to save memory
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'autoencoder' or 'swinir'")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Move to device
    model = model.to(device)
    
    return model


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy image.
    
    Args:
        tensor: Input tensor of shape [C, H, W]
        
    Returns:
        Numpy array of shape [H, W, C] with values in [0, 1]
    """
    # Convert to numpy and transpose from [C, H, W] to [H, W, C]
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Ensure values are in [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def visualize_batch(
    input_images: torch.Tensor,
    output_images: torch.Tensor,
    target_images: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[Path] = None
) -> None:
    """Visualize a batch of images (input, output, target).
    
    Args:
        input_images: Input (dirty) images tensor [B, C, H, W]
        output_images: Output images from the autoencoder [B, C, H, W]
        target_images: Target (clean) images tensor [B, C, H, W]
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
    """
    # Ensure save directory exists
    if save_path is not None:
        save_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Limit number of samples to batch size
    num_samples = min(num_samples, input_images.size(0))
    
    # Create a grid of images
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # If only one sample, wrap axes in a list to make it iterable
    if num_samples == 1:
        axes = [axes]
    
    # Plot each sample
    for i in range(num_samples):
        # Input image
        axes[i][0].imshow(tensor_to_image(input_images[i]))
        axes[i][0].set_title("Input (Dirty)")
        axes[i][0].axis('off')
        
        # Output image
        axes[i][1].imshow(tensor_to_image(output_images[i]))
        axes[i][1].set_title("Output (Cleaned)")
        axes[i][1].axis('off')
        
        # Target image
        axes[i][2].imshow(tensor_to_image(target_images[i]))
        axes[i][2].set_title("Target (Clean)")
        axes[i][2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to {save_path}")
    
    plt.close(fig)


def train_epoch(
    model: ModelType,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    log_interval: int = 10,
    log_to_wandb: bool = False
) -> float:
    """Train the model for one epoch using mixed precision training.
    
    Args:
        model: The autoencoder model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        log_interval: How often to log progress
        log_to_wandb: Whether to log metrics to wandb
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if device == 'cuda' else None
    use_amp = device == 'cuda'  # Only use mixed precision on CUDA devices
    
    # Use tqdm for progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
    for batch_idx, batch in pbar:
        # Get data
        dirty_images = batch['dirty_image'].to(device, non_blocking=True)  # Use non_blocking for async transfer
        clean_images = batch['clean_image'].to(device, non_blocking=True)
        
        # Dimension check: ensure every batch has the expected dimensions
        expected_height, expected_width = 1240, 877
        _, _, h, w = dirty_images.shape
        assert h == expected_height and w == expected_width, f"Training batch dimension mismatch: expected {expected_height}x{expected_width}, got {h}x{w}"
        
        _, _, h_clean, w_clean = clean_images.shape
        assert h_clean == expected_height and w_clean == expected_width, f"Training batch dimension mismatch (clean): expected {expected_height}x{expected_width}, got {h_clean}x{w_clean}"
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient than False
        
        # Mixed precision training path
        if use_amp:
            with autocast():
                # Forward pass
                outputs = model(dirty_images)
                
                # Calculate loss
                loss = criterion(outputs, clean_images)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Standard precision training path
        else:
            # Forward pass
            outputs = model(dirty_images)
            
            # Calculate loss
            loss = criterion(outputs, clean_images)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log batch loss to wandb
        if log_to_wandb:
            wandb.log({"batch_loss": loss.item(), "batch": batch_idx + (epoch-1) * len(train_loader)})
        
    # Calculate average loss
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss


def validate(
    model: ModelType,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    visualize_path: Optional[Path] = None,
    log_to_wandb: bool = False,
    epoch: Optional[int] = None
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the model on the validation set.
    
    Args:
        model: The autoencoder model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on
        visualize_path: Optional path to save visualizations
        
    Returns:
        Tuple of (average validation loss, sample input images, 
                 sample output images, sample target images)
    """
    model.eval()
    total_loss = 0.0
    
    # Store a batch for visualization
    sample_input = None
    sample_output = None
    sample_target = None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Get data
            dirty_images = batch['dirty_image'].to(device)
            clean_images = batch['clean_image'].to(device)
            
            # Dimension check: ensure every validation batch has the expected dimensions
            expected_height, expected_width = 1240, 877
            _, _, h, w = dirty_images.shape
            assert h == expected_height and w == expected_width, f"Validation batch dimension mismatch: expected {expected_height}x{expected_width}, got {h}x{w}"
            
            _, _, h_clean, w_clean = clean_images.shape
            assert h_clean == expected_height and w_clean == expected_width, f"Validation batch dimension mismatch (clean): expected {expected_height}x{expected_width}, got {h_clean}x{w_clean}"
            
            # Forward pass
            outputs = model(dirty_images)
            
            # Calculate loss
            loss = criterion(outputs, clean_images)
            
            # Update total loss
            total_loss += loss.item()
            
            # Store first batch for visualization
            if batch_idx == 0:
                sample_input = dirty_images
                sample_output = outputs
                sample_target = clean_images
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    
    # Visualize results if path is provided
    #if visualize_path is not None and sample_input is not None:
     #   visualize_batch(
      #      sample_input.cpu(),
       #     sample_output.cpu(),
        #    sample_target.cpu(),
         #   save_path=visualize_path
        #)
        
    # Log to wandb if enabled
    if log_to_wandb and sample_input is not None and epoch is not None:
        # Create a figure for wandb
        fig = plt.figure(figsize=(15, 5))
        
        # Plot input image
        plt.subplot(1, 3, 1)
        plt.imshow(tensor_to_image(sample_input[0].cpu()))
        plt.title("Input (Dirty)")
        plt.axis('off')
        
        # Plot output image
        plt.subplot(1, 3, 2)
        plt.imshow(tensor_to_image(sample_output[0].cpu()))
        plt.title("Output (Cleaned)")
        plt.axis('off')
        
        # Plot target image
        plt.subplot(1, 3, 3)
        plt.imshow(tensor_to_image(sample_target[0].cpu()))
        plt.title("Target (Clean)")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"validation_images_epoch_{epoch}": wandb.Image(fig)})
        plt.close(fig)
    
    return avg_loss, sample_input, sample_output, sample_target


def save_checkpoint(
    model: ModelType,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    is_best: bool = False
) -> None:
    """Save a model checkpoint.
    
    Args:
        model: The autoencoder model
        optimizer: The optimizer
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best model if this is the best
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(
    model: ModelType,
    optimizer: Optional[optim.Optimizer],
    checkpoint_path: Path
) -> Tuple[ModelType, Optional[optim.Optimizer], int, float]:
    """Load a model checkpoint.
    
    Args:
        model: The autoencoder model
        optimizer: The optimizer (can be None for inference only)
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return model, optimizer, epoch, and loss
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']


def train_model(
    model: ModelType,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    num_epochs: int,
    checkpoint_dir: Path,
    log_dir: Path,
    log_interval: int = 1,
    early_stopping_patience: int = 10,
    log_to_wandb: bool = False,
    experiment_name: str = "autoencoder"
) -> ModelType:
    """Train the model.
    
    Args:
        model: The model (autoencoder or SwinIR)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        num_epochs: Number of epochs to train for
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        log_interval: How often to log (in epochs)
        early_stopping_patience: Early stopping patience
        log_to_wandb: Whether to log to Weights & Biases
        experiment_name: Name of the experiment
        
    Returns:
        Trained model
    """
    # Create directories
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize variables for tracking progress
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Initialize wandb if enabled
    if log_to_wandb:
        wandb.init(
            project="document-cleaning-autoencoder",
            name=experiment_name,
            config={
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "early_stopping_patience": early_stopping_patience,
                "device": device,
                "model_type": model.__class__.__name__,
            }
        )
        # Log model architecture
        wandb.watch(model, log="all")
    
    # Start training
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            log_to_wandb=log_to_wandb
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, sample_input, sample_output, sample_target = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            visualize_path=log_dir / f"val_epoch_{epoch}.png",
            log_to_wandb=log_to_wandb,
            epoch=epoch
        )
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Log metrics to wandb
        if log_to_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss if val_loss > best_val_loss else val_loss,
                "patience_counter": patience_counter
            })
        
        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # # Save checkpoint
        # if is_best:
        #     save_checkpoint(
        #     model=model,
        #     optimizer=optimizer,
        #     epoch=epoch,
        #     loss=val_loss,
        #     checkpoint_dir=checkpoint_dir,
        #     is_best=is_best
        # )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs': list(range(1, epoch + 1))
        }
        with open(log_dir / "training_history.json", 'w') as f:
            json.dump(history, f)
        
        # # Plot losses
        # plt.figure(figsize=(10, 5))
        # plt.plot(history['epochs'], history['train_losses'], label='Train Loss')
        # plt.plot(history['epochs'], history['val_losses'], label='Val Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training and Validation Loss')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(log_dir / "loss_plot.png")
        # plt.close()
    
    # Calculate training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    
    # Load best model
    best_model_path = checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        model, _, _, _ = load_checkpoint(model, None, best_model_path)
        print(f"Loaded best model from {best_model_path}")
        
    # Finish wandb run
    if log_to_wandb:
        # Log the best model as an artifact
        if best_model_path.exists():
            artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
            artifact.add_file(str(best_model_path))
            wandb.log_artifact(artifact)
        
        # Close wandb run
        wandb.finish()
    
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an autoencoder for document cleaning")
    
    # Dataset arguments
    parser.add_argument("--data-root", type=str, default="src/dataset_maker/data",
                        help="Path to the dataset root directory")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Patience for early stopping")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="autoencoder", choices=["autoencoder", "swinir"],
                        help="Type of model to use (autoencoder or swinir)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="Hidden dimensions for each layer (autoencoder only)")
    parser.add_argument("--latent-dim", type=int, default=256,
                        help="Dimension of the latent space (autoencoder only)")
    parser.add_argument("--window-size", type=int, default=8,
                        help="Window size for SwinIR (swinir only)")
    parser.add_argument("--embed-dim", type=int, default=96, 
                        help="Embedding dimension for SwinIR (swinir only)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name of the experiment (default: timestamp)")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="How often to log and validate (in epochs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Weights & Biases arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="document-cleaning",
                        help="Weights & Biases project name")
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directories
    output_dir = Path(args.output_dir)
    experiment_dir = output_dir / args.experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"
    
    # Load dataset
    data_root = Path(args.data_root)
    train_dataset, val_dataset = load_dataset(data_root, args.val_split)
    
    # Prepare DataLoaders
    train_loader = prepare_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = prepare_dataloader(val_dataset, args.batch_size, shuffle=False)
    
    # Get a batch to determine image dimensions
    print("Loading a batch from the dataset...")
    batch = next(iter(train_loader))
    
    # Extract images and get dimensions (after scaling in collate_fn)
    dirty_images = batch['dirty_image']
    _, _, height, width = dirty_images.shape
    
    # Dimension check: ensure we have the expected target dimensions
    expected_height, expected_width = 1240, 877
    assert height == expected_height and width == expected_width, f"Dimension mismatch: expected {expected_height}x{expected_width}, got {height}x{width}"
    
    print(f"✓ Dimension check passed: {height}x{width} (target resolution for training)")
    
    # Initialize model
    if args.model_type == "autoencoder":
        # Create autoencoder config if using autoencoder
        autoencoder_config = AutoencoderConfig(
            input_channels=1,  # Using grayscale
            output_channels=1,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim
        )
        model = initialize_model(
            input_height=height,
            input_width=width,
            model_type=args.model_type,
            autoencoder_config=autoencoder_config,
            device=device
        )
    else:  # swinir
        # Use the command line arguments for SwinIR
        model = initialize_model(
            input_height=height,
            input_width=width,
            model_type=args.model_type,
            swinir_config=None,  # Using the defaults in initialize_model
            device=device
        )
        
    print(f"Using {args.model_type} model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, resume_path)
            print(f"Resumed from checkpoint: {resume_path}")
        else:
            print(f"Checkpoint not found: {resume_path}")
    
    # Train model
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        log_interval=args.log_interval,
        early_stopping_patience=args.early_stopping,
        log_to_wandb=args.wandb,
        experiment_name=args.experiment_name
    )
    
    print(f"Training completed. Results saved to {experiment_dir}")


if __name__ == "__main__":
    main()
