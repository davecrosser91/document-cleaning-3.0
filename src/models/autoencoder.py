"""
Autoencoder model for document cleaning.

This module defines a simple autoencoder architecture that can be used
to clean document images by learning to map from dirty to clean images.
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, cast, Any, Callable, Union, List

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class EncoderConfig(BaseModel):
    """Configuration for the encoder part of the autoencoder."""
    input_channels: int = Field(3, description="Number of input channels (RGB=3)")
    hidden_dims: List[int] = Field([16, 32, 64], description="Hidden dimensions for each layer")
    latent_dim: int = Field(128, description="Dimension of the latent space")
    
    class Config:
        extra = "forbid"


class DecoderConfig(BaseModel):
    """Configuration for the decoder part of the autoencoder."""
    output_channels: int = Field(3, description="Number of output channels (RGB=3)")
    hidden_dims: List[int] = Field([64, 32, 16], description="Hidden dimensions for each layer")
    latent_dim: int = Field(128, description="Dimension of the latent space")
    
    class Config:
        extra = "forbid"


class AutoencoderConfig(BaseModel):
    """Configuration for the complete autoencoder model."""
    encoder: EncoderConfig
    decoder: DecoderConfig
    
    class Config:
        extra = "forbid"
    
    def __init__(self, **data):
        if "encoder" not in data:
            data["encoder"] = EncoderConfig()
        if "decoder" not in data:
            data["decoder"] = DecoderConfig()
        super().__init__(**data)


class Encoder(nn.Module):
    """Encoder network for the autoencoder."""
    
    def __init__(self, config: EncoderConfig) -> None:
        """Initialize the encoder.
        
        Args:
            config: Configuration for the encoder
        """
        super().__init__()
        
        self.config = config
        layers: List[nn.Module] = []
        
        # Input layer
        in_channels = config.input_channels
        
        # Build encoder layers
        for h_dim in config.hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Calculate the flattened feature size
        # This will be determined dynamically during forward pass
        self.feature_size: Optional[int] = None
        
        # Final convolutional layer to reduce spatial dimensions
        self.final_conv = nn.Conv2d(in_channels, config.latent_dim, kernel_size=1)
        
        # Global average pooling to handle variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Encoded representation in latent space
        """
        # Pass through convolutional layers
        x = self.encoder(x)
        
        # Apply final convolution to get latent channels
        x = self.final_conv(x)
        
        # Apply global average pooling to handle variable spatial dimensions
        x = self.global_pool(x)
        
        # Flatten to [B, latent_dim]
        z = x.view(x.size(0), -1)
        
        return z


class Decoder(nn.Module):
    """Decoder network for the autoencoder."""
    
    def __init__(self, config: DecoderConfig, input_height: int, input_width: int) -> None:
        """Initialize the decoder.
        
        Args:
            config: Configuration for the decoder
            input_height: Height of the original input image
            input_width: Width of the original input image
        """
        super().__init__()
        
        self.config = config
        self.input_height = input_height
        self.input_width = input_width
        
        # Calculate the size after encoding
        # For 3 layers with stride 2, the size is reduced by factor 2^3 = 8
        self.encoded_height = input_height // 8
        self.encoded_width = input_width // 8
        
        # Initial convolution from latent space
        self.initial_conv = nn.ConvTranspose2d(
            config.latent_dim, 
            config.hidden_dims[0], 
            kernel_size=4, 
            stride=1, 
            padding=0
        )
        
        # Build decoder layers
        modules: List[nn.Module] = []
        
        # Reverse the hidden dimensions for the decoder
        hidden_dims = config.hidden_dims.copy()
        
        # Build transposed convolution layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final layer to output channels
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    config.output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.Sigmoid()  # Scale to [0, 1] range
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            z: Latent representation tensor
            
        Returns:
            Reconstructed image tensor
        """
        # Reshape latent vector to [B, latent_dim, 1, 1]
        batch_size = z.size(0)
        x = z.view(batch_size, -1, 1, 1)
        
        # Initial convolution to start upsampling
        x = self.initial_conv(x)
        
        # Upsample to match original dimensions through transposed convolutions
        # This will handle variable input sizes
        x = nn.functional.interpolate(
            x, 
            size=(self.encoded_height, self.encoded_width),
            mode='bilinear', 
            align_corners=False
        )
        
        # Pass through transposed convolutions
        x = self.decoder(x)
        
        return x


class Autoencoder(nn.Module):
    """Complete autoencoder model combining encoder and decoder."""
    
    def __init__(self, config: AutoencoderConfig, input_height: int, input_width: int) -> None:
        """Initialize the autoencoder.
        
        Args:
            config: Configuration for the autoencoder
            input_height: Height of the input images
            input_width: Width of the input images
        """
        super().__init__()
        
        self.config = config
        self.input_height = input_height
        self.input_width = input_width
        
        # Create encoder and decoder
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(
            config.decoder, 
            input_height=input_height, 
            input_width=input_width
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Reconstructed tensor of the same shape
        """
        # Store original dimensions
        _, _, orig_h, orig_w = x.shape
        
        # Encode
        z = self.encoder(x)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Ensure output dimensions match input dimensions exactly
        if x_recon.shape[2:] != x.shape[2:]:
            x_recon = nn.functional.interpolate(
                x_recon,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )
        
        return x_recon
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to image.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed image
        """
        return self.decoder(z)


def create_autoencoder(
    input_channels: int = 3,
    output_channels: int = 3,
    hidden_dims: List[int] = [16, 32, 64],
    latent_dim: int = 128,
    input_height: int = 64,
    input_width: int = 64
) -> Autoencoder:
    """Create an autoencoder model with the given parameters.
    
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        hidden_dims: List of hidden dimensions for the encoder (reversed for decoder)
        latent_dim: Dimension of the latent space
        input_height: Height of the input images
        input_width: Width of the input images
        
    Returns:
        Configured autoencoder model
    """
    # Create encoder config
    encoder_config = EncoderConfig(
        input_channels=input_channels,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    )
    
    # Create decoder config with reversed hidden dims
    decoder_config = DecoderConfig(
        output_channels=output_channels,
        hidden_dims=hidden_dims[::-1],  # Reverse the hidden dims
        latent_dim=latent_dim
    )
    
    # Create autoencoder config
    autoencoder_config = AutoencoderConfig(
        encoder=encoder_config,
        decoder=decoder_config
    )
    
    # Create and return the autoencoder
    return Autoencoder(
        config=autoencoder_config,
        input_height=input_height,
        input_width=input_width
    )
