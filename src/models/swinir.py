"""
SwinIR model for document cleaning.

This module implements the SwinIR architecture based on Swin Transformer,
which is effective for image restoration tasks like document cleaning.

Reference:
SwinIR: Image Restoration Using Swin Transformer (ICCV 2021)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from pydantic import BaseModel, Field
from einops import rearrange


class SwinIRConfig(BaseModel):
    """Configuration for the SwinIR model."""
    img_size: Union[int, Tuple[int, int]] = Field(128, description="Input image size (int for square, tuple for rectangular)")
    patch_size: int = Field(1, description="Patch size")
    in_chans: int = Field(3, description="Number of input channels (RGB=3)")
    out_chans: int = Field(3, description="Number of output channels (RGB=3)")
    embed_dim: int = Field(96, description="Embedding dimension")
    depths: List[int] = Field([6, 6, 6, 6], description="Depths of each Swin layer")
    num_heads: List[int] = Field([6, 6, 6, 6], description="Number of heads in each layer")
    window_size: int = Field(7, description="Window size")
    mlp_ratio: float = Field(4.0, description="MLP ratio")
    qkv_bias: bool = Field(True, description="Add bias to query, key, value projections")
    drop_rate: float = Field(0.0, description="Dropout rate")
    attn_drop_rate: float = Field(0.0, description="Attention dropout rate")
    drop_path_rate: float = Field(0.1, description="Drop path rate")
    ape: bool = Field(False, description="Absolute position embedding")
    patch_norm: bool = Field(True, description="Normalize patches")
    upscale: int = Field(1, description="Upscaling factor (1 for same size)")
    use_checkpoint: bool = Field(False, description="Use checkpointing to save memory")
    resi_connection: str = Field("1conv", description="Residual connection type ('1conv' or '3conv')")
    
    @property
    def img_height(self) -> int:
        """Get image height."""
        return self.img_size[0] if isinstance(self.img_size, tuple) else self.img_size
    
    @property
    def img_width(self) -> int:
        """Get image width."""
        return self.img_size[1] if isinstance(self.img_size, tuple) else self.img_size
    
    class Config:
        extra = "forbid"


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Window partition function.
    
    Args:
        x: Input tensor of shape (B, H, W, C)
        window_size: Window size
        
    Returns:
        Windows tensor of shape (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # Pad input to make it divisible by window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, H, W, _ = x.shape
    
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Window reverse function.
    
    Args:
        windows: Windows tensor of shape (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of the original tensor (before padding)
        W: Width of the original tensor (before padding)
        
    Returns:
        Original tensor of shape (B, H, W, C)
    """
    # Calculate padded dimensions
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    H_pad = H + pad_h
    W_pad = W + pad_w
    
    B = int(windows.shape[0] / (H_pad * W_pad / window_size / window_size))
    x = windows.reshape(B, H_pad // window_size, W_pad // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H_pad, W_pad, -1)
    
    # Remove padding to get back to original dimensions
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H, :W, :]
    
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self-attention module."""
    
    def __init__(
        self, 
        dim: int, 
        window_size: Tuple[int, int], 
        num_heads: int, 
        qkv_bias: bool = True, 
        attn_drop: float = 0.0, 
        proj_drop: float = 0.0
    ) -> None:
        """Initialize the window attention module.
        
        Args:
            dim: Input dimension
            window_size: Window size
            num_heads: Number of attention heads
            qkv_bias: Whether to add bias to query, key, value projections
            attn_drop: Attention dropout rate
            proj_drop: Output dropout rate
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (num_windows*B, N, C)
            mask: Attention mask (optional)
            
        Returns:
            Output tensor of shape (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP module."""
    
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        act_layer: nn.Module = nn.GELU, 
        drop: float = 0.0
    ) -> None:
        """Initialize the MLP module.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden feature dimension
            out_features: Output feature dimension
            act_layer: Activation layer
            drop: Dropout rate
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping the path
        training: Whether in training mode
        
    Returns:
        Output tensor
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize DropPath.
        
        Args:
            drop_prob: Probability of dropping the path
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return drop_path(x, self.drop_prob, self.training)


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    
    def __init__(
        self, 
        dim: int, 
        input_resolution: Tuple[int, int], 
        num_heads: int, 
        window_size: int = 7, 
        shift_size: int = 0, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = True, 
        drop: float = 0.0, 
        attn_drop: float = 0.0, 
        drop_path: float = 0.0, 
        act_layer: nn.Module = nn.GELU, 
        norm_layer: nn.Module = nn.LayerNorm
    ) -> None:
        """Initialize Swin Transformer Block.
        
        Args:
            dim: Number of input channels
            input_resolution: Input resolution
            num_heads: Number of attention heads
            window_size: Window size
            shift_size: Shift size for shifted window attention
            mlp_ratio: MLP ratio
            qkv_bias: Whether to add bias to query, key, value projections
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Drop path rate
            act_layer: Activation layer
            norm_layer: Normalization layer
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim, 
            window_size=(self.window_size, self.window_size), 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor with shape (B, H*W, C)
            
        Returns:
            Output tensor with shape (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L} != {H}*{W}"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(
        self, 
        img_size: Union[int, Tuple[int, int]] = 224, 
        patch_size: int = 4, 
        in_chans: int = 3, 
        embed_dim: int = 96, 
        norm_layer: Optional[nn.Module] = None
    ) -> None:
        """Initialize the patch embedding layer.
        
        Args:
            img_size: Image size (int for square, tuple for rectangular)
            patch_size: Patch token size
            in_chans: Number of input channels
            embed_dim: Number of linear projection output channels
            norm_layer: Normalization layer
        """
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], 
            img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        
        Args:
            x: Input tensor with shape (B, C, H, W)
            
        Returns:
            Embedded tokens with shape (B, L, D)
        """
        B, C, H, W = x.shape
        # Removed size constraint assertion to allow rectangular images
        # The projection will handle arbitrary H,W dimensions
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Image to Patch Unembedding."""
    
    def __init__(
        self, 
        img_size: Union[int, Tuple[int, int]] = 224, 
        patch_size: int = 4, 
        in_chans: int = 3, 
        embed_dim: int = 96
    ) -> None:
        """Initialize the patch unembedding layer.
        
        Args:
            img_size: Image size (int for square, tuple for rectangular)
            patch_size: Patch token size
            in_chans: Number of input channels
            embed_dim: Number of linear projection output channels
        """
        super().__init__()
        # Handle both int and tuple input for img_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], 
            img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """Forward function.
        
        Args:
            x: Input tokens with shape (B, L, C)
            x_size: Spatial size of the input
            
        Returns:
            Unembedded tensor with shape (B, C, H, W)
        """
        B, HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, self.embed_dim, x_size[0], x_size[1])  # B C H W
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    
    def __init__(
        self, 
        dim: int, 
        input_resolution: Tuple[int, int], 
        depth: int, 
        num_heads: int, 
        window_size: int, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = True, 
        drop: float = 0.0, 
        attn_drop: float = 0.0, 
        drop_path: Union[float, List[float]] = 0.0, 
        norm_layer: nn.Module = nn.LayerNorm, 
        use_checkpoint: bool = False
    ) -> None:
        """Initialize a basic Swin Transformer layer.
        
        Args:
            dim: Number of input channels
            input_resolution: Input resolution
            depth: Number of blocks in the layer
            num_heads: Number of attention heads
            window_size: Local window size
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add a learnable bias to query, key, value
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer
            use_checkpoint: Whether to use checkpointing to save memory
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, 
                input_resolution=input_resolution, 
                num_heads=num_heads, 
                window_size=window_size, 
                shift_size=0 if (i % 2 == 0) else window_size // 2, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop, 
                attn_drop=attn_drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        
        Args:
            x: Input tensor with shape (B, H*W, C)
            
        Returns:
            Output tensor with shape (B, H*W, C)
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""
    
    def __init__(
        self, 
        dim: int, 
        input_resolution: Tuple[int, int], 
        depth: int, 
        num_heads: int, 
        window_size: int, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = True, 
        drop: float = 0.0, 
        attn_drop: float = 0.0, 
        drop_path: Union[float, List[float]] = 0.0, 
        norm_layer: nn.Module = nn.LayerNorm, 
        use_checkpoint: bool = False, 
        img_size: Union[int, Tuple[int, int]] = 224, 
        patch_size: int = 4,
        resi_connection: str = '1conv'
    ) -> None:
        """Initialize RSTB.
        
        Args:
            dim: Number of input channels
            input_resolution: Input resolution
            depth: Number of blocks in each layer
            num_heads: Number of attention heads
            window_size: Local window size
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Whether to add a learnable bias to query, key, value
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer
            use_checkpoint: Whether to use checkpointing to save memory
            img_size: Input image size (int for square, tuple for rectangular)
            patch_size: Patch size
            resi_connection: The convolutional block for residual connection
        """
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop, 
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )
        
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=dim, 
            embed_dim=dim,
            norm_layer=None
        )
        
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=0, 
            embed_dim=dim
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """Forward function.
        
        Args:
            x: Input tensor with shape (B, H*W, C)
            x_size: Input spatial size
            
        Returns:
            Output tensor with shape (B, H*W, C)
        """
        # Process through residual group
        residual_out = self.residual_group(x)
        
        # Convert to spatial representation for conv processing
        spatial_tensor = self.patch_unembed(residual_out, x_size)  # (B, C, H, W)
        
        # Apply convolution
        conv_out = self.conv(spatial_tensor)  # (B, C, H, W)
        
        # Convert back to sequence representation
        # Use the actual spatial dimensions from x_size instead of the configured img_size
        B, C, H, W = conv_out.shape
        sequence_out = conv_out.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Add residual connection
        return x + sequence_out


class SwinIR(nn.Module):
    """SwinIR: Image Restoration Using Swin Transformer."""
    
    def __init__(self, config: SwinIRConfig) -> None:
        """Initialize SwinIR.
        
        Args:
            config: Configuration for SwinIR
        """
        super().__init__()
        self.config = config
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(config.in_chans, config.embed_dim, 3, 1, 1)
        
        # Deep feature extraction
        self.num_layers = len(config.depths)
        self.embed_dim = config.embed_dim
        self.ape = config.ape
        self.patch_norm = config.patch_norm
        self.num_features = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=(config.img_height, config.img_width), 
            patch_size=config.patch_size, 
            in_chans=config.embed_dim, 
            embed_dim=config.embed_dim,
            norm_layer=nn.LayerNorm if self.patch_norm else None
        )
        
        # Patch unembedding for converting back to spatial representation
        self.patch_unembed = PatchUnEmbed(
            img_size=(config.img_height, config.img_width),
            patch_size=config.patch_size,
            in_chans=0,
            embed_dim=config.embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, config.embed_dim)
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
            
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]  
        
        # Build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=config.embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=config.qkv_bias, 
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                use_checkpoint=config.use_checkpoint,
                img_size=(config.img_height, config.img_width),
                patch_size=config.patch_size,
                resi_connection=config.resi_connection
            )
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(self.num_features)
        
        # Build the last conv layer
        self.conv_after_body = nn.Conv2d(config.embed_dim, config.embed_dim, 3, 1, 1)
        
        # Upsampling (for super-resolution, set to 1 for our case)
        if config.upscale > 1:
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(config.embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=config.upscale, mode='nearest'),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv_before_upsample = nn.Identity()
            self.upsample = nn.Identity()
        
        # Final output conv
        self.conv_last = nn.Conv2d(64 if config.upscale > 1 else config.embed_dim, 
                                  config.out_chans, 3, 1, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for the model.
        
        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Return parameter names that should not decay.
        
        Returns:
            Set of parameter names
        """
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> Set[str]:
        """Return parameter keywords that should not decay.
        
        Returns:
            Set of parameter keywords
        """
        return {'relative_position_bias_table'}
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.
        
        Args:
            x: Input tensor
            
        Returns:
            Extracted feature tensor
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x, x_size)
        
        x = self.norm(x)  # B, L, C
        x = self.patch_unembed(x, x_size)  # B, C, H, W
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.
        
        Args:
            x: Input tensor with shape (B, C, H, W)
            
        Returns:
            Output tensor with shape (B, C, H, W)
        """
        # Shallow feature extraction
        x = self.conv_first(x)
        
        # Deep feature extraction
        res = x
        x = self.forward_features(x)
        x = self.conv_after_body(x) + res
        
        # Upsampling and final output
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        
        return x


def create_swinir_model(img_size: Union[int, Tuple[int, int]] = 256,
                       in_chans: int = 3,
                       out_chans: int = 3,
                       window_size: int = 8,
                       embed_dim: int = 96,
                       depths: Optional[List[int]] = None,
                       num_heads: Optional[List[int]] = None,
                       use_checkpoint: bool = False) -> SwinIR:
    """Create a SwinIR model with specified configurations.
    
    Args:
        img_size: Input image size (int for square, tuple for rectangular)
        in_chans: Number of input channels
        out_chans: Number of output channels
        window_size: Window size for attention
        embed_dim: Embedding dimension
        depths: Depths of each Swin Transformer layer
        num_heads: Number of attention heads in different layers
        use_checkpoint: Whether to use checkpointing to save memory
        
    Returns:
        SwinIR model
    """
    if depths is None:
        depths = [6, 6, 6, 6]  # Default depths for moderate-sized model
        
    if num_heads is None:
        num_heads = [6, 6, 6, 6]  # Default number of heads
    
    config = SwinIRConfig(
        img_size=img_size,
        in_chans=in_chans,
        out_chans=out_chans,
        window_size=window_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        patch_size=1,  # Set to 1 for image restoration (no downsampling)
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        upscale=1,  # No upscaling for restoration
        patch_norm=True,
        ape=False,  # Not using absolute position embedding for restoration
        use_checkpoint=use_checkpoint,
        resi_connection="1conv"  # Using 1 conv for residual connection
    )
    
    return SwinIR(config)
