"""Temporal Convolutional Network (TCN) model.

No external TCN libraries - pure PyTorch implementation.
Ensures causality: no future information leaks into predictions.
"""
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal convolution with left padding.
    
    Ensures output at time t only depends on inputs at times <= t.
    Padding is applied only to the left side.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        # Left padding to maintain causality: (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad left side only: (left_pad, right_pad)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Single TCN residual block with two causal convolutions.
    
    Structure:
        x -> CausalConv -> ReLU -> Dropout -> CausalConv -> ReLU -> Dropout -> (+residual) -> ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 1x1 conv for channel matching in residual
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return self.relu(out + residual)


class TCNNet(nn.Module):
    """
    Temporal Convolutional Network for sequence classification.
    
    Architecture:
        Input (N, W, F) -> transpose to (N, F, W)
        -> Stack of TCN blocks with exponential dilation (1, 2, 4, 8, ...)
        -> Pooling (last timestep or global average)
        -> Linear head -> num_classes
    
    Receptive field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    With default num_channels=[64,64,64], kernel_size=3, dilations=[1,2,4]:
        RF = 1 + 2 * 2 * (1+2+4) = 1 + 28 = 29 timesteps
    """
    
    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_size: int = 1,
        pooling: str = "last"  # "last" or "avg"
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.output_size = output_size
        self.pooling = pooling
        
        assert pooling in ("last", "avg"), f"pooling must be 'last' or 'avg', got {pooling}"
        
        layers = []
        in_channels = input_size
        
        # Build TCN blocks with exponential dilation: 1, 2, 4, 8, ...
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) - input sequence
            
        Returns:
            output: (batch, output_size) - class logits
        """
        # Transpose to (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply TCN blocks
        out = self.tcn(x)
        
        # Pooling
        if self.pooling == "last":
            # Take last timestep (causal: only uses past information)
            out = out[:, :, -1]
        else:
            # Global average pooling over time dimension
            out = out.mean(dim=2)
        
        # Classification head
        out = self.fc(out)
        
        return out.squeeze(-1) if self.output_size == 1 else out
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "output_size": self.output_size,
            "pooling": self.pooling
        }


def create_tcn(
    input_size: int,
    task: str = "regression",
    num_channels: List[int] = [64, 64, 64],
    kernel_size: int = 3,
    dropout: float = 0.2,
    pooling: str = "last",
    num_classes: int = 3
) -> TCNNet:
    """Create TCN model."""
    output_size = 1 if task == "regression" else num_classes
    return TCNNet(
        input_size=input_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        output_size=output_size,
        pooling=pooling
    )
