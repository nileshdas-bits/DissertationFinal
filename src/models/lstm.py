"""LSTM model."""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """LSTM network for sequence modeling."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            
        Returns:
            output: (batch, output_size)
        """
        lstm_out, (h_n, _) = self.lstm(x)
        
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        
        out = self.dropout_layer(last_hidden)
        out = self.fc(out)
        
        return out.squeeze(-1) if self.output_size == 1 else out
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size,
            "bidirectional": self.bidirectional
        }


def create_lstm(
    input_size: int,
    task: str = "regression",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    num_classes: int = 3
) -> LSTMNet:
    """Create LSTM model."""
    output_size = 1 if task == "regression" else num_classes
    return LSTMNet(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=output_size,
        bidirectional=bidirectional
    )
