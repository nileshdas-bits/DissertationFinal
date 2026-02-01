"""Autoencoder model for representation learning with classification head.

Two-stage training:
  Stage A: Autoencoder pretraining (reconstruction MSE)
  Stage B: Classification head training (CrossEntropyLoss, optionally freeze encoder)
"""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder with classification head."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 3
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc_latent = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder_init = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc_reconstruct = nn.Linear(hidden_size, input_size)
        
        # Classification head (MLP: latent -> hidden -> output)
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self._encoder_frozen = False
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representation."""
        _, (h_n, _) = self.encoder(x)
        latent = self.fc_latent(h_n[-1])
        return latent
    
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent representation to reconstructed sequence."""
        h_init = self.decoder_init(latent)
        h_init = h_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_init = torch.zeros_like(h_init)
        
        # Teacher forcing with zeros (autoregressive would be better but slower)
        decoder_input = torch.zeros(
            latent.size(0), seq_len, self.input_size,
            device=latent.device
        )
        
        decoder_out, _ = self.decoder(decoder_input, (h_init, c_init))
        reconstructed = self.fc_reconstruct(decoder_out)
        
        return reconstructed
    
    def freeze_encoder(self) -> None:
        """Freeze encoder and decoder weights for classification fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_latent.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.decoder_init.parameters():
            param.requires_grad = False
        for param in self.fc_reconstruct.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder and decoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.fc_latent.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        for param in self.decoder_init.parameters():
            param.requires_grad = True
        for param in self.fc_reconstruct.parameters():
            param.requires_grad = True
        self._encoder_frozen = False
    
    @property
    def encoder_frozen(self) -> bool:
        return self._encoder_frozen
    
    def forward(
        self,
        x: torch.Tensor,
        return_reconstruction: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_size)
            return_reconstruction: whether to return reconstruction
            
        Returns:
            If return_reconstruction=False:
                prediction: (batch, output_size) - classification logits
            If return_reconstruction=True:
                prediction: (batch, output_size) - classification logits
                reconstruction: (batch, seq_len, input_size)
                latent: (batch, latent_size)
        """
        latent = self.encode(x)
        prediction = self.classifier(latent)
        
        if return_reconstruction:
            reconstruction = self.decode(latent, x.size(1))
            return prediction, reconstruction, latent
        
        return prediction
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size
        }


def create_autoencoder(
    input_size: int,
    task: str = "classification",
    hidden_size: int = 64,
    latent_size: int = 32,
    num_layers: int = 2,
    dropout: float = 0.2,
    num_classes: int = 3
) -> LSTMAutoencoder:
    """Create autoencoder model."""
    output_size = 1 if task == "regression" else num_classes
    return LSTMAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=output_size
    )
