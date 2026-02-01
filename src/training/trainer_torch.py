"""Trainer for PyTorch models."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .early_stopping import EarlyStopping
from ..utils.logging import get_logger


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


class TorchTrainer:
    """Trainer for PyTorch models."""
    
    def __init__(
        self,
        model: nn.Module,
        task: str = "regression",
        learning_rate: float = 1e-3,
        device: str = "auto",
        early_stopping_patience: int = 10
    ):
        self.model = model
        self.task = task
        self.learning_rate = learning_rate
        self.device = get_device(device)
        self.early_stopping_patience = early_stopping_patience
        
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        if task == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.logger = get_logger()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.pretrain_losses: List[float] = []
        self.pretrain_val_losses: List[float] = []
        self.best_model_state: Optional[Dict[str, Any]] = None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        reconstruction_weight: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train the model (single-stage).
        
        Args:
            train_loader: training data loader
            val_loader: validation data loader
            epochs: number of epochs
            reconstruction_weight: weight for reconstruction loss (autoencoder)
            
        Returns:
            training history
        """
        self.logger.info(f"Training on {self.device}")
        
        early_stopping = EarlyStopping(patience=self.early_stopping_patience)
        is_autoencoder = hasattr(self.model, "encode")
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(
                train_loader, is_autoencoder, reconstruction_weight
            )
            self.train_losses.append(train_loss)
            
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(
                    val_loader, is_autoencoder, reconstruction_weight
                )
                self.val_losses.append(val_loss)
                
                if early_stopping(val_loss, epoch):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if early_stopping.best_epoch == epoch:
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                self.logger.info(msg)
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": early_stopping.best_epoch
        }
    
    def train_two_stage(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        pretrain_epochs: int = 50,
        finetune_epochs: int = 50,
        freeze_encoder: bool = True,
        pretrain_lr: Optional[float] = None,
        finetune_lr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Two-stage training for autoencoder models.
        
        Stage A: Autoencoder pretraining (reconstruction MSE only)
        Stage B: Classification fine-tuning (optionally freeze encoder)
        
        Args:
            train_loader: training data loader
            val_loader: validation data loader
            pretrain_epochs: epochs for pretraining
            finetune_epochs: epochs for fine-tuning
            freeze_encoder: whether to freeze encoder during fine-tuning
            pretrain_lr: learning rate for pretraining (default: self.learning_rate)
            finetune_lr: learning rate for fine-tuning (default: self.learning_rate)
            
        Returns:
            training history with both stages
        """
        assert hasattr(self.model, "encode"), "Two-stage training requires autoencoder model"
        
        pretrain_lr = pretrain_lr or self.learning_rate
        finetune_lr = finetune_lr or self.learning_rate
        
        # ========== Stage A: Autoencoder Pretraining ==========
        self.logger.info(f"=== Stage A: Autoencoder Pretraining ({pretrain_epochs} epochs) ===")
        self.logger.info(f"Training on {self.device}, lr={pretrain_lr}")
        
        # Reset optimizer for pretraining
        self.optimizer = Adam(self.model.parameters(), lr=pretrain_lr)
        early_stopping_pretrain = EarlyStopping(patience=self.early_stopping_patience)
        
        for epoch in range(pretrain_epochs):
            train_loss = self._pretrain_epoch(train_loader)
            self.pretrain_losses.append(train_loss)
            
            val_loss = None
            if val_loader is not None:
                val_loss = self._pretrain_validate_epoch(val_loader)
                self.pretrain_val_losses.append(val_loss)
                
                if early_stopping_pretrain(val_loss, epoch):
                    self.logger.info(f"Pretrain early stopping at epoch {epoch + 1}")
                    break
                
                if early_stopping_pretrain.best_epoch == epoch:
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Pretrain Epoch {epoch + 1}/{pretrain_epochs} | Recon Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Recon Loss: {val_loss:.6f}"
                self.logger.info(msg)
        
        # Load best pretrained state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # ========== Stage B: Classification Fine-tuning ==========
        self.logger.info(f"=== Stage B: Classification Fine-tuning ({finetune_epochs} epochs) ===")
        self.logger.info(f"freeze_encoder={freeze_encoder}, lr={finetune_lr}")
        
        if freeze_encoder:
            self.model.freeze_encoder()
            # Only optimize classifier parameters
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=finetune_lr
            )
        else:
            self.model.unfreeze_encoder()
            self.optimizer = Adam(self.model.parameters(), lr=finetune_lr)
        
        early_stopping_finetune = EarlyStopping(patience=self.early_stopping_patience)
        self.best_model_state = None
        
        for epoch in range(finetune_epochs):
            train_loss = self._finetune_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            val_loss = None
            if val_loader is not None:
                val_loss = self._finetune_validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                if early_stopping_finetune(val_loss, epoch):
                    self.logger.info(f"Finetune early stopping at epoch {epoch + 1}")
                    break
                
                if early_stopping_finetune.best_epoch == epoch:
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Finetune Epoch {epoch + 1}/{finetune_epochs} | CE Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val CE Loss: {val_loss:.6f}"
                self.logger.info(msg)
        
        # Load best finetuned state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            "pretrain_losses": self.pretrain_losses,
            "pretrain_val_losses": self.pretrain_val_losses,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": early_stopping_finetune.best_epoch
        }
    
    def _pretrain_epoch(self, loader: DataLoader) -> float:
        """Pretrain epoch: reconstruction loss only."""
        self.model.train()
        total_loss = 0.0
        recon_criterion = nn.MSELoss()
        
        for X, _ in loader:
            X = X.to(self.device)
            
            self.optimizer.zero_grad()
            
            _, recon, _ = self.model(X, return_reconstruction=True)
            loss = recon_criterion(recon, X)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(X)
        
        return total_loss / len(loader.dataset)
    
    def _pretrain_validate_epoch(self, loader: DataLoader) -> float:
        """Pretrain validation: reconstruction loss only."""
        self.model.eval()
        total_loss = 0.0
        recon_criterion = nn.MSELoss()
        
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                _, recon, _ = self.model(X, return_reconstruction=True)
                loss = recon_criterion(recon, X)
                total_loss += loss.item() * len(X)
        
        return total_loss / len(loader.dataset)
    
    def _finetune_epoch(self, loader: DataLoader) -> float:
        """Finetune epoch: classification loss only."""
        self.model.train()
        total_loss = 0.0
        
        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device).long()
            
            self.optimizer.zero_grad()
            
            pred = self.model(X, return_reconstruction=False)
            loss = self.criterion(pred, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(X)
        
        return total_loss / len(loader.dataset)
    
    def _finetune_validate_epoch(self, loader: DataLoader) -> float:
        """Finetune validation: classification loss only."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device).long()
                pred = self.model(X, return_reconstruction=False)
                loss = self.criterion(pred, y)
                total_loss += loss.item() * len(X)
        
        return total_loss / len(loader.dataset)
    
    def _train_epoch(
        self,
        loader: DataLoader,
        is_autoencoder: bool,
        reconstruction_weight: float
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            if is_autoencoder:
                pred, recon, _ = self.model(X, return_reconstruction=True)
                pred_loss = self._compute_loss(pred, y)
                recon_loss = nn.MSELoss()(recon, X)
                loss = pred_loss + reconstruction_weight * recon_loss
            else:
                pred = self.model(X)
                loss = self._compute_loss(pred, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(X)
        
        return total_loss / len(loader.dataset)
    
    def _validate_epoch(
        self,
        loader: DataLoader,
        is_autoencoder: bool,
        reconstruction_weight: float
    ) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                if is_autoencoder:
                    pred, recon, _ = self.model(X, return_reconstruction=True)
                    pred_loss = self._compute_loss(pred, y)
                    recon_loss = nn.MSELoss()(recon, X)
                    loss = pred_loss + reconstruction_weight * recon_loss
                else:
                    pred = self.model(X)
                    loss = self._compute_loss(pred, y)
                
                total_loss += loss.item() * len(X)
        
        return total_loss / len(loader.dataset)
    
    def _compute_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss based on task. Expects y in {0, 1, 2} for classification."""
        if self.task == "classification":
            y = y.long()
        return self.criterion(pred, y)
    
    def predict(self, loader: DataLoader) -> np.ndarray:
        """Make predictions. Returns class indices {0, 1, 2} for classification."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                
                if hasattr(self.model, "encode"):
                    pred = self.model(X, return_reconstruction=False)
                else:
                    pred = self.model(X)
                
                if self.task == "classification":
                    pred = torch.argmax(pred, dim=1)
                
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def predict_proba(self, loader: DataLoader) -> Optional[np.ndarray]:
        """Predict probabilities (classification only)."""
        if self.task != "classification":
            return None
        
        self.model.eval()
        probas = []
        
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                
                if hasattr(self.model, "encode"):
                    pred = self.model(X, return_reconstruction=False)
                else:
                    pred = self.model(X)
                
                proba = torch.softmax(pred, dim=1)
                probas.append(proba.cpu().numpy())
        
        return np.concatenate(probas)
    
    def save(self, path: Path) -> None:
        """Save model state."""
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "pretrain_losses": self.pretrain_losses,
            "pretrain_val_losses": self.pretrain_val_losses
        }, path)
    
    def load(self, path: Path) -> None:
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.pretrain_losses = checkpoint.get("pretrain_losses", [])
        self.pretrain_val_losses = checkpoint.get("pretrain_val_losses", [])
