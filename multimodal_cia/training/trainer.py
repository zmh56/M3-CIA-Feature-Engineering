"""
Training module for multi-modal cognitive impairment assessment models.

This module provides comprehensive training capabilities including
multi-task learning, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

from ..evaluation.metrics import EvaluationMetrics


class Trainer:
    """
    Trainer for multi-modal cognitive impairment assessment models.
    
    This class provides comprehensive training capabilities including
    multi-task learning, early stopping, model checkpointing, and
    comprehensive evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        intermediate_loss_weight: float = 0.5,
        final_loss_weight: float = 1.0,
        num_epochs: int = 100,
        patience: int = 10,
        save_dir: str = './checkpoints',
        experiment_name: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            intermediate_loss_weight: Weight for intermediate supervision loss
            final_loss_weight: Weight for final classification loss
            num_epochs: Maximum number of training epochs
            patience: Number of epochs to wait for improvement before early stopping
            save_dir: Directory to save model checkpoints
            experiment_name: Name for the experiment (used for logging)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.intermediate_loss_weight = intermediate_loss_weight
        self.final_loss_weight = final_loss_weight
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss functions
        self.intermediate_loss = nn.BCEWithLogitsLoss()
        self.final_loss = nn.CrossEntropyLoss()
        
        # Evaluation metrics
        self.evaluator = EvaluationMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.best_model_state = None
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Setup directories
        self.save_dir = save_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Early stopping
        self.early_stopping_counter = 0
        self.early_stopping_best_score = 0.0
        
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and checkpointing.
        
        Returns:
            Dict[str, List[float]]: Training history including losses and metrics
        """
        print(f"Starting training on {self.device}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': [],
            'learning_rate': []
        }
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_metrics['auc'])
            
            # Store metrics
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['val_precision'].append(val_metrics['precision'])
            training_history['val_recall'].append(val_metrics['recall'])
            training_history['val_f1'].append(val_metrics['f1'])
            training_history['val_auc'].append(val_metrics['auc'])
            training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            self._print_epoch_progress(epoch, train_loss, val_metrics)
            
            # Check for early stopping
            if self._check_early_stopping(val_metrics['auc']):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Save best model
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, val_metrics, is_best=True)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final results
        self._save_training_history(training_history)
        
        print(f"Training completed. Best validation AUC: {self.best_val_auc:.4f}")
        
        return training_history
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            intermediate_outputs, final_output = self.model(batch)
            
            # Debug: Check for NaN values in model outputs
            if torch.isnan(final_output).any():
                print(f"Warning: NaN detected in final_output at batch {batch_idx}")
                print(f"final_output stats: min={final_output.min():.4f}, max={final_output.max():.4f}")
                # Replace NaN with zeros
                final_output = torch.where(torch.isnan(final_output), torch.zeros_like(final_output), final_output)
            
            # Compute losses
            intermediate_loss = self._compute_intermediate_loss(
                intermediate_outputs, batch
            )
            final_loss = self.final_loss(final_output, batch['target'])
            
            # Combined loss
            total_loss_batch = (
                self.intermediate_loss_weight * intermediate_loss +
                self.final_loss_weight * final_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Int_Loss': f'{intermediate_loss.item():.4f}',
                'Fin_Loss': f'{final_loss.item():.4f}'
            })
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                intermediate_outputs, final_output = self.model(batch)
                
                # Debug: Check for NaN values in model outputs
                if torch.isnan(final_output).any():
                    print(f"Warning: NaN detected in final_output during validation")
                    # Replace NaN with zeros
                    final_output = torch.where(torch.isnan(final_output), torch.zeros_like(final_output), final_output)
                
                # Compute losses
                intermediate_loss = self._compute_intermediate_loss(
                    intermediate_outputs, batch
                )
                final_loss = self.final_loss(final_output, batch['target'])
                
                total_loss_batch = (
                    self.intermediate_loss_weight * intermediate_loss +
                    self.final_loss_weight * final_loss
                )
                
                total_loss += total_loss_batch.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(final_output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['target'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics(
            all_targets, all_predictions, all_probabilities
        )
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def _compute_intermediate_loss(
        self, 
        intermediate_outputs: List[torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute intermediate supervision loss."""
        if 'intermediate_targets' in batch:
            # Multi-task learning with intermediate targets
            total_loss = 0.0
            for i, output in enumerate(intermediate_outputs):
                if i < len(batch['intermediate_targets']):
                    target = batch['intermediate_targets'][i]
                    # Ensure target is on the same device as output
                    if target.device != output.device:
                        target = target.to(output.device)
                    # Convert target to float for BCEWithLogitsLoss
                    target_float = target.float()
                    loss = self.intermediate_loss(output.squeeze(), target_float)
                    total_loss += loss
            return total_loss / len(intermediate_outputs)
        else:
            # Use final target for intermediate supervision (simplified approach)
            target = batch['target']
            total_loss = 0.0
            for output in intermediate_outputs:
                # Convert target to binary for each intermediate output
                target_binary = (target == 0).float()  # Use first class as binary target
                # Ensure target is on the same device as output
                if target_binary.device != output.device:
                    target_binary = target_binary.to(output.device)
                loss = self.intermediate_loss(output.squeeze(), target_binary)
                total_loss += loss
            return total_loss / len(intermediate_outputs)
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to the appropriate device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list) and key == 'intermediate_targets':
                # Handle intermediate_targets list
                device_batch[key] = [tensor.to(self.device) for tensor in value]
            else:
                device_batch[key] = value
        return device_batch
    
    def _check_early_stopping(self, val_auc: float) -> bool:
        """Check if early stopping criteria are met."""
        if val_auc > self.early_stopping_best_score:
            self.early_stopping_best_score = val_auc
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.patience
    
    def _print_epoch_progress(
        self, 
        epoch: int, 
        train_loss: float, 
        val_metrics: Dict[str, float]
    ):
        """Print training progress for the current epoch."""
        print(f"Epoch {epoch:3d}/{self.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        val_metrics: Dict[str, float], 
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'best_val_auc': self.best_val_auc,
            'experiment_name': self.experiment_name
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_checkpoint_path)
    
    def _save_training_history(self, training_history: Dict[str, List[float]]):
        """Save training history to JSON file."""
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = self._move_batch_to_device(batch)
                
                _, final_output = self.model(batch)
                
                probabilities = torch.softmax(final_output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['target'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute comprehensive metrics
        metrics = self.evaluator.compute_metrics(
            all_targets, all_predictions, all_probabilities
        )
        
        print("Test Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def train_simple(self) -> Dict[str, List[float]]:
        """
        Simplified training method.
        
        Returns:
            Dict[str, List[float]]: Training history and final validation results
        """
        print("=" * 70)
        print(f"Starting training on device: {self.device}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Total epochs: {self.num_epochs}")
        print("-" * 70)
        
        training_history = {
            'train_loss': [],
        }
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                intermediate_outputs, final_output = self.model(batch)
                
                # Check for NaN
                if torch.isnan(final_output).any():
                    final_output = torch.where(torch.isnan(final_output), 
                                              torch.zeros_like(final_output), final_output)
                
                # Compute losses
                intermediate_loss = self._compute_intermediate_loss(intermediate_outputs, batch)
                final_loss = self.final_loss(final_output, batch['target'])
                
                # Combined loss
                total_loss_batch = (
                    self.intermediate_loss_weight * intermediate_loss +
                    self.final_loss_weight * final_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
            
            # Calculate and record average training loss
            avg_train_loss = total_loss / num_batches
            training_history['train_loss'].append(avg_train_loss)
            
            # Print training loss only
            print(f"Epoch [{epoch+1:3d}/{self.num_epochs}] - Train Loss: {avg_train_loss:.4f}")
        
        print("-" * 70)
        print("Training completed! Starting final evaluation...")
        print("=" * 70)
        
        # Evaluate validation set once after training
        val_metrics = self._validate_epoch()
        
        # Save validation results
        training_history['final_val_metrics'] = val_metrics
        
        # Print final validation results
        print("\nFinal validation results:")
        print("-" * 70)
        print(f"AUC-ROC:            {val_metrics['auc']:.4f}")
        print("=" * 70)
        
        # # Save final model
        # self._save_checkpoint(self.num_epochs - 1, val_metrics, is_best=False)
        # self._save_training_history(training_history)
        
        return training_history
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and probabilities
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                batch = self._move_batch_to_device(batch)
                
                _, final_output = self.model(batch)
                
                probabilities = torch.softmax(final_output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
