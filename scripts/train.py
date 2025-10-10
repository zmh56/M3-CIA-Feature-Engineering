#!/usr/bin/env python3
"""
Training script example for Multi-Modal Cognitive Assessment Model
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import from the package
from multimodal_cia.models.multimodal_model import MultiModalModel
from multimodal_cia.training.trainer import Trainer
from multimodal_cia.data.mat_loader import create_mat_data_loaders


def simple_train(args):
    """Simple training function."""
    print("Starting simple training...")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data from .mat files
    if args.data_path is None or not os.path.exists(args.data_path):
        raise ValueError(f"Data file not found: {args.data_path}. Please provide a valid data file path.")
    
    print(f"Loading data from {args.data_path}")
    train_loader, val_loader = create_mat_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        train_ratio=0.8,
        use_multi_file=args.use_multi_file
    )
    
    # Create model
    model = MultiModalModel(
        num_classes=args.num_classes,
        intermediate_dim=args.intermediate_dim
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        patience=args.patience
    )
    
    # Train model
    print("Training model...")
    training_history = trainer.train_simple()
    
    print("Training completed!")
    return model, training_history


def main():
    parser = argparse.ArgumentParser(description="Simple Training Script")
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--intermediate_dim', type=int, default=7)
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data_example/sampled_100.npz', help='Path to .mat/npz data file or directory')
    parser.add_argument('--use_multi_file', action='store_true', help='Use multiple .mat files')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=30)
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Train model
    model, history = simple_train(args)
    
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
