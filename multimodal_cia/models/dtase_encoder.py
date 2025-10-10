"""
DTASE (Dynamic Task-Adaptive Squeeze-and-Excitation) Encoder Module

This module implements the original DTASE encoder for multi-modal cognitive assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSEEncoder(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 num_tasks: int,
                 reduction: int = 4,
                 output_dim: int = None):
        """
        DTASE (Dynamic Task-Adaptive Squeeze-and-Excitation) module.
        
        Args:
            feat_dim: Feature dimension
            num_tasks: Number of tasks
            reduction: Reduction ratio for squeeze operation
            output_dim: Output dimension (optional)
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_tasks = num_tasks
        self.reduction = reduction

        # FC1: ℝ^{T} → ℝ^{T//reduction}
        hidden_dim = max(1, num_tasks // reduction)
        self.fc1 = nn.Linear(num_tasks, hidden_dim, bias=True)
        # FC2: ℝ^{T//reduction} → ℝ^{T}
        self.fc2 = nn.Linear(hidden_dim, num_tasks, bias=True)

        # LayerNorm for residual connection
        self.norm = nn.LayerNorm(feat_dim)

        # Optional projection layer
        self.output_dim = output_dim if (output_dim is not None) else feat_dim
        if self.output_dim != self.feat_dim:
            self.proj = nn.Linear(self.feat_dim, self.output_dim)
        else:
            self.proj = None

    def forward(self, feat: torch.Tensor):
        """
        Forward pass through DTASE encoder.
        
        Args:
            feat: Input features [B, T, D]
            
        Returns:
            encoded: Encoded features [B, output_dim]
            gate: Attention gates [B, T, 1]
        """
        B, T, D = feat.shape
        assert T == self.num_tasks and D == self.feat_dim

        # Squeeze operation
        z = feat.mean(dim=2)  # [B, T]

        # Excitation operation
        s = F.relu(self.fc1(z), inplace=True)  # [B, T//r]
        w_raw = self.fc2(s)                    # [B, T]
        w = torch.sigmoid(w_raw)               # [B, T]

        # Rescale and gate application
        gate = w.unsqueeze(-1)  # [B, T, 1]

        # Apply gates with residual connection
        feat_gated = feat * gate        # [B, T, D]
        feat_out = self.norm(feat_gated + feat)  # [B, T, D]

        # Pooling and optional projection
        encoded = torch.sum(feat_out * gate, dim=1)  # [B, D]
        if self.proj is not None:
            encoded = self.proj(encoded)  # [B, output_dim]

        return encoded, gate           # [B, output_dim], [B, T, 1]


class ModalityFusion(nn.Module):
    """Modality fusion module for combining multiple modality features."""
    
    def __init__(self, num_modalities: int, feature_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        
        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        
    def forward(self, modality_features: list):
        """
        Args:
            modality_features: List of [B, feature_dim] tensors
        Returns:
            fused_features: [B, feature_dim]
        """
        # Normalize weights
        weights = F.softmax(self.modality_weights, dim=0)
        
        # Weighted combination
        fused = torch.zeros_like(modality_features[0])
        for i, features in enumerate(modality_features):
            fused += weights[i] * features
            
        return fused
