"""
Multi-Modal Deep Learning Model for Cognitive Assessment

This module implements the multi-modal model using DTASE encoders
based on the user's proven architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .dtase_encoder import TaskSEEncoder


class MultiModalModel(nn.Module):
    """
    Multi-Modal Model for Cognitive Assessment using DTASE encoders
    
    Multi-modal fusion architecture with intermediate supervision.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        intermediate_dim: int = 7,
        dropout_rate: float = 0.2
    ):
        super(MultiModalModel, self).__init__()
        
        self.num_classes = num_classes
        self.intermediate_dim = intermediate_dim
        
        # Feature mapping layers
        self.mod_a_map = nn.Linear(27, 16)
        self.mod_b_map = nn.Linear(20, 16)
        self.mod_c_map = nn.Linear(17, 16)
        
        # DTASE encoders
        self.mod_a_dtase = TaskSEEncoder(feat_dim=16, num_tasks=11, reduction=4, output_dim=8)
        self.mod_b_dtase = TaskSEEncoder(feat_dim=16, num_tasks=11, reduction=4, output_dim=8)
        self.mod_c_dtase = TaskSEEncoder(feat_dim=16, num_tasks=3, reduction=3, output_dim=8)
        
        # Output mapping layers
        self.mod_a_map2 = nn.Linear(8, 8)
        self.mod_b_map2 = nn.Linear(8, 8)
        self.mod_c_map2 = nn.Linear(8, 8)
        
        # Static modality encoders
        self.mod_d_fc = nn.Sequential(
            nn.Linear(20, 8),
            nn.BatchNorm1d(8, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )
        self.mod_e_fc = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )
        self.mod_f_fc = nn.Sequential(
            nn.Linear(11, 8),
            nn.BatchNorm1d(8, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )
        
        # Modality fusion weights
        init_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.modality_weights = nn.Parameter(init_weights)
        
        # Feature fusion layers
        self.combined_fc = nn.Sequential(
            nn.BatchNorm1d(6 * 8, momentum=0.05),
            nn.Linear(6 * 8, 32),
            nn.BatchNorm1d(32, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )

        self.combined_fc_ablation = nn.Sequential(
            nn.BatchNorm1d(1 * 8, momentum=0.05),
            nn.Linear(1 * 8, 32),
            nn.BatchNorm1d(32, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )
        
        # Intermediate supervision layers
        self.intermediate = nn.ModuleList([nn.Linear(16, 1) for _ in range(intermediate_dim)])
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classifier
        self.classifier = nn.Linear(intermediate_dim, num_classes)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the multi-modal model.
        
        Args:
            batch: Dictionary containing modality data
        
        Returns:
            Tuple of (intermediate_outputs, final_output)
        """
        # Extract data from batch
        mod_a = batch['mod_a']
        mod_b = batch['mod_b']
        mod_c = batch['mod_c']
        mod_d = batch['mod_d']
        mod_e = batch['mod_e']
        mod_f = batch['mod_f']
        
        B = mod_e.shape[0]
        
        # Process modality A
        mod_a_mapped = self.mod_a_map(mod_a)
        mod_a_encoded, _ = self.mod_a_dtase(mod_a_mapped)
        mod_a_enc = self.mod_a_map2(mod_a_encoded)
        
        # Process modality B
        mod_b_mapped = self.mod_b_map(mod_b)
        mod_b_encoded, _ = self.mod_b_dtase(mod_b_mapped)
        mod_b_enc = self.mod_b_map2(mod_b_encoded)
        
        # Process modality C
        mod_c_mapped = self.mod_c_map(mod_c)
        mod_c_encoded, _ = self.mod_c_dtase(mod_c_mapped)
        mod_c_enc = self.mod_c_map2(mod_c_encoded)
        
        # Process static modalities
        mod_d_enc = self.mod_d_fc(mod_d)
        mod_e_enc = self.mod_e_fc(mod_e)
        mod_f_enc = self.mod_f_fc(mod_f)
        
        # Modality fusion with learnable weights
        weights = F.softmax(self.modality_weights, dim=0)
        w_a, w_b, w_c, w_d, w_e, w_f = weights
        
        # Apply weights to modality features
        mod_a_f = w_a * mod_a_enc
        mod_b_f = w_b * mod_b_enc
        mod_c_f = w_c * mod_c_enc
        mod_d_f = w_d * mod_d_enc
        mod_e_f = w_e * mod_e_enc
        mod_f_f = w_f * mod_f_enc
        
        # Concatenate weighted features
        fused = torch.cat([mod_a_f, mod_b_f, mod_c_f, mod_d_f, mod_e_f, mod_f_f], dim=1)
        
        # Feature fusion
        fused_feature = self.combined_fc(fused)
        
        # Intermediate supervision
        intermediate_outputs = []
        for layer in self.intermediate:
            out = self.activation(layer(fused_feature))
            out = self.dropout(out)
            intermediate_outputs.append(out)
        
        concatenated = torch.cat(intermediate_outputs, dim=1)
        final_output = self.classifier(concatenated)
        
        return intermediate_outputs, final_output


class CascadedMultiLabelModel(nn.Module):
    """
    Cascaded Multi-Label Model for multi-level classification
    
    This model extends the MultiModalModel to support cascaded classification
    with intermediate supervision at multiple levels.
    """
    
    def __init__(
        self,
        base_model: MultiModalModel,
        cascade_levels: List[int] = [7, 5, 3, 2],
        dropout_rate: float = 0.2
    ):
        super(CascadedMultiLabelModel, self).__init__()
        
        self.base_model = base_model
        self.cascade_levels = cascade_levels
        
        # Create cascade classifiers
        self.cascade_classifiers = nn.ModuleList()
        for level in cascade_levels:
            classifier = nn.Sequential(
                nn.Linear(16, level),
                nn.BatchNorm1d(level),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            )
            self.cascade_classifiers.append(classifier)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass with cascaded classification
        
        Returns:
            Tuple of (intermediate_outputs, final_output)
        """
        # Get base model outputs
        intermediate_outputs, final_output = self.base_model(batch)
        
        # Get the fused features from base model (we need to modify base model to expose this)
        # For now, we'll use the intermediate features
        return intermediate_outputs, final_output