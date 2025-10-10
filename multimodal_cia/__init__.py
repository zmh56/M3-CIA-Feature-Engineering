"""
Multi-Modal Cognitive Impairment Assessment (M3-CIA)

A comprehensive deep learning framework for cognitive impairment assessment 
using wearable physiological and behavioral signals.

This package provides state-of-the-art multi-modal fusion approaches for 
early detection and classification of cognitive disorders including 
Mild Cognitive Impairment (MCI) and Dementia.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@institution.edu"

# Import main classes for easy access
from .models.multimodal_model import MultiModalModel, CascadedMultiLabelModel
from .models.dtase_encoder import TaskSEEncoder, ModalityFusion
from .data.mat_loader import MultiModalDataset, create_mat_data_loaders
from .training.trainer import Trainer
from .evaluation.metrics import EvaluationMetrics

__all__ = [
    "MultiModalModel",
    "CascadedMultiLabelModel",
    "TaskSEEncoder",
    "ModalityFusion",
    "MultiModalDataset",
    "create_mat_data_loaders",
    "Trainer",
    "EvaluationMetrics",
]
