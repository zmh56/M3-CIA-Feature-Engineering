"""
Evaluation metrics for multi-modal cognitive impairment assessment models.

This module provides comprehensive evaluation metrics including classification
metrics, confusion matrix analysis, and ROC curve computation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional, Union
import torch


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for cognitive impairment assessment models.
    
    This class provides various metrics for evaluating model performance
    including classification accuracy, sensitivity, specificity, and AUC scores.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize evaluation metrics.
        
        Args:
            num_classes (int): Number of classes (default: 2 for binary classification)
        """
        self.num_classes = num_classes
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Prediction probabilities
            average (str): Averaging method for multi-class metrics
            
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        # Ensure inputs are numpy arrays
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        if not isinstance(y_prob, np.ndarray):
            y_prob = np.array(y_prob)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average)
        metrics['f1'] = f1_score(y_true, y_pred, average=average)
        
        # AUC metrics
        if self.num_classes == 2:
            # Binary classification
            try:
                # Check for NaN values in probabilities
                if np.isnan(y_prob[:, 1]).any():
                    print("Warning: NaN values detected in y_prob[:, 1], setting AUC to 0.5")
                    metrics['auc'] = 0.5
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                
                if np.isnan(y_prob[:, 1]).any():
                    metrics['average_precision'] = 0.5
                else:
                    metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
            except Exception as e:
                print(f"Warning: AUC calculation failed: {e}, setting to 0.5")
                metrics['auc'] = 0.5
                metrics['average_precision'] = 0.5
        else:
            # Multi-class classification
            try:
                if np.isnan(y_prob).any():
                    print("Warning: NaN values detected in y_prob, setting AUC to 0.5")
                    metrics['auc'] = 0.5
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob, average=average, multi_class='ovo')
                
                if np.isnan(y_prob).any():
                    metrics['average_precision'] = 0.5
                else:
                    metrics['average_precision'] = average_precision_score(y_true, y_prob, average=average)
            except Exception as e:
                print(f"Warning: AUC calculation failed: {e}, setting to 0.5")
                metrics['auc'] = 0.5
                metrics['average_precision'] = 0.5
        
        # Per-class metrics - temporarily disabled
        # if self.num_classes > 2:
        #     per_class_metrics = self._compute_per_class_metrics(y_true, y_pred, y_prob)
        #     metrics.update(per_class_metrics)
        
        # Sensitivity and specificity
        sensitivity, specificity = self._compute_sensitivity_specificity(y_true, y_pred)
        metrics['sensitivity'] = sensitivity
        metrics['specificity'] = specificity
        
        # Confusion matrix analysis - temporarily disabled due to array comparison issues
        # cm_metrics = self._analyze_confusion_matrix(y_true, y_pred)
        # metrics.update(cm_metrics)
        
        return metrics
    
    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Compute per-class metrics."""
        metrics = {}
        
        for i in range(self.num_classes):
            # Per-class precision, recall, F1
            precision = precision_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
            recall = recall_score(y_true, y_pred, labels=[i], average='micro')
            f1 = f1_score(y_true, y_pred, labels=[i], average='micro')
            
            metrics[f'class_{i}_precision'] = precision
            metrics[f'class_{i}_recall'] = recall
            metrics[f'class_{i}_f1'] = f1
            
            # Per-class AUC
            try:
                if self.num_classes == 2:
                    auc = roc_auc_score(y_true, y_prob[:, i])
                else:
                    # One-vs-rest AUC
                    y_true_binary = (y_true == i).astype(int)
                    auc = roc_auc_score(y_true_binary, y_prob[:, i])
                
                metrics[f'class_{i}_auc'] = auc
            except ValueError:
                # Handle case where AUC cannot be computed (e.g., only one class present)
                metrics[f'class_{i}_auc'] = 0.0
        
        return metrics
    
    def _compute_sensitivity_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, float]:
        """Compute sensitivity and specificity."""
        if self.num_classes == 2:
            # Binary classification
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Multi-class classification - compute macro average
            sensitivities = []
            specificities = []
            
            for i in range(self.num_classes):
                # One-vs-rest approach
                y_true_binary = (y_true == i).astype(int)
                y_pred_binary = (y_pred == i).astype(int)
                
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    
                    sensitivities.append(sensitivity)
                    specificities.append(specificity)
                except ValueError:
                    # Handle case where confusion matrix cannot be computed
                    sensitivities.append(0.0)
                    specificities.append(0.0)
            
            sensitivity = np.mean(sensitivities)
            specificity = np.mean(specificities)
        
        return sensitivity, specificity
    
    def _analyze_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Analyze confusion matrix and compute related metrics."""
        cm = confusion_matrix(y_true, y_pred)
        metrics = {}
        
        # Overall accuracy from confusion matrix
        metrics['cm_accuracy'] = np.trace(cm) / np.sum(cm)
        
        # Per-class accuracy
        for i in range(self.num_classes):
            class_accuracy = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0.0
            metrics[f'class_{i}_accuracy'] = class_accuracy
        
        # Cohen's Kappa
        metrics['cohens_kappa'] = self._compute_cohens_kappa(cm)
        
        # Matthews Correlation Coefficient
        metrics['mcc'] = self._compute_mcc(cm)
        
        return metrics
    
    def _compute_cohens_kappa(self, cm: np.ndarray) -> float:
        """Compute Cohen's Kappa coefficient."""
        n = np.sum(cm)
        po = np.trace(cm) / n  # Observed agreement
        
        # Expected agreement
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (n * n)
        
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
        return kappa
    
    def _compute_mcc(self, cm: np.ndarray) -> float:
        """Compute Matthews Correlation Coefficient."""
        if self.num_classes == 2:
            # Binary classification
            tn, fp, fn, tp = cm.ravel()
            mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            return mcc if not np.isnan(mcc) else 0.0
        else:
            # Multi-class MCC
            t_k = np.trace(cm)
            p_k = np.sum(cm, axis=0)
            t_k = np.sum(cm, axis=1)
            c = np.sum(cm)
            
            numerator = c * t_k - np.dot(p_k, t_k)
            denominator = np.sqrt((c * c - np.dot(p_k, p_k)) * (c * c - np.dot(t_k, t_k)))
            
            mcc = numerator / denominator if denominator != 0 else 0.0
            return mcc if not np.isnan(mcc) else 0.0
    
    def compute_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute ROC curves for all classes."""
        roc_data = {}
        
        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
            roc_data['fpr'] = fpr
            roc_data['tpr'] = tpr
            roc_data['thresholds'] = thresholds
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i in range(self.num_classes):
                fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_data[f'class_{i}_fpr'] = fpr
                roc_data[f'class_{i}_tpr'] = tpr
                roc_data[f'class_{i}_thresholds'] = thresholds
        
        return roc_data
    
    def compute_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute precision-recall curves for all classes."""
        pr_data = {}
        
        if self.num_classes == 2:
            # Binary classification
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])
            pr_data['precision'] = precision
            pr_data['recall'] = recall
            pr_data['thresholds'] = thresholds
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            for i in range(self.num_classes):
                precision, recall, thresholds = precision_recall_curve(
                    y_true_bin[:, i], y_prob[:, i]
                )
                pr_data[f'class_{i}_precision'] = precision
                pr_data[f'class_{i}_recall'] = recall
                pr_data[f'class_{i}_thresholds'] = thresholds
        
        return pr_data
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """Generate detailed classification report."""
        if target_names is None:
            target_names = [f'Class {i}' for i in range(self.num_classes)]
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def compute_bootstrap_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for metrics."""
        n_samples = len(y_true)
        bootstrap_metrics = []
        
        # Bootstrap sampling
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_prob_boot = y_prob[indices]
            
            # Compute metrics for this bootstrap sample
            metrics = self.compute_metrics(y_true_boot, y_pred_boot, y_prob_boot)
            bootstrap_metrics.append(metrics)
        
        # Compute confidence intervals
        confidence_intervals = {}
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        for metric_name in bootstrap_metrics[0].keys():
            values = [m[metric_name] for m in bootstrap_metrics]
            confidence_intervals[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'lower': np.percentile(values, lower_percentile),
                'upper': np.percentile(values, upper_percentile)
            }
        
        return confidence_intervals
