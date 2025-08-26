"""Evaluation metrics for machine learning models.

Provides comprehensive metrics for classification, regression,
and other ML tasks.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from collections import Counter


class Metrics:
    """Collection of evaluation metrics for ML models."""
    
    # Classification Metrics
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Union[float, np.ndarray]:
        """Calculate precision score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('macro', 'micro', 'weighted', None)
            
        Returns:
            Precision score(s)
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            
            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)
            precisions.append(precision)
        
        precisions = np.array(precisions)
        
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
            fp_total = np.sum([np.sum((y_true != cls) & (y_pred == cls)) for cls in classes])
            return tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
        elif average == 'weighted':
            weights = [np.sum(y_true == cls) for cls in classes]
            return np.average(precisions, weights=weights)
        else:
            return precisions
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Union[float, np.ndarray]:
        """Calculate recall score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('macro', 'micro', 'weighted', None)
            
        Returns:
            Recall score(s)
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            recalls.append(recall)
        
        recalls = np.array(recalls)
        
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            tp_total = np.sum([np.sum((y_true == cls) & (y_pred == cls)) for cls in classes])
            fn_total = np.sum([np.sum((y_true == cls) & (y_pred != cls)) for cls in classes])
            return tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
        elif average == 'weighted':
            weights = [np.sum(y_true == cls) for cls in classes]
            return np.average(recalls, weights=weights)
        else:
            return recalls
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Union[float, np.ndarray]:
        """Calculate F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging strategy ('macro', 'micro', 'weighted', None)
            
        Returns:
            F1 score(s)
        """
        precision_scores = Metrics.precision(y_true, y_pred, average=None)
        recall_scores = Metrics.recall(y_true, y_pred, average=None)
        
        f1_scores = []
        for p, r in zip(precision_scores, recall_scores):
            if p + r == 0:
                f1 = 0.0
            else:
                f1 = 2 * (p * r) / (p + r)
            f1_scores.append(f1)
        
        f1_scores = np.array(f1_scores)
        
        if average == 'macro':
            return np.mean(f1_scores)
        elif average == 'micro':
            p_micro = Metrics.precision(y_true, y_pred, average='micro')
            r_micro = Metrics.recall(y_true, y_pred, average='micro')
            return 2 * (p_micro * r_micro) / (p_micro + r_micro) if p_micro + r_micro > 0 else 0.0
        elif average == 'weighted':
            classes = np.unique(np.concatenate([y_true, y_pred]))
            weights = [np.sum(y_true == cls) for cls in classes]
            return np.average(f1_scores, weights=weights)
        else:
            return f1_scores
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        # Create mapping from class to index
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1
        
        return cm
    
    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report dictionary
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        report = {}
        
        # Per-class metrics
        precision_scores = Metrics.precision(y_true, y_pred, average=None)
        recall_scores = Metrics.recall(y_true, y_pred, average=None)
        f1_scores = Metrics.f1_score(y_true, y_pred, average=None)
        
        for i, cls in enumerate(classes):
            support = np.sum(y_true == cls)
            report[str(cls)] = {
                'precision': precision_scores[i],
                'recall': recall_scores[i],
                'f1-score': f1_scores[i],
                'support': support
            }
        
        # Overall metrics
        report['accuracy'] = Metrics.accuracy(y_true, y_pred)
        report['macro avg'] = {
            'precision': Metrics.precision(y_true, y_pred, average='macro'),
            'recall': Metrics.recall(y_true, y_pred, average='macro'),
            'f1-score': Metrics.f1_score(y_true, y_pred, average='macro'),
            'support': len(y_true)
        }
        report['weighted avg'] = {
            'precision': Metrics.precision(y_true, y_pred, average='weighted'),
            'recall': Metrics.recall(y_true, y_pred, average='weighted'),
            'f1-score': Metrics.f1_score(y_true, y_pred, average='weighted'),
            'support': len(y_true)
        }
        
        return report
    
    # Regression Metrics
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(Metrics.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R² value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value
        """
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Information Theory Metrics
    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
        """Calculate cross-entropy loss.
        
        Args:
            y_true: True probabilities (one-hot encoded)
            y_pred: Predicted probabilities
            epsilon: Small value to prevent log(0)
            
        Returns:
            Cross-entropy loss
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-15) -> float:
        """Calculate Kullback-Leibler divergence.
        
        Args:
            p: True distribution
            q: Predicted distribution
            epsilon: Small value to prevent log(0)
            
        Returns:
            KL divergence
        """
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return np.sum(p * np.log(p / q))
    
    # Ranking Metrics
    @staticmethod
    def top_k_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 5) -> float:
        """Calculate top-k accuracy.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = 0
        
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    @staticmethod
    def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Generate comprehensive regression report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Regression metrics dictionary
        """
        return {
            'mse': Metrics.mean_squared_error(y_true, y_pred),
            'rmse': Metrics.root_mean_squared_error(y_true, y_pred),
            'mae': Metrics.mean_absolute_error(y_true, y_pred),
            'r2': Metrics.r2_score(y_true, y_pred),
            'mape': Metrics.mean_absolute_percentage_error(y_true, y_pred)
        }