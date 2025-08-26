"""Tensor operations utilities.

Provides common tensor operations and mathematical functions
for neural network computations.
"""

import numpy as np
from typing import Union, Tuple, List, Optional


class TensorOperations:
    """Collection of tensor operations and mathematical utilities."""
    
    @staticmethod
    def normalize(x: np.ndarray, axis: Optional[int] = None, epsilon: float = 1e-8) -> np.ndarray:
        """Normalize tensor along specified axis.
        
        Args:
            x: Input tensor
            axis: Axis along which to normalize
            epsilon: Small value to prevent division by zero
            
        Returns:
            Normalized tensor
        """
        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        return (x - mean) / (std + epsilon)
    
    @staticmethod
    def standardize(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Standardize tensor to zero mean and unit variance.
        
        Args:
            x: Input tensor
            axis: Axis along which to standardize
            
        Returns:
            Standardized tensor
        """
        return TensorOperations.normalize(x, axis)
    
    @staticmethod
    def min_max_scale(x: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Scale tensor to specified range.
        
        Args:
            x: Input tensor
            feature_range: Target range (min, max)
            
        Returns:
            Scaled tensor
        """
        x_min = np.min(x)
        x_max = np.max(x)
        
        if x_max - x_min == 0:
            return np.full_like(x, feature_range[0])
        
        x_scaled = (x - x_min) / (x_max - x_min)
        return x_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    @staticmethod
    def one_hot_encode(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
        """Convert labels to one-hot encoding.
        
        Args:
            labels: Integer labels
            num_classes: Number of classes (inferred if None)
            
        Returns:
            One-hot encoded labels
        """
        if num_classes is None:
            num_classes = int(np.max(labels)) + 1
        
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels.astype(int)] = 1
        return one_hot
    
    @staticmethod
    def pad_sequences(sequences: List[np.ndarray], maxlen: Optional[int] = None, 
                     padding: str = 'pre', truncating: str = 'pre', 
                     value: float = 0.0) -> np.ndarray:
        """Pad sequences to the same length.
        
        Args:
            sequences: List of sequences
            maxlen: Maximum length (use longest if None)
            padding: 'pre' or 'post' padding
            truncating: 'pre' or 'post' truncating
            value: Padding value
            
        Returns:
            Padded sequences array
        """
        if maxlen is None:
            maxlen = max(len(seq) for seq in sequences)
        
        padded = np.full((len(sequences), maxlen), value, dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            
            if len(seq) > maxlen:
                if truncating == 'pre':
                    seq = seq[-maxlen:]
                else:
                    seq = seq[:maxlen]
            
            if padding == 'pre':
                padded[i, -len(seq):] = seq
            else:
                padded[i, :len(seq)] = seq
        
        return padded
    
    @staticmethod
    def batch_generator(X: np.ndarray, y: Optional[np.ndarray] = None, 
                       batch_size: int = 32, shuffle: bool = True) -> Tuple[np.ndarray, ...]:
        """Generate batches from data.
        
        Args:
            X: Input data
            y: Target data (optional)
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of data
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            if y is not None:
                yield X[batch_indices], y[batch_indices]
            else:
                yield X[batch_indices]
    
    @staticmethod
    def train_test_split(X: np.ndarray, y: Optional[np.ndarray] = None, 
                        test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """Split data into training and testing sets.
        
        Args:
            X: Input data
            y: Target data (optional)
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Split data (X_train, X_test, y_train, y_test) or (X_train, X_test)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        
        if y is not None:
            y_train, y_test = y[train_indices], y[test_indices]
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(a - b)
    
    @staticmethod
    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Manhattan distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Manhattan distance
        """
        return np.sum(np.abs(a - b))
    
    @staticmethod
    def apply_dropout(x: np.ndarray, dropout_rate: float, training: bool = True) -> np.ndarray:
        """Apply dropout to tensor.
        
        Args:
            x: Input tensor
            dropout_rate: Dropout probability
            training: Whether in training mode
            
        Returns:
            Tensor with dropout applied
        """
        if not training or dropout_rate == 0:
            return x
        
        keep_prob = 1 - dropout_rate
        mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
        return x * mask
    
    @staticmethod
    def gradient_clipping(gradients: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
        """Apply gradient clipping to prevent exploding gradients.
        
        Args:
            gradients: List of gradient arrays
            max_norm: Maximum gradient norm
            
        Returns:
            Clipped gradients
        """
        total_norm = np.sqrt(sum(np.sum(grad**2) for grad in gradients))
        
        if total_norm > max_norm:
            clip_coef = max_norm / total_norm
            gradients = [grad * clip_coef for grad in gradients]
        
        return gradients
    
    @staticmethod
    def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
        """Compute moving average of values.
        
        Args:
            values: Input values
            window_size: Size of moving window
            
        Returns:
            Moving averages
        """
        if window_size <= 0:
            return values
        
        cumsum = np.cumsum(np.insert(values, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    @staticmethod
    def exponential_moving_average(values: np.ndarray, alpha: float) -> np.ndarray:
        """Compute exponential moving average.
        
        Args:
            values: Input values
            alpha: Smoothing factor (0 < alpha <= 1)
            
        Returns:
            Exponential moving averages
        """
        ema = np.zeros_like(values)
        ema[0] = values[0]
        
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
        
        return ema