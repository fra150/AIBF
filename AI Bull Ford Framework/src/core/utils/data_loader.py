"""Data loading and preprocessing utilities.

Provides efficient data loading, batching, and preprocessing
functionalities for machine learning workflows.
"""

import numpy as np
from typing import Iterator, Tuple, Optional, List, Union, Dict, Any
import random
from abc import ABC, abstractmethod


class Dataset(ABC):
    """Abstract base class for datasets."""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, target)
        """
        pass


class ArrayDataset(Dataset):
    """Dataset wrapper for numpy arrays."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """Initialize dataset with features and targets.
        
        Args:
            features: Input features array
            targets: Target values array
        """
        if len(features) != len(targets):
            raise ValueError("Features and targets must have the same length")
        
        self.features = features
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.features[idx], self.targets[idx]


class DataLoader:
    """Efficient data loading with batching and shuffling capabilities."""
    
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 seed: Optional[int] = None):
        """Initialize DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Size of each batch
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches.
        
        Yields:
            Batches of (features, targets)
        """
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # Collect batch data
            batch_features = []
            batch_targets = []
            
            for idx in batch_indices:
                features, targets = self.dataset[idx]
                batch_features.append(features)
                batch_targets.append(targets)
            
            yield np.array(batch_features), np.array(batch_targets)
    
    def get_batch(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a specific batch by index.
        
        Args:
            batch_idx: Index of the batch to retrieve
            
        Returns:
            Batch of (features, targets)
        """
        if batch_idx >= len(self):
            raise IndexError(f"Batch index {batch_idx} out of range")
        
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        
        batch_features = []
        batch_targets = []
        
        for idx in range(start_idx, end_idx):
            features, targets = self.dataset[idx]
            batch_features.append(features)
            batch_targets.append(targets)
        
        return np.array(batch_features), np.array(batch_targets)


class DataPreprocessor:
    """Data preprocessing utilities."""
    
    @staticmethod
    def normalize_features(X: np.ndarray, 
                          method: str = 'standard',
                          axis: int = 0,
                          epsilon: float = 1e-8) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Normalize features using various methods.
        
        Args:
            X: Input features
            method: Normalization method ('standard', 'minmax', 'robust')
            axis: Axis along which to normalize
            epsilon: Small value to prevent division by zero
            
        Returns:
            Normalized features and normalization parameters
        """
        if method == 'standard':
            mean = np.mean(X, axis=axis, keepdims=True)
            std = np.std(X, axis=axis, keepdims=True) + epsilon
            X_norm = (X - mean) / std
            params = {'mean': mean, 'std': std}
            
        elif method == 'minmax':
            min_val = np.min(X, axis=axis, keepdims=True)
            max_val = np.max(X, axis=axis, keepdims=True)
            range_val = max_val - min_val + epsilon
            X_norm = (X - min_val) / range_val
            params = {'min': min_val, 'max': max_val}
            
        elif method == 'robust':
            median = np.median(X, axis=axis, keepdims=True)
            mad = np.median(np.abs(X - median), axis=axis, keepdims=True) + epsilon
            X_norm = (X - median) / mad
            params = {'median': median, 'mad': mad}
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return X_norm, params
    
    @staticmethod
    def apply_normalization(X: np.ndarray, 
                           params: Dict[str, np.ndarray], 
                           method: str = 'standard') -> np.ndarray:
        """Apply previously computed normalization parameters.
        
        Args:
            X: Input features
            params: Normalization parameters
            method: Normalization method used
            
        Returns:
            Normalized features
        """
        if method == 'standard':
            return (X - params['mean']) / params['std']
        elif method == 'minmax':
            return (X - params['min']) / (params['max'] - params['min'])
        elif method == 'robust':
            return (X - params['median']) / params['mad']
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def encode_categorical(y: np.ndarray, 
                          num_classes: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """One-hot encode categorical labels.
        
        Args:
            y: Categorical labels
            num_classes: Number of classes (auto-detected if None)
            
        Returns:
            One-hot encoded labels and encoding parameters
        """
        unique_classes = np.unique(y)
        if num_classes is None:
            num_classes = len(unique_classes)
        
        # Create class to index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        # One-hot encode
        y_encoded = np.zeros((len(y), num_classes))
        for i, label in enumerate(y):
            if label in class_to_idx:
                y_encoded[i, class_to_idx[label]] = 1
        
        params = {
            'classes': unique_classes,
            'class_to_idx': class_to_idx,
            'num_classes': num_classes
        }
        
        return y_encoded, params
    
    @staticmethod
    def decode_categorical(y_encoded: np.ndarray, 
                          params: Dict[str, Any]) -> np.ndarray:
        """Decode one-hot encoded labels back to categorical.
        
        Args:
            y_encoded: One-hot encoded labels
            params: Encoding parameters
            
        Returns:
            Categorical labels
        """
        idx_to_class = {idx: cls for cls, idx in params['class_to_idx'].items()}
        y_decoded = []
        
        for encoded_label in y_encoded:
            class_idx = np.argmax(encoded_label)
            y_decoded.append(idx_to_class[class_idx])
        
        return np.array(y_decoded)
    
    @staticmethod
    def train_test_split(X: np.ndarray, 
                        y: np.ndarray, 
                        test_size: float = 0.2,
                        random_state: Optional[int] = None,
                        stratify: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets.
        
        Args:
            X: Features
            y: Targets
            test_size: Proportion of data for testing
            random_state: Random seed
            stratify: Whether to maintain class distribution
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        if stratify:
            # Stratified split
            unique_classes, class_counts = np.unique(y, return_counts=True)
            
            train_indices = []
            test_indices = []
            
            for cls in unique_classes:
                cls_indices = np.where(y == cls)[0]
                np.random.shuffle(cls_indices)
                
                n_test_cls = int(len(cls_indices) * test_size)
                test_indices.extend(cls_indices[:n_test_cls])
                train_indices.extend(cls_indices[n_test_cls:])
            
            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)
            
        else:
            # Random split
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    @staticmethod
    def augment_data(X: np.ndarray, 
                    y: np.ndarray,
                    noise_factor: float = 0.1,
                    rotation_range: float = 0.0,
                    flip_probability: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation techniques.
        
        Args:
            X: Input features
            y: Target labels
            noise_factor: Amount of noise to add
            rotation_range: Range of rotation (for image data)
            flip_probability: Probability of flipping
            
        Returns:
            Augmented features and labels
        """
        X_aug = X.copy()
        y_aug = y.copy()
        
        # Add noise
        if noise_factor > 0:
            noise = np.random.normal(0, noise_factor, X.shape)
            X_aug = X_aug + noise
        
        # Random flipping (for appropriate data types)
        if flip_probability > 0:
            flip_mask = np.random.random(len(X)) < flip_probability
            if len(X.shape) > 1:  # Assume last dimension can be flipped
                X_aug[flip_mask] = np.flip(X_aug[flip_mask], axis=-1)
        
        return X_aug, y_aug


class CrossValidator:
    """Cross-validation utilities."""
    
    @staticmethod
    def k_fold_split(X: np.ndarray, 
                    y: np.ndarray, 
                    k: int = 5,
                    shuffle: bool = True,
                    random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate k-fold cross-validation splits.
        
        Args:
            X: Features
            y: Targets
            k: Number of folds
            shuffle: Whether to shuffle before splitting
            random_state: Random seed
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        fold_size = n_samples // k
        splits = []
        
        for i in range(k):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k - 1 else n_samples
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            splits.append((train_indices, val_indices))
        
        return splits
    
    @staticmethod
    def stratified_k_fold_split(X: np.ndarray, 
                               y: np.ndarray, 
                               k: int = 5,
                               random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified k-fold cross-validation splits.
        
        Args:
            X: Features
            y: Targets
            k: Number of folds
            random_state: Random seed
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        unique_classes = np.unique(y)
        splits = [[] for _ in range(k)]
        
        # Split each class separately
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            
            fold_size = len(cls_indices) // k
            
            for i in range(k):
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < k - 1 else len(cls_indices)
                splits[i].extend(cls_indices[start_idx:end_idx])
        
        # Convert to train/val splits
        result_splits = []
        for i in range(k):
            val_indices = np.array(splits[i])
            train_indices = np.concatenate([np.array(splits[j]) for j in range(k) if j != i])
            result_splits.append((train_indices, val_indices))
        
        return result_splits