"""Deep Neural Network implementation.

Provides a flexible implementation of deep neural networks with
customizable layers, activation functions, and training procedures.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pickle

from ..base_model import BaseModel
from .activations import ActivationFunctions
from .optimizers import Optimizers


class DeepNeuralNetwork(BaseModel):
    """Deep Neural Network implementation.
    
    A flexible deep neural network with customizable architecture,
    activation functions, and optimization algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the deep neural network.
        
        Args:
            config: Configuration dictionary containing:
                - layers: List of layer sizes
                - activation: Activation function name
                - optimizer: Optimizer configuration
                - learning_rate: Learning rate
                - regularization: Regularization parameters
        """
        super().__init__(config)
        
        self.layers = config.get("layers", [784, 128, 64, 10])
        self.activation_name = config.get("activation", "relu")
        self.optimizer_config = config.get("optimizer", {"type": "adam"})
        self.learning_rate = config.get("learning_rate", 0.001)
        self.regularization = config.get("regularization", {"l2": 0.01})
        
        # Initialize activation function
        self.activation = ActivationFunctions()
        
        # Initialize optimizer
        self.optimizer = Optimizers(self.optimizer_config)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        # Training history
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
    
    def _initialize_parameters(self) -> None:
        """Initialize network parameters using Xavier initialization."""
        np.random.seed(42)  # For reproducibility
        
        for i in range(len(self.layers) - 1):
            # Xavier initialization
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            weight = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i + 1]))
            bias = np.zeros((1, self.layers[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network.
        
        Args:
            inputs: Input data of shape (batch_size, input_dim)
            
        Returns:
            Network output of shape (batch_size, output_dim)
        """
        self.activations = [inputs]
        self.z_values = []
        
        current_input = inputs
        
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            z = np.dot(current_input, weight) + bias
            self.z_values.append(z)
            
            # Apply activation function (except for output layer)
            if i < len(self.weights) - 1:
                current_input = self.activation.forward(z, self.activation_name)
            else:
                # Softmax for output layer (classification)
                current_input = self.activation.forward(z, "softmax")
            
            self.activations.append(current_input)
        
        return current_input
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[np.ndarray]:
        """Backward pass to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            List of gradients for weights and biases
        """
        m = y_true.shape[0]  # Batch size
        
        # Initialize gradients
        weight_gradients = []
        bias_gradients = []
        
        # Output layer error
        delta = y_pred - y_true
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            weight_grad = np.dot(self.activations[i].T, delta) / m
            bias_grad = np.mean(delta, axis=0, keepdims=True)
            
            # Add L2 regularization
            if "l2" in self.regularization:
                weight_grad += self.regularization["l2"] * self.weights[i]
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            # Compute delta for previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * \
                       self.activation.backward(self.z_values[i-1], self.activation_name)
        
        return weight_gradients, bias_gradients
    
    def train_step(self, batch: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of training metrics
        """
        inputs, targets = batch
        
        # Forward pass
        predictions = self.forward(inputs)
        
        # Compute loss
        loss = self._compute_loss(targets, predictions)
        
        # Backward pass
        weight_grads, bias_grads = self.backward(targets, predictions)
        
        # Update parameters
        self.optimizer.update_parameters(
            self.weights, self.biases, weight_grads, bias_grads, self.learning_rate
        )
        
        # Compute accuracy
        accuracy = self._compute_accuracy(targets, predictions)
        
        return {"loss": loss, "accuracy": accuracy}
    
    def validate_step(self, batch: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Single validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of validation metrics
        """
        inputs, targets = batch
        
        # Forward pass only
        predictions = self.forward(inputs)
        
        # Compute metrics
        loss = self._compute_loss(targets, predictions)
        accuracy = self._compute_accuracy(targets, predictions)
        
        return {"val_loss": loss, "val_accuracy": accuracy}
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cross-entropy loss.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Cross-entropy loss
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        # Add regularization
        if "l2" in self.regularization:
            l2_penalty = self.regularization["l2"] * \
                        sum(np.sum(w**2) for w in self.weights)
            loss += l2_penalty
        
        return loss
    
    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Classification accuracy
        """
        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(true_labels == pred_labels)
    
    def _save_weights(self, path: Path) -> None:
        """Save model weights to disk.
        
        Args:
            path: Directory path to save weights
        """
        weights_path = path / "weights.pkl"
        with open(weights_path, "wb") as f:
            pickle.dump({
                "weights": self.weights,
                "biases": self.biases,
                "training_history": self.training_history
            }, f)
    
    def _load_weights(self, path: Path) -> None:
        """Load model weights from disk.
        
        Args:
            path: Directory path to load weights from
        """
        weights_path = path / "weights.pkl"
        if weights_path.exists():
            with open(weights_path, "rb") as f:
                data = pickle.load(f)
                self.weights = data["weights"]
                self.biases = data["biases"]
                self.training_history = data.get("training_history", self.training_history)
                self.is_trained = True
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            inputs: Input data
            
        Returns:
            Predicted class probabilities
        """
        return self.forward(inputs)
    
    def get_layer_outputs(self, inputs: np.ndarray, layer_idx: int) -> np.ndarray:
        """Get outputs from a specific layer.
        
        Args:
            inputs: Input data
            layer_idx: Index of the layer
            
        Returns:
            Layer outputs
        """
        self.forward(inputs)
        if 0 <= layer_idx < len(self.activations):
            return self.activations[layer_idx]
        else:
            raise ValueError(f"Layer index {layer_idx} out of range")