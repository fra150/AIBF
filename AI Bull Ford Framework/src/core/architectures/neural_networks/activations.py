"""Activation functions for neural networks.

Provides implementations of common activation functions and their derivatives.
"""

import numpy as np
from typing import Union


class ActivationFunctions:
    """Collection of activation functions and their derivatives."""
    
    def forward(self, x: np.ndarray, activation_type: str) -> np.ndarray:
        """Apply activation function.
        
        Args:
            x: Input array
            activation_type: Type of activation function
            
        Returns:
            Activated output
        """
        if activation_type == "relu":
            return self.relu(x)
        elif activation_type == "sigmoid":
            return self.sigmoid(x)
        elif activation_type == "tanh":
            return self.tanh(x)
        elif activation_type == "softmax":
            return self.softmax(x)
        elif activation_type == "leaky_relu":
            return self.leaky_relu(x)
        elif activation_type == "elu":
            return self.elu(x)
        elif activation_type == "swish":
            return self.swish(x)
        elif activation_type == "gelu":
            return self.gelu(x)
        elif activation_type == "linear":
            return x
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    def backward(self, x: np.ndarray, activation_type: str) -> np.ndarray:
        """Compute derivative of activation function.
        
        Args:
            x: Input array (pre-activation)
            activation_type: Type of activation function
            
        Returns:
            Derivative of activation function
        """
        if activation_type == "relu":
            return self.relu_derivative(x)
        elif activation_type == "sigmoid":
            return self.sigmoid_derivative(x)
        elif activation_type == "tanh":
            return self.tanh_derivative(x)
        elif activation_type == "leaky_relu":
            return self.leaky_relu_derivative(x)
        elif activation_type == "elu":
            return self.elu_derivative(x)
        elif activation_type == "swish":
            return self.swish_derivative(x)
        elif activation_type == "gelu":
            return self.gelu_derivative(x)
        elif activation_type == "linear":
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function.
        
        Args:
            x: Input array
            
        Returns:
            ReLU output: max(0, x)
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function.
        
        Args:
            x: Input array
            
        Returns:
            ReLU derivative
        """
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function.
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid output: 1 / (1 + exp(-x))
        """
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function.
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid derivative
        """
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function.
        
        Args:
            x: Input array
            
        Returns:
            Tanh output
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function.
        
        Args:
            x: Input array
            
        Returns:
            Tanh derivative
        """
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function.
        
        Args:
            x: Input array of shape (batch_size, num_classes)
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function.
        
        Args:
            x: Input array
            alpha: Slope for negative values
            
        Returns:
            Leaky ReLU output
        """
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU function.
        
        Args:
            x: Input array
            alpha: Slope for negative values
            
        Returns:
            Leaky ReLU derivative
        """
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Exponential Linear Unit (ELU) activation function.
        
        Args:
            x: Input array
            alpha: Scale for negative values
            
        Returns:
            ELU output
        """
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Derivative of ELU function.
        
        Args:
            x: Input array
            alpha: Scale for negative values
            
        Returns:
            ELU derivative
        """
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish activation function.
        
        Args:
            x: Input array
            
        Returns:
            Swish output: x * sigmoid(x)
        """
        return x * ActivationFunctions.sigmoid(x)
    
    @staticmethod
    def swish_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of Swish function.
        
        Args:
            x: Input array
            
        Returns:
            Swish derivative
        """
        sigmoid_x = ActivationFunctions.sigmoid(x)
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit (GELU) activation function.
        
        Args:
            x: Input array
            
        Returns:
            GELU output
        """
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def gelu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of GELU function (approximation).
        
        Args:
            x: Input array
            
        Returns:
            GELU derivative
        """
        # Approximation of GELU derivative
        tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        tanh_val = np.tanh(tanh_arg)
        sech2_val = 1 - tanh_val**2
        
        return 0.5 * (1 + tanh_val) + 0.5 * x * sech2_val * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
    
    @classmethod
    def get_available_activations(cls) -> list:
        """Get list of available activation functions.
        
        Returns:
            List of activation function names
        """
        return [
            "relu", "sigmoid", "tanh", "softmax", "leaky_relu",
            "elu", "swish", "gelu", "linear"
        ]