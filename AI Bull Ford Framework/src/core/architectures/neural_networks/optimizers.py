"""Optimization algorithms for neural network training.

Provides implementations of various optimization algorithms including
SGD, Adam, RMSprop, and AdaGrad.
"""

import numpy as np
from typing import List, Dict, Any


class Optimizers:
    """Collection of optimization algorithms for neural network training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize optimizer.
        
        Args:
            config: Optimizer configuration containing type and parameters
        """
        self.config = config
        self.optimizer_type = config.get("type", "sgd")
        
        # Common parameters
        self.beta1 = config.get("beta1", 0.9)
        self.beta2 = config.get("beta2", 0.999)
        self.epsilon = config.get("epsilon", 1e-8)
        self.decay = config.get("decay", 0.0)
        
        # Initialize optimizer state
        self.t = 0  # Time step
        self.m_weights = []  # First moment for weights
        self.v_weights = []  # Second moment for weights
        self.m_biases = []   # First moment for biases
        self.v_biases = []   # Second moment for biases
        
        # RMSprop specific
        self.cache_weights = []
        self.cache_biases = []
        
        self.initialized = False
    
    def _initialize_state(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        """Initialize optimizer state variables.
        
        Args:
            weights: List of weight matrices
            biases: List of bias vectors
        """
        if not self.initialized:
            # Initialize moments for Adam
            self.m_weights = [np.zeros_like(w) for w in weights]
            self.v_weights = [np.zeros_like(w) for w in weights]
            self.m_biases = [np.zeros_like(b) for b in biases]
            self.v_biases = [np.zeros_like(b) for b in biases]
            
            # Initialize cache for RMSprop
            self.cache_weights = [np.zeros_like(w) for w in weights]
            self.cache_biases = [np.zeros_like(b) for b in biases]
            
            self.initialized = True
    
    def update_parameters(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
        learning_rate: float
    ) -> None:
        """Update model parameters using the specified optimizer.
        
        Args:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_grads: List of weight gradients
            bias_grads: List of bias gradients
            learning_rate: Learning rate
        """
        self._initialize_state(weights, biases)
        self.t += 1
        
        # Apply learning rate decay
        if self.decay > 0:
            learning_rate = learning_rate / (1 + self.decay * self.t)
        
        if self.optimizer_type == "sgd":
            self._sgd_update(weights, biases, weight_grads, bias_grads, learning_rate)
        elif self.optimizer_type == "momentum":
            self._momentum_update(weights, biases, weight_grads, bias_grads, learning_rate)
        elif self.optimizer_type == "adam":
            self._adam_update(weights, biases, weight_grads, bias_grads, learning_rate)
        elif self.optimizer_type == "rmsprop":
            self._rmsprop_update(weights, biases, weight_grads, bias_grads, learning_rate)
        elif self.optimizer_type == "adagrad":
            self._adagrad_update(weights, biases, weight_grads, bias_grads, learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def _sgd_update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
        learning_rate: float
    ) -> None:
        """Stochastic Gradient Descent update.
        
        Args:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_grads: List of weight gradients
            bias_grads: List of bias gradients
            learning_rate: Learning rate
        """
        for i in range(len(weights)):
            weights[i] -= learning_rate * weight_grads[i]
            biases[i] -= learning_rate * bias_grads[i]
    
    def _momentum_update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
        learning_rate: float
    ) -> None:
        """SGD with Momentum update.
        
        Args:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_grads: List of weight gradients
            bias_grads: List of bias gradients
            learning_rate: Learning rate
        """
        for i in range(len(weights)):
            # Update momentum for weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_grads[i]
            weights[i] -= learning_rate * self.m_weights[i]
            
            # Update momentum for biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_grads[i]
            biases[i] -= learning_rate * self.m_biases[i]
    
    def _adam_update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
        learning_rate: float
    ) -> None:
        """Adam optimizer update.
        
        Args:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_grads: List of weight gradients
            bias_grads: List of bias gradients
            learning_rate: Learning rate
        """
        for i in range(len(weights)):
            # Update biased first moment estimate for weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_grads[i]
            # Update biased second raw moment estimate for weights
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_grads[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.t)
            
            # Update weights
            weights[i] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            
            # Update biased first moment estimate for biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_grads[i]
            # Update biased second raw moment estimate for biases
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (bias_grads[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            # Update biases
            biases[i] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    def _rmsprop_update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
        learning_rate: float
    ) -> None:
        """RMSprop optimizer update.
        
        Args:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_grads: List of weight gradients
            bias_grads: List of bias gradients
            learning_rate: Learning rate
        """
        for i in range(len(weights)):
            # Update cache for weights
            self.cache_weights[i] = self.beta2 * self.cache_weights[i] + (1 - self.beta2) * (weight_grads[i] ** 2)
            weights[i] -= learning_rate * weight_grads[i] / (np.sqrt(self.cache_weights[i]) + self.epsilon)
            
            # Update cache for biases
            self.cache_biases[i] = self.beta2 * self.cache_biases[i] + (1 - self.beta2) * (bias_grads[i] ** 2)
            biases[i] -= learning_rate * bias_grads[i] / (np.sqrt(self.cache_biases[i]) + self.epsilon)
    
    def _adagrad_update(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
        learning_rate: float
    ) -> None:
        """AdaGrad optimizer update.
        
        Args:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_grads: List of weight gradients
            bias_grads: List of bias gradients
            learning_rate: Learning rate
        """
        for i in range(len(weights)):
            # Accumulate squared gradients for weights
            self.cache_weights[i] += weight_grads[i] ** 2
            weights[i] -= learning_rate * weight_grads[i] / (np.sqrt(self.cache_weights[i]) + self.epsilon)
            
            # Accumulate squared gradients for biases
            self.cache_biases[i] += bias_grads[i] ** 2
            biases[i] -= learning_rate * bias_grads[i] / (np.sqrt(self.cache_biases[i]) + self.epsilon)
    
    def get_state(self) -> Dict[str, Any]:
        """Get optimizer state for saving.
        
        Returns:
            Dictionary containing optimizer state
        """
        return {
            "optimizer_type": self.optimizer_type,
            "t": self.t,
            "m_weights": self.m_weights,
            "v_weights": self.v_weights,
            "m_biases": self.m_biases,
            "v_biases": self.v_biases,
            "cache_weights": self.cache_weights,
            "cache_biases": self.cache_biases,
            "initialized": self.initialized
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set optimizer state from loaded data.
        
        Args:
            state: Dictionary containing optimizer state
        """
        self.optimizer_type = state.get("optimizer_type", self.optimizer_type)
        self.t = state.get("t", 0)
        self.m_weights = state.get("m_weights", [])
        self.v_weights = state.get("v_weights", [])
        self.m_biases = state.get("m_biases", [])
        self.v_biases = state.get("v_biases", [])
        self.cache_weights = state.get("cache_weights", [])
        self.cache_biases = state.get("cache_biases", [])
        self.initialized = state.get("initialized", False)
    
    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        """Get list of available optimizers.
        
        Returns:
            List of optimizer names
        """
        return ["sgd", "momentum", "adam", "rmsprop", "adagrad"]