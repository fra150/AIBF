"""Attention mechanisms for transformer architectures.

Implements multi-head attention, self-attention, and related
components for transformer-based models.
"""

import numpy as np
from typing import Optional, Tuple
from ..base_model import BaseModel


class SelfAttention:
    """Self-attention mechanism implementation."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """Initialize self-attention.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.dropout = dropout
        
        # Initialize weight matrices
        self.W_q = self._xavier_init((d_model, d_model))
        self.W_k = self._xavier_init((d_model, d_model))
        self.W_v = self._xavier_init((d_model, d_model))
        
        # Bias terms
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        
        # Cache for backward pass
        self.cache = {}
    
    def _xavier_init(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Xavier initialization for weights."""
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, 
                x: np.ndarray, 
                mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass of self-attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len)
            training: Whether in training mode
            
        Returns:
            Attention output (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        Q = np.dot(x, self.W_q) + self.b_q  # (batch_size, seq_len, d_model)
        K = np.dot(x, self.W_k) + self.b_k  # (batch_size, seq_len, d_model)
        V = np.dot(x, self.W_v) + self.b_v  # (batch_size, seq_len, d_model)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_model)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self._softmax(scores)
        
        # Apply dropout during training
        if training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        # Cache for backward pass
        self.cache = {
            'x': x,
            'Q': Q,
            'K': K,
            'V': V,
            'scores': scores,
            'attention_weights': attention_weights,
            'mask': mask
        }
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of self-attention.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        x = self.cache['x']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        attention_weights = self.cache['attention_weights']
        
        batch_size, seq_len, d_model = x.shape
        
        # Gradient w.r.t. V
        grad_V = np.matmul(attention_weights.transpose(0, 2, 1), grad_output)
        
        # Gradient w.r.t. attention weights
        grad_attention = np.matmul(grad_output, V.transpose(0, 2, 1))
        
        # Gradient w.r.t. scores (through softmax)
        grad_scores = self._softmax_backward(attention_weights, grad_attention)
        
        # Gradient w.r.t. Q and K
        grad_Q = np.matmul(grad_scores, K) / np.sqrt(d_model)
        grad_K = np.matmul(grad_scores.transpose(0, 2, 1), Q) / np.sqrt(d_model)
        
        # Gradient w.r.t. weights and biases
        self.grad_W_q = np.sum(np.matmul(x.transpose(0, 2, 1), grad_Q), axis=0)
        self.grad_W_k = np.sum(np.matmul(x.transpose(0, 2, 1), grad_K), axis=0)
        self.grad_W_v = np.sum(np.matmul(x.transpose(0, 2, 1), grad_V), axis=0)
        
        self.grad_b_q = np.sum(grad_Q, axis=(0, 1))
        self.grad_b_k = np.sum(grad_K, axis=(0, 1))
        self.grad_b_v = np.sum(grad_V, axis=(0, 1))
        
        # Gradient w.r.t. input
        grad_x = (np.matmul(grad_Q, self.W_q.T) + 
                 np.matmul(grad_K, self.W_k.T) + 
                 np.matmul(grad_V, self.W_v.T))
        
        return grad_x
    
    def _softmax_backward(self, softmax_output: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through softmax."""
        # Compute Jacobian of softmax
        s = softmax_output
        grad_input = s * (grad_output - np.sum(s * grad_output, axis=-1, keepdims=True))
        return grad_input


class MultiHeadAttention:
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Weight matrices for all heads
        self.W_q = self._xavier_init((d_model, d_model))
        self.W_k = self._xavier_init((d_model, d_model))
        self.W_v = self._xavier_init((d_model, d_model))
        self.W_o = self._xavier_init((d_model, d_model))
        
        # Bias terms
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)
        
        # Cache for backward pass
        self.cache = {}
    
    def _xavier_init(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Xavier initialization for weights."""
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, 
                query: np.ndarray,
                key: np.ndarray, 
                value: np.ndarray,
                mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len_q, d_model)
            key: Key tensor (batch_size, seq_len_k, d_model)
            value: Value tensor (batch_size, seq_len_v, d_model)
            mask: Attention mask
            training: Whether in training mode
            
        Returns:
            Multi-head attention output
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Linear transformations
        Q = np.dot(query, self.W_q) + self.b_q
        K = np.dot(key, self.W_k) + self.b_k
        V = np.dot(value, self.W_v) + self.b_v
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention for each head
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask, training)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len_q, self.d_model
        )
        
        # Final linear transformation
        output = np.dot(attention_output, self.W_o) + self.b_o
        
        # Cache for backward pass
        self.cache = {
            'query': query,
            'key': key,
            'value': value,
            'Q': Q,
            'K': K,
            'V': V,
            'attention_output': attention_output,
            'mask': mask
        }
        
        return output
    
    def _scaled_dot_product_attention(self, 
                                     Q: np.ndarray,
                                     K: np.ndarray, 
                                     V: np.ndarray,
                                     mask: Optional[np.ndarray] = None,
                                     training: bool = True) -> np.ndarray:
        """Scaled dot-product attention.
        
        Args:
            Q: Query tensor (batch_size, num_heads, seq_len_q, d_k)
            K: Key tensor (batch_size, num_heads, seq_len_k, d_k)
            V: Value tensor (batch_size, num_heads, seq_len_v, d_k)
            mask: Attention mask
            training: Whether in training mode
            
        Returns:
            Attention output
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multiple heads
            if mask.ndim == 3:  # (batch_size, seq_len_q, seq_len_k)
                mask = mask[:, np.newaxis, :, :]  # (batch_size, 1, seq_len_q, seq_len_k)
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self._softmax(scores)
        
        # Apply dropout during training
        if training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        # Store attention weights for visualization/analysis
        self.attention_weights = attention_weights
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def get_attention_weights(self) -> np.ndarray:
        """Get attention weights for visualization.
        
        Returns:
            Attention weights (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        return getattr(self, 'attention_weights', None)
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass of multi-head attention.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradients w.r.t. query, key, value
        """
        # This is a simplified backward pass
        # In practice, you would implement the full gradient computation
        # through all the matrix operations
        
        query = self.cache['query']
        key = self.cache['key']
        value = self.cache['value']
        
        # Gradient w.r.t. output projection
        grad_attention_output = np.dot(grad_output, self.W_o.T)
        self.grad_W_o = np.sum(np.matmul(
            self.cache['attention_output'].transpose(0, 2, 1), 
            grad_output
        ), axis=0)
        self.grad_b_o = np.sum(grad_output, axis=(0, 1))
        
        # For simplicity, return identity gradients
        # Full implementation would compute exact gradients
        grad_query = grad_attention_output
        grad_key = np.zeros_like(key)
        grad_value = np.zeros_like(value)
        
        return grad_query, grad_key, grad_value
    
    def get_parameters(self) -> dict:
        """Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'W_q': self.W_q,
            'W_k': self.W_k,
            'W_v': self.W_v,
            'W_o': self.W_o,
            'b_q': self.b_q,
            'b_k': self.b_k,
            'b_v': self.b_v,
            'b_o': self.b_o
        }
    
    def set_parameters(self, params: dict):
        """Set model parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.W_q = params['W_q']
        self.W_k = params['W_k']
        self.W_v = params['W_v']
        self.W_o = params['W_o']
        self.b_q = params['b_q']
        self.b_k = params['b_k']
        self.b_v = params['b_v']
        self.b_o = params['b_o']