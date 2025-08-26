"""Positional encoding for transformer architectures.

Implements various positional encoding schemes including
sinusoidal, learned, and relative positional encodings.
"""

import numpy as np
from typing import Optional


class PositionalEncoding:
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.max_length = max_length
        self.dropout = dropout
        
        # Create positional encoding matrix
        self.pe = self._create_positional_encoding()
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding matrix.
        
        Returns:
            Positional encoding matrix (max_length, d_model)
        """
        pe = np.zeros((self.max_length, self.d_model))
        position = np.arange(0, self.max_length).reshape(-1, 1)
        
        # Create division term for sinusoidal pattern
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         -(np.log(10000.0) / self.d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cosine to odd indices
        if self.d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            training: Whether in training mode
            
        Returns:
            Input with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_length}")
        
        if d_model != self.d_model:
            raise ValueError(f"Model dimension {d_model} doesn't match expected {self.d_model}")
        
        # Add positional encoding
        x_with_pe = x + self.pe[:seq_len, :]
        
        # Apply dropout during training
        if training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, x_with_pe.shape)
            x_with_pe = x_with_pe * dropout_mask / (1 - self.dropout)
        
        return x_with_pe
    
    def get_encoding(self, seq_len: int) -> np.ndarray:
        """Get positional encoding for a specific sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encoding (seq_len, d_model)
        """
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_length}")
        
        return self.pe[:seq_len, :]


class LearnedPositionalEncoding:
    """Learned positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        """Initialize learned positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.max_length = max_length
        self.dropout = dropout
        
        # Initialize learnable positional embeddings
        self.pe = self._xavier_init((max_length, d_model))
        
        # Gradient storage
        self.grad_pe = np.zeros_like(self.pe)
    
    def _xavier_init(self, shape: tuple) -> np.ndarray:
        """Xavier initialization for positional embeddings."""
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Add learned positional encoding to input embeddings.
        
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            training: Whether in training mode
            
        Returns:
            Input with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_length}")
        
        if d_model != self.d_model:
            raise ValueError(f"Model dimension {d_model} doesn't match expected {self.d_model}")
        
        # Add positional encoding
        x_with_pe = x + self.pe[:seq_len, :]
        
        # Apply dropout during training
        if training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, x_with_pe.shape)
            x_with_pe = x_with_pe * dropout_mask / (1 - self.dropout)
        
        return x_with_pe
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for learned positional encoding.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        batch_size, seq_len, d_model = grad_output.shape
        
        # Accumulate gradients for positional embeddings
        self.grad_pe[:seq_len, :] += np.sum(grad_output, axis=0)
        
        # Return gradient w.r.t. input (same as grad_output)
        return grad_output
    
    def get_parameters(self) -> dict:
        """Get learnable parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {'pe': self.pe}
    
    def set_parameters(self, params: dict):
        """Set learnable parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.pe = params['pe']


class RelativePositionalEncoding:
    """Relative positional encoding for transformer models.
    
    Based on "Self-Attention with Relative Position Representations"
    by Shaw et al. (2018).
    """
    
    def __init__(self, d_model: int, max_relative_distance: int = 128):
        """Initialize relative positional encoding.
        
        Args:
            d_model: Model dimension
            max_relative_distance: Maximum relative distance to consider
        """
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        
        # Create relative position embeddings
        # We need embeddings for distances from -max_distance to +max_distance
        vocab_size = 2 * max_relative_distance + 1
        self.relative_embeddings = self._xavier_init((vocab_size, d_model))
    
    def _xavier_init(self, shape: tuple) -> np.ndarray:
        """Xavier initialization for embeddings."""
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def _get_relative_positions(self, seq_len: int) -> np.ndarray:
        """Get relative position matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position matrix (seq_len, seq_len)
        """
        positions = np.arange(seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to maximum distance
        relative_positions = np.clip(
            relative_positions, 
            -self.max_relative_distance, 
            self.max_relative_distance
        )
        
        # Shift to make all values positive (for indexing)
        relative_positions += self.max_relative_distance
        
        return relative_positions
    
    def forward(self, seq_len: int) -> np.ndarray:
        """Get relative positional encodings for attention.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position encodings (seq_len, seq_len, d_model)
        """
        relative_positions = self._get_relative_positions(seq_len)
        
        # Get embeddings for relative positions
        relative_encodings = self.relative_embeddings[relative_positions]
        
        return relative_encodings
    
    def get_parameters(self) -> dict:
        """Get learnable parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {'relative_embeddings': self.relative_embeddings}
    
    def set_parameters(self, params: dict):
        """Set learnable parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self.relative_embeddings = params['relative_embeddings']


class RotaryPositionalEncoding:
    """Rotary Positional Encoding (RoPE).
    
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    by Su et al. (2021).
    """
    
    def __init__(self, d_model: int, max_length: int = 5000, base: float = 10000.0):
        """Initialize rotary positional encoding.
        
        Args:
            d_model: Model dimension (must be even)
            max_length: Maximum sequence length
            base: Base for frequency computation
        """
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for rotary positional encoding")
        
        self.d_model = d_model
        self.max_length = max_length
        self.base = base
        
        # Precompute frequency matrix
        self.freqs = self._compute_frequencies()
    
    def _compute_frequencies(self) -> np.ndarray:
        """Compute frequency matrix for rotary encoding.
        
        Returns:
            Frequency matrix (max_length, d_model // 2)
        """
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.d_model, 2) / self.d_model))
        
        # Create position indices
        positions = np.arange(self.max_length)
        
        # Compute frequencies for each position
        freqs = np.outer(positions, inv_freq)
        
        return freqs
    
    def _apply_rotary_encoding(self, x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Apply rotary encoding to input tensor.
        
        Args:
            x: Input tensor (..., seq_len, d_model)
            freqs: Frequency matrix (seq_len, d_model // 2)
            
        Returns:
            Rotary encoded tensor
        """
        # Split into even and odd dimensions
        x_even = x[..., 0::2]  # Even dimensions
        x_odd = x[..., 1::2]   # Odd dimensions
        
        # Compute cos and sin
        cos_freqs = np.cos(freqs)
        sin_freqs = np.sin(freqs)
        
        # Apply rotation
        x_rotated_even = x_even * cos_freqs - x_odd * sin_freqs
        x_rotated_odd = x_even * sin_freqs + x_odd * cos_freqs
        
        # Interleave back
        x_rotated = np.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        
        return x_rotated
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply rotary positional encoding.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Rotary encoded tensor
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_length}")
        
        if d_model != self.d_model:
            raise ValueError(f"Model dimension {d_model} doesn't match expected {self.d_model}")
        
        # Get frequencies for current sequence length
        freqs = self.freqs[:seq_len, :]
        
        # Apply rotary encoding
        return self._apply_rotary_encoding(x, freqs)
    
    def apply_to_query_key(self, query: np.ndarray, key: np.ndarray) -> tuple:
        """Apply rotary encoding to query and key tensors.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tuple of (rotary_query, rotary_key)
        """
        rotary_query = self.forward(query)
        rotary_key = self.forward(key)
        
        return rotary_query, rotary_key