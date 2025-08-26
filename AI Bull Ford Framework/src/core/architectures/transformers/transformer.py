"""Transformer architecture implementation.

Implements complete transformer blocks, encoder, and decoder
architectures with layer normalization and feed-forward networks.
"""

import numpy as np
from typing import Optional, List
from .attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding
from ..base_model import BaseModel


class LayerNormalization:
    """Layer normalization implementation."""
    
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        """Initialize layer normalization.
        
        Args:
            d_model: Model dimension
            epsilon: Small value for numerical stability
        """
        self.d_model = d_model
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of layer normalization.
        
        Args:
            x: Input tensor (..., d_model)
            
        Returns:
            Normalized tensor
        """
        # Compute mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        # Cache for backward pass
        self.cache = {
            'x': x,
            'mean': mean,
            'variance': variance,
            'x_normalized': x_normalized
        }
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of layer normalization.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        x = self.cache['x']
        mean = self.cache['mean']
        variance = self.cache['variance']
        x_normalized = self.cache['x_normalized']
        
        # Gradients w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * x_normalized, axis=tuple(range(grad_output.ndim - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))
        
        # Gradient w.r.t. input
        N = x.shape[-1]
        grad_x_normalized = grad_output * self.gamma
        
        grad_variance = np.sum(grad_x_normalized * (x - mean) * 
                              (-0.5) * (variance + self.epsilon) ** (-1.5), 
                              axis=-1, keepdims=True)
        
        grad_mean = np.sum(grad_x_normalized * (-1.0) / np.sqrt(variance + self.epsilon), 
                          axis=-1, keepdims=True) + \
                   grad_variance * np.sum(-2.0 * (x - mean), axis=-1, keepdims=True) / N
        
        grad_x = grad_x_normalized / np.sqrt(variance + self.epsilon) + \
                grad_variance * 2.0 * (x - mean) / N + \
                grad_mean / N
        
        return grad_x


class FeedForward:
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Weight matrices
        self.W1 = self._xavier_init((d_model, d_ff))
        self.W2 = self._xavier_init((d_ff, d_model))
        
        # Bias terms
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
        
        # Cache for backward pass
        self.cache = {}
    
    def _xavier_init(self, shape: tuple) -> np.ndarray:
        """Xavier initialization for weights."""
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of feed-forward network.
        
        Args:
            x: Input tensor (..., d_model)
            training: Whether in training mode
            
        Returns:
            Output tensor (..., d_model)
        """
        # First linear transformation + ReLU
        hidden = np.dot(x, self.W1) + self.b1
        hidden = np.maximum(0, hidden)  # ReLU activation
        
        # Apply dropout during training
        if training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, hidden.shape)
            hidden = hidden * dropout_mask / (1 - self.dropout)
        
        # Second linear transformation
        output = np.dot(hidden, self.W2) + self.b2
        
        # Cache for backward pass
        self.cache = {
            'x': x,
            'hidden': hidden,
            'dropout_mask': dropout_mask if training and self.dropout > 0 else None
        }
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of feed-forward network.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input
        """
        x = self.cache['x']
        hidden = self.cache['hidden']
        dropout_mask = self.cache.get('dropout_mask')
        
        # Gradient w.r.t. second layer
        self.grad_W2 = np.sum(np.matmul(hidden[..., :, np.newaxis], 
                                       grad_output[..., np.newaxis, :]), axis=tuple(range(hidden.ndim - 1)))
        self.grad_b2 = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))
        
        # Gradient w.r.t. hidden layer
        grad_hidden = np.dot(grad_output, self.W2.T)
        
        # Apply dropout mask if used
        if dropout_mask is not None:
            grad_hidden = grad_hidden * dropout_mask / (1 - self.dropout)
        
        # Gradient through ReLU
        grad_hidden = grad_hidden * (hidden > 0)
        
        # Gradient w.r.t. first layer
        self.grad_W1 = np.sum(np.matmul(x[..., :, np.newaxis], 
                                       grad_hidden[..., np.newaxis, :]), axis=tuple(range(x.ndim - 1)))
        self.grad_b1 = np.sum(grad_hidden, axis=tuple(range(grad_hidden.ndim - 1)))
        
        # Gradient w.r.t. input
        grad_x = np.dot(grad_hidden, self.W1.T)
        
        return grad_x


class TransformerBlock:
    """Single transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Components
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
    
    def forward(self, 
                x: np.ndarray, 
                mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass of transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask
            training: Whether in training mode
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention.forward(x, x, x, mask, training)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward.forward(x, training)
        x = self.norm2.forward(x + ff_output)
        
        return x
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get attention weights from the self-attention layer.
        
        Returns:
            Attention weights if available
        """
        return self.self_attention.get_attention_weights()


class TransformerEncoder(BaseModel):
    """Transformer encoder with multiple transformer blocks."""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int, 
                 num_heads: int, 
                 num_layers: int,
                 d_ff: int,
                 max_length: int = 5000,
                 dropout: float = 0.1):
        """Initialize transformer encoder.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_length = max_length
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = self._xavier_init((vocab_size, d_model))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff, dropout) 
                      for _ in range(num_layers)]
        
        # Final layer normalization
        self.final_norm = LayerNormalization(d_model)
    
    def _xavier_init(self, shape: tuple) -> np.ndarray:
        """Xavier initialization for embeddings."""
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(self, 
                input_ids: np.ndarray, 
                mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass of transformer encoder.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            mask: Attention mask (batch_size, seq_len, seq_len)
            training: Whether in training mode
            
        Returns:
            Encoded representations (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding lookup
        x = self.embedding[input_ids]  # (batch_size, seq_len, d_model)
        
        # Scale embeddings
        x = x * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding.forward(x, training)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask, training)
        
        # Final layer normalization
        x = self.final_norm.forward(x)
        
        return x
    
    def get_attention_weights(self) -> List[Optional[np.ndarray]]:
        """Get attention weights from all layers.
        
        Returns:
            List of attention weights for each layer
        """
        return [block.get_attention_weights() for block in self.blocks]


class TransformerDecoder(BaseModel):
    """Transformer decoder with masked self-attention and cross-attention."""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int, 
                 num_heads: int, 
                 num_layers: int,
                 d_ff: int,
                 max_length: int = 5000,
                 dropout: float = 0.1):
        """Initialize transformer decoder.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_length = max_length
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = self._xavier_init((vocab_size, d_model))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        # Decoder blocks
        self.blocks = [self._create_decoder_block() for _ in range(num_layers)]
        
        # Final layer normalization
        self.final_norm = LayerNormalization(d_model)
        
        # Output projection
        self.output_projection = self._xavier_init((d_model, vocab_size))
        self.output_bias = np.zeros(vocab_size)
    
    def _xavier_init(self, shape: tuple) -> np.ndarray:
        """Xavier initialization for weights."""
        limit = np.sqrt(6.0 / sum(shape))
        return np.random.uniform(-limit, limit, shape)
    
    def _create_decoder_block(self) -> dict:
        """Create a decoder block with masked self-attention and cross-attention.
        
        Returns:
            Dictionary containing decoder block components
        """
        return {
            'masked_self_attention': MultiHeadAttention(self.d_model, self.num_heads, self.dropout),
            'cross_attention': MultiHeadAttention(self.d_model, self.num_heads, self.dropout),
            'feed_forward': FeedForward(self.d_model, self.d_ff, self.dropout),
            'norm1': LayerNormalization(self.d_model),
            'norm2': LayerNormalization(self.d_model),
            'norm3': LayerNormalization(self.d_model)
        }
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal (lower triangular) mask for decoder.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Causal mask (seq_len, seq_len)
        """
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask
    
    def forward(self, 
                input_ids: np.ndarray,
                encoder_output: Optional[np.ndarray] = None,
                encoder_mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass of transformer decoder.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            encoder_output: Encoder output for cross-attention
            encoder_mask: Encoder attention mask
            training: Whether in training mode
            
        Returns:
            Decoder output logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding lookup
        x = self.embedding[input_ids]  # (batch_size, seq_len, d_model)
        
        # Scale embeddings
        x = x * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding.forward(x, training)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len)
        causal_mask = np.broadcast_to(causal_mask[np.newaxis, :, :], (batch_size, seq_len, seq_len))
        
        # Pass through decoder blocks
        for block in self.blocks:
            # Masked self-attention
            self_attn_output = block['masked_self_attention'].forward(
                x, x, x, causal_mask, training
            )
            x = block['norm1'].forward(x + self_attn_output)
            
            # Cross-attention (if encoder output is provided)
            if encoder_output is not None:
                cross_attn_output = block['cross_attention'].forward(
                    x, encoder_output, encoder_output, encoder_mask, training
                )
                x = block['norm2'].forward(x + cross_attn_output)
            
            # Feed-forward
            ff_output = block['feed_forward'].forward(x, training)
            x = block['norm3'].forward(x + ff_output)
        
        # Final layer normalization
        x = self.final_norm.forward(x)
        
        # Output projection
        logits = np.dot(x, self.output_projection) + self.output_bias
        
        return logits
    
    def generate(self, 
                 input_ids: np.ndarray,
                 max_length: int,
                 encoder_output: Optional[np.ndarray] = None,
                 encoder_mask: Optional[np.ndarray] = None,
                 temperature: float = 1.0) -> np.ndarray:
        """Generate sequences using the decoder.
        
        Args:
            input_ids: Initial input token IDs (batch_size, initial_seq_len)
            max_length: Maximum generation length
            encoder_output: Encoder output for cross-attention
            encoder_mask: Encoder attention mask
            temperature: Sampling temperature
            
        Returns:
            Generated sequences (batch_size, max_length)
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.copy()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            logits = self.forward(generated, encoder_output, encoder_mask, training=False)
            
            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax
            probs = self._softmax(next_token_logits)
            
            # Sample next token
            next_tokens = self._sample(probs)
            
            # Append to generated sequence
            generated = np.concatenate([generated, next_tokens[:, np.newaxis]], axis=1)
        
        return generated
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _sample(self, probs: np.ndarray) -> np.ndarray:
        """Sample from probability distribution.
        
        Args:
            probs: Probability distribution (batch_size, vocab_size)
            
        Returns:
            Sampled token IDs (batch_size,)
        """
        batch_size, vocab_size = probs.shape
        samples = []
        
        for i in range(batch_size):
            sample = np.random.choice(vocab_size, p=probs[i])
            samples.append(sample)
        
        return np.array(samples)