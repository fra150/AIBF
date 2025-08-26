"""Modality fusion module for AI Bull Ford.

This module provides advanced fusion techniques for combining multiple modalities:
- Early fusion (feature-level)
- Late fusion (decision-level)
- Intermediate fusion (hybrid)
- Attention-based fusion
- Transformer-based fusion
- Adaptive fusion strategies
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from .cross_modal import ModalityData, ModalityType, CrossModalConfig


class FusionStrategy(Enum):
    """Fusion strategies for combining modalities."""
    EARLY = "early"
    LATE = "late"
    INTERMEDIATE = "intermediate"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    GATED = "gated"


class FusionMethod(Enum):
    """Methods for fusion computation."""
    CONCATENATION = "concatenation"
    ADDITION = "addition"
    MULTIPLICATION = "multiplication"
    WEIGHTED_SUM = "weighted_sum"
    MAX_POOLING = "max_pooling"
    AVERAGE_POOLING = "average_pooling"
    ATTENTION_POOLING = "attention_pooling"
    BILINEAR = "bilinear"
    TENSOR_FUSION = "tensor_fusion"


@dataclass
class FusionConfig:
    """Configuration for modality fusion."""
    strategy: FusionStrategy = FusionStrategy.ATTENTION
    method: FusionMethod = FusionMethod.WEIGHTED_SUM
    embedding_dim: int = 768
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    temperature: float = 1.0
    device: str = "cpu"
    use_layer_norm: bool = True
    use_residual: bool = True


@dataclass
class FusionResult:
    """Result from modality fusion."""
    fused_embedding: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    modality_weights: Optional[Dict[ModalityType, float]] = None
    fusion_confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFusionModule(ABC):
    """Abstract base class for fusion modules."""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def fuse(self, modalities: List[ModalityData]) -> FusionResult:
        """Fuse multiple modalities."""
        pass
    
    def _validate_modalities(self, modalities: List[ModalityData]) -> None:
        """Validate input modalities."""
        if not modalities:
            raise ValueError("No modalities provided for fusion")
        
        for modality in modalities:
            if modality.embedding is None:
                raise ValueError(f"Modality {modality.modality_type} missing embedding")


class EarlyFusion(BaseFusionModule):
    """Early fusion at feature level."""
    
    def fuse(self, modalities: List[ModalityData]) -> FusionResult:
        """Perform early fusion by concatenating features."""
        start_time = datetime.now()
        
        try:
            self._validate_modalities(modalities)
            
            # Extract embeddings
            embeddings = [mod.embedding for mod in modalities]
            
            if self.config.method == FusionMethod.CONCATENATION:
                fused_embedding = np.concatenate(embeddings, axis=-1)
            elif self.config.method == FusionMethod.ADDITION:
                # Ensure same dimensions
                min_dim = min(emb.shape[-1] for emb in embeddings)
                truncated_embeddings = [emb[:min_dim] for emb in embeddings]
                fused_embedding = np.sum(truncated_embeddings, axis=0)
            elif self.config.method == FusionMethod.AVERAGE_POOLING:
                min_dim = min(emb.shape[-1] for emb in embeddings)
                truncated_embeddings = [emb[:min_dim] for emb in embeddings]
                fused_embedding = np.mean(truncated_embeddings, axis=0)
            else:
                # Default to concatenation
                fused_embedding = np.concatenate(embeddings, axis=-1)
            
            # Project to target dimension if needed
            if fused_embedding.shape[-1] != self.config.embedding_dim:
                # Simple projection (in real implementation, use learned projection)
                if fused_embedding.shape[-1] > self.config.embedding_dim:
                    fused_embedding = fused_embedding[:self.config.embedding_dim]
                else:
                    padding = np.zeros(self.config.embedding_dim - fused_embedding.shape[-1])
                    fused_embedding = np.concatenate([fused_embedding, padding])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return FusionResult(
                fused_embedding=fused_embedding,
                fusion_confidence=0.8,  # Placeholder
                processing_time=processing_time,
                metadata={"strategy": "early", "method": self.config.method.value}
            )
        except Exception as e:
            self.logger.error(f"Early fusion failed: {e}")
            raise


class LateFusion(BaseFusionModule):
    """Late fusion at decision level."""
    
    def fuse(self, modalities: List[ModalityData]) -> FusionResult:
        """Perform late fusion by combining decisions."""
        start_time = datetime.now()
        
        try:
            self._validate_modalities(modalities)
            
            # Extract embeddings (representing decisions/predictions)
            embeddings = [mod.embedding for mod in modalities]
            
            # Compute modality weights based on confidence or other metrics
            modality_weights = {}
            weights = []
            for i, modality in enumerate(modalities):
                # Use metadata confidence if available, otherwise equal weights
                confidence = modality.metadata.get('confidence', 1.0)
                weight = confidence / len(modalities)
                modality_weights[modality.modality_type] = weight
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Weighted combination
            if self.config.method == FusionMethod.WEIGHTED_SUM:
                fused_embedding = np.zeros_like(embeddings[0])
                for i, embedding in enumerate(embeddings):
                    fused_embedding += weights[i] * embedding
            elif self.config.method == FusionMethod.MAX_POOLING:
                fused_embedding = np.maximum.reduce(embeddings)
            else:
                # Default weighted sum
                fused_embedding = np.zeros_like(embeddings[0])
                for i, embedding in enumerate(embeddings):
                    fused_embedding += weights[i] * embedding
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return FusionResult(
                fused_embedding=fused_embedding,
                modality_weights=modality_weights,
                fusion_confidence=np.mean(weights),
                processing_time=processing_time,
                metadata={"strategy": "late", "method": self.config.method.value}
            )
        except Exception as e:
            self.logger.error(f"Late fusion failed: {e}")
            raise


class AttentionFusion(BaseFusionModule):
    """Attention-based fusion."""
    
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for attention fusion")
        
        # Initialize attention mechanism (placeholder)
        self.attention_weights = None
    
    def fuse(self, modalities: List[ModalityData]) -> FusionResult:
        """Perform attention-based fusion."""
        start_time = datetime.now()
        
        try:
            self._validate_modalities(modalities)
            
            # Extract embeddings
            embeddings = [mod.embedding for mod in modalities]
            n_modalities = len(embeddings)
            
            # Compute attention weights
            attention_weights = self._compute_attention(embeddings)
            
            # Apply attention weights
            fused_embedding = np.zeros_like(embeddings[0])
            for i, embedding in enumerate(embeddings):
                fused_embedding += attention_weights[i] * embedding
            
            # Compute modality weights from attention
            modality_weights = {}
            for i, modality in enumerate(modalities):
                modality_weights[modality.modality_type] = float(attention_weights[i])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return FusionResult(
                fused_embedding=fused_embedding,
                attention_weights=attention_weights,
                modality_weights=modality_weights,
                fusion_confidence=np.max(attention_weights),
                processing_time=processing_time,
                metadata={"strategy": "attention", "num_heads": self.config.num_heads}
            )
        except Exception as e:
            self.logger.error(f"Attention fusion failed: {e}")
            raise
    
    def _compute_attention(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute attention weights for embeddings."""
        n_modalities = len(embeddings)
        
        # Simple attention mechanism (placeholder)
        # In real implementation, would use learned attention
        
        # Compute pairwise similarities
        similarities = np.zeros((n_modalities, n_modalities))
        for i in range(n_modalities):
            for j in range(n_modalities):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities[i, j] = sim
        
        # Compute attention weights as average similarity
        attention_weights = np.mean(similarities, axis=1)
        
        # Apply temperature and softmax
        attention_weights = attention_weights / self.config.temperature
        attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))
        
        return attention_weights


class TransformerFusion(BaseFusionModule):
    """Transformer-based fusion."""
    
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for transformer fusion")
        
        # Initialize transformer layers (placeholder)
        self.transformer_layers = None
    
    def fuse(self, modalities: List[ModalityData]) -> FusionResult:
        """Perform transformer-based fusion."""
        start_time = datetime.now()
        
        try:
            self._validate_modalities(modalities)
            
            # Extract embeddings
            embeddings = [mod.embedding for mod in modalities]
            
            # Stack embeddings as sequence
            sequence = np.stack(embeddings, axis=0)  # [seq_len, embed_dim]
            
            # Apply transformer (placeholder - would use actual transformer)
            # For now, use multi-head attention simulation
            fused_embedding = self._apply_transformer(sequence)
            
            # Compute attention weights (placeholder)
            attention_weights = np.ones(len(embeddings)) / len(embeddings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return FusionResult(
                fused_embedding=fused_embedding,
                attention_weights=attention_weights,
                fusion_confidence=0.9,  # Placeholder
                processing_time=processing_time,
                metadata={
                    "strategy": "transformer",
                    "num_layers": self.config.num_layers,
                    "num_heads": self.config.num_heads
                }
            )
        except Exception as e:
            self.logger.error(f"Transformer fusion failed: {e}")
            raise
    
    def _apply_transformer(self, sequence: np.ndarray) -> np.ndarray:
        """Apply transformer to sequence."""
        # Placeholder transformer application
        # In real implementation, would use actual transformer layers
        
        # Simple aggregation for now
        fused_embedding = np.mean(sequence, axis=0)
        
        return fused_embedding


class AdaptiveFusion(BaseFusionModule):
    """Adaptive fusion that selects strategy based on input."""
    
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        
        # Initialize different fusion modules
        self.early_fusion = EarlyFusion(config)
        self.late_fusion = LateFusion(config)
        self.attention_fusion = AttentionFusion(config) if TORCH_AVAILABLE else None
    
    def fuse(self, modalities: List[ModalityData]) -> FusionResult:
        """Perform adaptive fusion by selecting best strategy."""
        start_time = datetime.now()
        
        try:
            self._validate_modalities(modalities)
            
            # Analyze modalities to select fusion strategy
            strategy = self._select_strategy(modalities)
            
            # Apply selected strategy
            if strategy == FusionStrategy.EARLY:
                result = self.early_fusion.fuse(modalities)
            elif strategy == FusionStrategy.LATE:
                result = self.late_fusion.fuse(modalities)
            elif strategy == FusionStrategy.ATTENTION and self.attention_fusion:
                result = self.attention_fusion.fuse(modalities)
            else:
                # Default to early fusion
                result = self.early_fusion.fuse(modalities)
            
            # Update metadata
            result.metadata["adaptive_strategy"] = strategy.value
            result.metadata["selection_time"] = (datetime.now() - start_time).total_seconds()
            
            return result
        except Exception as e:
            self.logger.error(f"Adaptive fusion failed: {e}")
            raise
    
    def _select_strategy(self, modalities: List[ModalityData]) -> FusionStrategy:
        """Select fusion strategy based on modalities."""
        # Simple heuristics for strategy selection
        n_modalities = len(modalities)
        
        # Check modality types
        modality_types = {mod.modality_type for mod in modalities}
        
        # If many modalities, use attention
        if n_modalities > 3 and TORCH_AVAILABLE:
            return FusionStrategy.ATTENTION
        
        # If mixed modalities (text + vision), use late fusion
        if ModalityType.TEXT in modality_types and ModalityType.IMAGE in modality_types:
            return FusionStrategy.LATE
        
        # Default to early fusion
        return FusionStrategy.EARLY


class ModalityFusionEngine:
    """Main engine for modality fusion."""
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize fusion modules
        self.fusion_modules = {
            FusionStrategy.EARLY: EarlyFusion(self.config),
            FusionStrategy.LATE: LateFusion(self.config),
            FusionStrategy.ADAPTIVE: AdaptiveFusion(self.config)
        }
        
        if TORCH_AVAILABLE:
            self.fusion_modules[FusionStrategy.ATTENTION] = AttentionFusion(self.config)
            self.fusion_modules[FusionStrategy.TRANSFORMER] = TransformerFusion(self.config)
    
    def fuse(self, modalities: List[ModalityData], strategy: Optional[FusionStrategy] = None) -> FusionResult:
        """Fuse modalities using specified or default strategy."""
        try:
            if strategy is None:
                strategy = self.config.strategy
            
            if strategy not in self.fusion_modules:
                self.logger.warning(f"Strategy {strategy} not available, using adaptive")
                strategy = FusionStrategy.ADAPTIVE
            
            fusion_module = self.fusion_modules[strategy]
            return fusion_module.fuse(modalities)
        except Exception as e:
            self.logger.error(f"Fusion failed: {e}")
            raise
    
    def batch_fuse(self, modality_batches: List[List[ModalityData]], strategy: Optional[FusionStrategy] = None) -> List[FusionResult]:
        """Fuse multiple batches of modalities."""
        try:
            results = []
            for batch in modality_batches:
                result = self.fuse(batch, strategy)
                results.append(result)
            return results
        except Exception as e:
            self.logger.error(f"Batch fusion failed: {e}")
            raise
    
    async def async_fuse(self, modalities: List[ModalityData], strategy: Optional[FusionStrategy] = None) -> FusionResult:
        """Asynchronously fuse modalities."""
        # Run fusion in thread pool for CPU-bound operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fuse, modalities, strategy)


# Global fusion engine instance
_fusion_engine: Optional[ModalityFusionEngine] = None


def fuse_modalities(modalities: List[ModalityData], strategy: Optional[FusionStrategy] = None, config: Optional[FusionConfig] = None) -> FusionResult:
    """Fuse multiple modalities using specified strategy."""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = ModalityFusionEngine(config)
    
    return _fusion_engine.fuse(modalities, strategy)


async def async_fuse_modalities(modalities: List[ModalityData], strategy: Optional[FusionStrategy] = None, config: Optional[FusionConfig] = None) -> FusionResult:
    """Asynchronously fuse multiple modalities."""
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = ModalityFusionEngine(config)
    
    return await _fusion_engine.async_fuse(modalities, strategy)


def initialize_fusion(config: Optional[FusionConfig] = None) -> None:
    """Initialize fusion engine."""
    global _fusion_engine
    _fusion_engine = ModalityFusionEngine(config)


async def shutdown_fusion() -> None:
    """Shutdown fusion engine."""
    global _fusion_engine
    _fusion_engine = None