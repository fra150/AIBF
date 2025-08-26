"""Enhancement module for AI Bull Ford.

This module provides advanced AI enhancement capabilities including:
- RAG (Retrieval-Augmented Generation) systems
- Fine-tuning capabilities
- Memory management systems"""

from . import rag, fine_tuning, memory
from .rag import (
    RAGSystem, RAGConfig, Document, RetrievalResult,
    DocumentProcessor, VectorStore, MemoryVectorStore,
    EmbeddingModel, RetrievalEngine, ContextGenerator,
    KnowledgeBase, get_rag_system, initialize_rag_system
)
from .fine_tuning import (
    FineTuner, TrainingConfig, TrainingMetrics, TrainingState,
    FineTuningMethod, OptimizerType, SchedulerType, TaskType,
    DatasetProcessor, ModelAdapter, LoRAAdapter,
    TrainingMonitor, ModelOptimizer, get_fine_tuner, initialize_fine_tuner
)
from .memory import (
    MemoryManager, MemoryConfig, MemoryItem, MemoryQuery,
    MemoryType, MemoryPriority, ConsolidationStrategy,
    MemoryStore, SQLiteMemoryStore, MemoryEmbedding,
    MemoryConsolidator, get_memory_manager, initialize_memory_manager
)

__all__ = [
    # Modules
    'rag',
    'fine_tuning', 
    'memory',
    
    # RAG components
    'RAGSystem', 'RAGConfig', 'Document', 'RetrievalResult',
    'DocumentProcessor', 'VectorStore', 'MemoryVectorStore',
    'EmbeddingModel', 'RetrievalEngine', 'ContextGenerator',
    'KnowledgeBase', 'get_rag_system', 'initialize_rag_system',
    
    # Fine-tuning components
    'FineTuner', 'TrainingConfig', 'TrainingMetrics', 'TrainingState',
    'FineTuningMethod', 'OptimizerType', 'SchedulerType', 'TaskType',
    'DatasetProcessor', 'ModelAdapter', 'LoRAAdapter',
    'TrainingMonitor', 'ModelOptimizer', 'get_fine_tuner', 'initialize_fine_tuner',
    
    # Memory components
    'MemoryManager', 'MemoryConfig', 'MemoryItem', 'MemoryQuery',
    'MemoryType', 'MemoryPriority', 'ConsolidationStrategy',
    'MemoryStore', 'SQLiteMemoryStore', 'MemoryEmbedding',
    'MemoryConsolidator', 'get_memory_manager', 'initialize_memory_manager'
]