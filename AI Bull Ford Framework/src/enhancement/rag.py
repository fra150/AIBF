"""RAG (Retrieval-Augmented Generation) module for AI Bull Ford.

This module implements a comprehensive RAG system with:
- Document processing and indexing
- Vector storage and retrieval
- Context generation and augmentation
- Knowledge base management
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np


class DocumentType(Enum):
    """Types of documents that can be processed."""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"
    UNKNOWN = "unknown"


class VectorStoreType(Enum):
    """Types of vector stores supported."""
    MEMORY = "memory"
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class RetrievalStrategy(Enum):
    """Retrieval strategies for document search."""
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximal Marginal Relevance
    HYBRID = "hybrid"  # Combines semantic and keyword search
    RERANK = "rerank"  # Uses reranking models


@dataclass
class Document:
    """Represents a document in the RAG system."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_type: DocumentType = DocumentType.TEXT
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.content:
            raise ValueError("Document content cannot be empty")


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    document: Document
    score: float
    rank: int
    explanation: Optional[str] = None


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    vector_store_type: VectorStoreType = VectorStoreType.MEMORY
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_retrieval_results: int = 10
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY
    similarity_threshold: float = 0.7
    rerank_model: Optional[str] = None
    enable_hybrid_search: bool = False
    

class DocumentProcessor:
    """Processes documents for RAG system."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, content: str, doc_type: DocumentType, 
                        metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Process a single document."""
        if not content.strip():
            raise ValueError("Document content cannot be empty")
        
        # Clean and normalize content
        cleaned_content = self._clean_content(content, doc_type)
        
        # Create document
        doc = Document(
            content=cleaned_content,
            doc_type=doc_type,
            metadata=metadata or {}
        )
        
        self.logger.info(f"Processed document {doc.id} of type {doc_type}")
        return doc
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split document into chunks for better retrieval."""
        content = document.content
        chunks = []
        
        # Simple text chunking (can be enhanced with semantic chunking)
        for i in range(0, len(content), self.config.chunk_size - self.config.chunk_overlap):
            chunk_content = content[i:i + self.config.chunk_size]
            
            if chunk_content.strip():
                chunk = Document(
                    content=chunk_content,
                    doc_type=document.doc_type,
                    metadata={
                        **document.metadata,
                        'parent_id': document.id,
                        'chunk_index': len(chunks),
                        'is_chunk': True
                    },
                    source=document.source
                )
                chunks.append(chunk)
        
        self.logger.info(f"Split document {document.id} into {len(chunks)} chunks")
        return chunks
    
    def _clean_content(self, content: str, doc_type: DocumentType) -> str:
        """Clean and normalize document content."""
        # Basic cleaning
        content = content.strip()
        
        # Type-specific cleaning
        if doc_type == DocumentType.HTML:
            # Remove HTML tags (simplified)
            import re
            content = re.sub(r'<[^>]+>', '', content)
        elif doc_type == DocumentType.MARKDOWN:
            # Remove markdown formatting (simplified)
            import re
            content = re.sub(r'[#*`_]', '', content)
        
        # Normalize whitespace
        content = ' '.join(content.split())
        
        return content


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store."""
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass
    
    @abstractmethod
    def list_documents(self) -> List[str]:
        """List all document IDs."""
        pass


class MemoryVectorStore(VectorStore):
    """In-memory vector store implementation."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to memory store."""
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} must have embedding")
            
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = doc.embedding
        
        self.logger.info(f"Added {len(documents)} documents to memory store")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Document, float]]:
        """Search using cosine similarity."""
        if not self.embeddings:
            return []
        
        similarities = []
        for doc_id, embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((doc_id, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for doc_id, score in similarities[:k]:
            doc = self.documents[doc_id]
            results.append((doc, score))
        
        return results
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from memory store."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            del self.embeddings[doc_id]
            return True
        return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[str]:
        """List all document IDs."""
        return list(self.documents.keys())


class EmbeddingModel:
    """Handles text embeddings for RAG system."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        # In a real implementation, you would load the actual model here
        # For now, we'll simulate embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        # Simulate embedding generation (replace with actual model)
        # This creates a simple hash-based embedding for demonstration
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Convert to a 384-dimensional vector (typical for sentence transformers)
        embedding = np.random.RandomState(int(hash_obj.hexdigest(), 16) % (2**32)).normal(0, 1, 384)
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for multiple documents."""
        for doc in documents:
            doc.embedding = self.embed_text(doc.content)
        
        self.logger.info(f"Generated embeddings for {len(documents)} documents")
        return documents


class RetrievalEngine:
    """Handles document retrieval with various strategies."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel, config: RAGConfig):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        k = k or self.config.max_retrieval_results
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search vector store
        raw_results = self.vector_store.search(query_embedding, k)
        
        # Apply retrieval strategy
        if self.config.retrieval_strategy == RetrievalStrategy.SIMILARITY:
            results = self._similarity_retrieval(raw_results)
        elif self.config.retrieval_strategy == RetrievalStrategy.MMR:
            results = self._mmr_retrieval(raw_results, query_embedding)
        else:
            results = self._similarity_retrieval(raw_results)
        
        # Filter by threshold
        filtered_results = [
            result for result in results 
            if result.score >= self.config.similarity_threshold
        ]
        
        self.logger.info(f"Retrieved {len(filtered_results)} documents for query")
        return filtered_results
    
    def _similarity_retrieval(self, raw_results: List[Tuple[Document, float]]) -> List[RetrievalResult]:
        """Simple similarity-based retrieval."""
        results = []
        for rank, (doc, score) in enumerate(raw_results):
            result = RetrievalResult(
                document=doc,
                score=score,
                rank=rank,
                explanation=f"Similarity score: {score:.3f}"
            )
            results.append(result)
        return results
    
    def _mmr_retrieval(self, raw_results: List[Tuple[Document, float]], 
                      query_embedding: np.ndarray, lambda_param: float = 0.5) -> List[RetrievalResult]:
        """Maximal Marginal Relevance retrieval for diversity."""
        if not raw_results:
            return []
        
        selected = []
        remaining = list(raw_results)
        
        # Select first document (highest similarity)
        first_doc, first_score = remaining.pop(0)
        selected.append(RetrievalResult(
            document=first_doc,
            score=first_score,
            rank=0,
            explanation=f"MMR selected (first): {first_score:.3f}"
        ))
        
        # Select remaining documents using MMR
        while remaining and len(selected) < len(raw_results):
            best_mmr_score = -1
            best_idx = 0
            
            for i, (doc, similarity) in enumerate(remaining):
                # Calculate max similarity to already selected documents
                max_sim_to_selected = 0
                for selected_result in selected:
                    if doc.embedding is not None and selected_result.document.embedding is not None:
                        sim = np.dot(doc.embedding, selected_result.document.embedding)
                        max_sim_to_selected = max(max_sim_to_selected, sim)
                
                # MMR score
                mmr_score = lambda_param * similarity - (1 - lambda_param) * max_sim_to_selected
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i
            
            # Add best MMR document
            doc, original_score = remaining.pop(best_idx)
            selected.append(RetrievalResult(
                document=doc,
                score=best_mmr_score,
                rank=len(selected),
                explanation=f"MMR score: {best_mmr_score:.3f} (orig: {original_score:.3f})"
            ))
        
        return selected


class ContextGenerator:
    """Generates context from retrieved documents."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_context(self, retrieval_results: List[RetrievalResult], 
                        query: str, max_length: int = 2000) -> str:
        """Generate context string from retrieval results."""
        if not retrieval_results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in retrieval_results:
            doc_content = result.document.content
            
            # Add document with metadata
            doc_part = f"[Document {result.rank + 1}] {doc_content}"
            
            if current_length + len(doc_part) > max_length:
                # Truncate if needed
                remaining_space = max_length - current_length
                if remaining_space > 100:  # Only add if meaningful space left
                    doc_part = doc_part[:remaining_space] + "..."
                    context_parts.append(doc_part)
                break
            
            context_parts.append(doc_part)
            current_length += len(doc_part)
        
        context = "\n\n".join(context_parts)
        self.logger.info(f"Generated context of {len(context)} characters from {len(retrieval_results)} documents")
        
        return context


class KnowledgeBase:
    """Manages a knowledge base of documents."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.processor = DocumentProcessor(config)
        self.embedding_model = EmbeddingModel(config.embedding_model)
        
        # Initialize vector store
        if config.vector_store_type == VectorStoreType.MEMORY:
            self.vector_store = MemoryVectorStore()
        else:
            raise NotImplementedError(f"Vector store {config.vector_store_type} not implemented")
        
        self.logger = logging.getLogger(__name__)
    
    def add_document(self, content: str, doc_type: DocumentType = DocumentType.TEXT,
                    metadata: Optional[Dict[str, Any]] = None, chunk: bool = True) -> List[str]:
        """Add a document to the knowledge base."""
        # Process document
        doc = self.processor.process_document(content, doc_type, metadata)
        
        # Chunk if requested
        if chunk and len(content) > self.config.chunk_size:
            docs = self.processor.chunk_document(doc)
        else:
            docs = [doc]
        
        # Generate embeddings
        docs = self.embedding_model.embed_documents(docs)
        
        # Add to vector store
        self.vector_store.add_documents(docs)
        
        doc_ids = [d.id for d in docs]
        self.logger.info(f"Added document to knowledge base: {len(docs)} chunks")
        
        return doc_ids
    
    def add_documents_from_directory(self, directory: Path, 
                                   supported_types: Optional[List[DocumentType]] = None) -> List[str]:
        """Add all documents from a directory."""
        if supported_types is None:
            supported_types = [DocumentType.TEXT, DocumentType.MARKDOWN]
        
        doc_ids = []
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    doc_type = self._detect_document_type(file_path)
                    
                    if doc_type in supported_types:
                        metadata = {
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'file_size': file_path.stat().st_size
                        }
                        
                        ids = self.add_document(content, doc_type, metadata)
                        doc_ids.extend(ids)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process file {file_path}: {e}")
        
        self.logger.info(f"Added {len(doc_ids)} documents from directory {directory}")
        return doc_ids
    
    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type from file extension."""
        suffix = file_path.suffix.lower()
        
        type_mapping = {
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.pdf': DocumentType.PDF,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.json': DocumentType.JSON,
            '.csv': DocumentType.CSV,
            '.docx': DocumentType.DOCX
        }
        
        return type_mapping.get(suffix, DocumentType.UNKNOWN)


class RAGSystem:
    """Main RAG system that orchestrates all components."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.knowledge_base = KnowledgeBase(self.config)
        self.retrieval_engine = RetrievalEngine(
            self.knowledge_base.vector_store,
            self.knowledge_base.embedding_model,
            self.config
        )
        self.context_generator = ContextGenerator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def add_document(self, content: str, doc_type: DocumentType = DocumentType.TEXT,
                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Add a document to the RAG system."""
        return self.knowledge_base.add_document(content, doc_type, metadata)
    
    def query(self, question: str, max_context_length: int = 2000) -> Dict[str, Any]:
        """Query the RAG system."""
        # Retrieve relevant documents
        retrieval_results = self.retrieval_engine.retrieve(question)
        
        # Generate context
        context = self.context_generator.generate_context(
            retrieval_results, question, max_context_length
        )
        
        # Prepare response
        response = {
            'query': question,
            'context': context,
            'retrieved_documents': [
                {
                    'id': result.document.id,
                    'content': result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content,
                    'score': result.score,
                    'rank': result.rank,
                    'metadata': result.document.metadata
                }
                for result in retrieval_results
            ],
            'num_retrieved': len(retrieval_results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Processed query: {question[:50]}... Retrieved {len(retrieval_results)} documents")
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        doc_ids = self.knowledge_base.vector_store.list_documents()
        
        return {
            'total_documents': len(doc_ids),
            'config': {
                'vector_store_type': self.config.vector_store_type.value,
                'embedding_model': self.config.embedding_model,
                'chunk_size': self.config.chunk_size,
                'retrieval_strategy': self.config.retrieval_strategy.value
            }
        }


# Global RAG system instance
_rag_system: Optional[RAGSystem] = None


def get_rag_system(config: Optional[RAGConfig] = None) -> RAGSystem:
    """Get or create global RAG system instance."""
    global _rag_system
    
    if _rag_system is None:
        _rag_system = RAGSystem(config)
    
    return _rag_system


def initialize_rag_system(config: RAGConfig) -> RAGSystem:
    """Initialize global RAG system with specific config."""
    global _rag_system
    _rag_system = RAGSystem(config)
    return _rag_system


def shutdown_rag_system() -> None:
    """Shutdown global RAG system."""
    global _rag_system
    _rag_system = None