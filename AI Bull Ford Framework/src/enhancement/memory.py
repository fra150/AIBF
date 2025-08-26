"""Memory management module for AI Bull Ford.

This module provides comprehensive memory management capabilities including:
- Short-term and long-term memory systems
- Memory consolidation and retrieval
- Context-aware memory operations
- Memory optimization and cleanup
"""

import json
import logging
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import numpy as np


class MemoryType(Enum):
    """Types of memory."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryPriority(Enum):
    """Memory priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConsolidationStrategy(Enum):
    """Memory consolidation strategies."""
    FREQUENCY_BASED = "frequency_based"
    RECENCY_BASED = "recency_based"
    IMPORTANCE_BASED = "importance_based"
    HYBRID = "hybrid"


@dataclass
class MemoryItem:
    """Individual memory item."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.SHORT_TERM
    priority: MemoryPriority = MemoryPriority.MEDIUM
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    decay_factor: float = 1.0
    embedding: Optional[np.ndarray] = None
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_relevance_score(self, current_time: Optional[datetime] = None) -> float:
        """Calculate relevance score based on recency, frequency, and priority."""
        if current_time is None:
            current_time = datetime.now()
        
        # Recency score (exponential decay)
        time_diff = (current_time - self.last_accessed).total_seconds()
        recency_score = np.exp(-time_diff / 3600)  # 1-hour half-life
        
        # Frequency score (logarithmic)
        frequency_score = np.log(1 + self.access_count)
        
        # Priority score
        priority_weights = {
            MemoryPriority.LOW: 0.5,
            MemoryPriority.MEDIUM: 1.0,
            MemoryPriority.HIGH: 2.0,
            MemoryPriority.CRITICAL: 4.0
        }
        priority_score = priority_weights[self.priority]
        
        # Combined score
        relevance = (recency_score * 0.4 + frequency_score * 0.3 + priority_score * 0.3) * self.decay_factor
        
        return relevance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'priority': self.priority.value,
            'tags': list(self.tags),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'decay_factor': self.decay_factor,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary."""
        item = cls(
            id=data['id'],
            content=data['content'],
            memory_type=MemoryType(data['memory_type']),
            priority=MemoryPriority(data['priority']),
            tags=set(data['tags']),
            metadata=data['metadata'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data['access_count'],
            decay_factor=data['decay_factor']
        )
        
        if data['embedding'] is not None:
            item.embedding = np.array(data['embedding'])
        
        return item


@dataclass
class MemoryQuery:
    """Query for memory retrieval."""
    content: Optional[str] = None
    tags: Optional[Set[str]] = None
    memory_type: Optional[MemoryType] = None
    priority: Optional[MemoryPriority] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    limit: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    # Capacity limits
    short_term_capacity: int = 1000
    working_memory_capacity: int = 100
    long_term_capacity: int = 10000
    
    # Consolidation settings
    consolidation_strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID
    consolidation_interval: int = 3600  # seconds
    consolidation_threshold: float = 0.5
    
    # Cleanup settings
    cleanup_interval: int = 86400  # seconds (24 hours)
    min_relevance_threshold: float = 0.1
    max_age_days: int = 30
    
    # Embedding settings
    embedding_dimension: int = 384
    use_embeddings: bool = True
    
    # Storage settings
    storage_path: str = "./memory_storage"
    backup_interval: int = 3600  # seconds
    
    # Performance settings
    batch_size: int = 100
    max_search_results: int = 1000


class MemoryStore(ABC):
    """Abstract base class for memory storage."""
    
    @abstractmethod
    def store(self, item: MemoryItem) -> None:
        """Store memory item."""
        pass
    
    @abstractmethod
    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve memory item by ID."""
        pass
    
    @abstractmethod
    def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search memory items."""
        pass
    
    @abstractmethod
    def update(self, item: MemoryItem) -> None:
        """Update memory item."""
        pass
    
    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Delete memory item."""
        pass
    
    @abstractmethod
    def list_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """List items by memory type."""
        pass
    
    @abstractmethod
    def cleanup(self, threshold: float, max_age: timedelta) -> int:
        """Cleanup old/irrelevant items."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass


class SQLiteMemoryStore(MemoryStore):
    """SQLite-based memory storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    memory_type TEXT,
                    priority TEXT,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER,
                    decay_factor REAL,
                    embedding BLOB
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON memory_items(priority)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memory_items(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON memory_items(last_accessed)")
            
            conn.commit()
    
    def store(self, item: MemoryItem) -> None:
        """Store memory item."""
        with sqlite3.connect(self.db_path) as conn:
            embedding_blob = None
            if item.embedding is not None:
                embedding_blob = pickle.dumps(item.embedding)
            
            conn.execute("""
                INSERT OR REPLACE INTO memory_items 
                (id, content, memory_type, priority, tags, metadata, created_at, 
                 last_accessed, access_count, decay_factor, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id,
                json.dumps(item.content) if item.content is not None else None,
                item.memory_type.value,
                item.priority.value,
                json.dumps(list(item.tags)),
                json.dumps(item.metadata),
                item.created_at.isoformat(),
                item.last_accessed.isoformat(),
                item.access_count,
                item.decay_factor,
                embedding_blob
            ))
            conn.commit()
    
    def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve memory item by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?",
                (item_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_item(row)
            
            return None
    
    def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search memory items."""
        conditions = []
        params = []
        
        if query.memory_type:
            conditions.append("memory_type = ?")
            params.append(query.memory_type.value)
        
        if query.priority:
            conditions.append("priority = ?")
            params.append(query.priority.value)
        
        if query.time_range:
            conditions.append("created_at BETWEEN ? AND ?")
            params.extend([query.time_range[0].isoformat(), query.time_range[1].isoformat()])
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        sql = f"""
            SELECT * FROM memory_items 
            {where_clause}
            ORDER BY last_accessed DESC 
            LIMIT ?
        """
        params.append(query.limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            
            items = [self._row_to_item(row) for row in rows]
            
            # Filter by tags if specified
            if query.tags:
                items = [item for item in items if query.tags.intersection(item.tags)]
            
            return items[:query.limit]
    
    def update(self, item: MemoryItem) -> None:
        """Update memory item."""
        self.store(item)  # SQLite REPLACE handles updates
    
    def delete(self, item_id: str) -> bool:
        """Delete memory item."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM memory_items WHERE id = ?",
                (item_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def list_by_type(self, memory_type: MemoryType) -> List[MemoryItem]:
        """List items by memory type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM memory_items WHERE memory_type = ? ORDER BY last_accessed DESC",
                (memory_type.value,)
            )
            rows = cursor.fetchall()
            return [self._row_to_item(row) for row in rows]
    
    def cleanup(self, threshold: float, max_age: timedelta) -> int:
        """Cleanup old/irrelevant items."""
        cutoff_date = datetime.now() - max_age
        
        with sqlite3.connect(self.db_path) as conn:
            # Delete old items
            cursor = conn.execute(
                "DELETE FROM memory_items WHERE created_at < ?",
                (cutoff_date.isoformat(),)
            )
            deleted_count = cursor.rowcount
            
            # Delete low-relevance items (simplified - would need to calculate relevance)
            cursor = conn.execute(
                "DELETE FROM memory_items WHERE decay_factor < ? AND access_count < 2",
                (threshold,)
            )
            deleted_count += cursor.rowcount
            
            conn.commit()
            
            return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM memory_items")
            total_count = cursor.fetchone()[0]
            
            # Count by type
            cursor = conn.execute(
                "SELECT memory_type, COUNT(*) FROM memory_items GROUP BY memory_type"
            )
            type_counts = dict(cursor.fetchall())
            
            # Count by priority
            cursor = conn.execute(
                "SELECT priority, COUNT(*) FROM memory_items GROUP BY priority"
            )
            priority_counts = dict(cursor.fetchall())
            
            return {
                'total_items': total_count,
                'by_type': type_counts,
                'by_priority': priority_counts
            }
    
    def _row_to_item(self, row: Tuple) -> MemoryItem:
        """Convert database row to MemoryItem."""
        (
            item_id, content, memory_type, priority, tags, metadata,
            created_at, last_accessed, access_count, decay_factor, embedding_blob
        ) = row
        
        # Parse JSON fields
        content = json.loads(content) if content else None
        tags = set(json.loads(tags)) if tags else set()
        metadata = json.loads(metadata) if metadata else {}
        
        # Parse embedding
        embedding = None
        if embedding_blob:
            embedding = pickle.loads(embedding_blob)
        
        return MemoryItem(
            id=item_id,
            content=content,
            memory_type=MemoryType(memory_type),
            priority=MemoryPriority(priority),
            tags=tags,
            metadata=metadata,
            created_at=datetime.fromisoformat(created_at),
            last_accessed=datetime.fromisoformat(last_accessed),
            access_count=access_count,
            decay_factor=decay_factor,
            embedding=embedding
        )


class MemoryEmbedding:
    """Handles memory embeddings for similarity search."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.logger = logging.getLogger(__name__)
    
    def generate_embedding(self, content: Any) -> np.ndarray:
        """Generate embedding for content."""
        # Simplified embedding generation
        # In practice, use a proper embedding model
        if isinstance(content, str):
            # Simple hash-based embedding for demonstration
            hash_value = hash(content)
            embedding = np.random.RandomState(hash_value % (2**32)).normal(0, 1, self.dimension)
            return embedding / np.linalg.norm(embedding)
        else:
            # Random embedding for non-string content
            return np.random.normal(0, 1, self.dimension)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def find_similar_items(self, query_embedding: np.ndarray, 
                          items: List[MemoryItem], 
                          threshold: float = 0.7) -> List[Tuple[MemoryItem, float]]:
        """Find items similar to query embedding."""
        similar_items = []
        
        for item in items:
            if item.embedding is not None:
                similarity = self.calculate_similarity(query_embedding, item.embedding)
                if similarity >= threshold:
                    similar_items.append((item, similarity))
        
        # Sort by similarity (descending)
        similar_items.sort(key=lambda x: x[1], reverse=True)
        
        return similar_items


class MemoryConsolidator:
    """Handles memory consolidation from short-term to long-term."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def should_consolidate(self, item: MemoryItem) -> bool:
        """Determine if item should be consolidated to long-term memory."""
        relevance_score = item.calculate_relevance_score()
        
        if self.config.consolidation_strategy == ConsolidationStrategy.FREQUENCY_BASED:
            return item.access_count >= 3
        
        elif self.config.consolidation_strategy == ConsolidationStrategy.RECENCY_BASED:
            age = datetime.now() - item.created_at
            return age.total_seconds() > 3600  # 1 hour
        
        elif self.config.consolidation_strategy == ConsolidationStrategy.IMPORTANCE_BASED:
            return item.priority in [MemoryPriority.HIGH, MemoryPriority.CRITICAL]
        
        elif self.config.consolidation_strategy == ConsolidationStrategy.HYBRID:
            # Combine multiple factors
            return (
                relevance_score > self.config.consolidation_threshold or
                item.priority in [MemoryPriority.HIGH, MemoryPriority.CRITICAL] or
                item.access_count >= 5
            )
        
        return False
    
    def consolidate_memories(self, short_term_items: List[MemoryItem]) -> List[MemoryItem]:
        """Consolidate short-term memories to long-term."""
        consolidated = []
        
        for item in short_term_items:
            if self.should_consolidate(item):
                # Convert to long-term memory
                item.memory_type = MemoryType.LONG_TERM
                # Boost decay factor for consolidated memories
                item.decay_factor = min(1.0, item.decay_factor * 1.2)
                consolidated.append(item)
        
        self.logger.info(f"Consolidated {len(consolidated)} memories to long-term storage")
        return consolidated
    
    def merge_similar_memories(self, items: List[MemoryItem], 
                              similarity_threshold: float = 0.9) -> List[MemoryItem]:
        """Merge similar memories to reduce redundancy."""
        if not items:
            return items
        
        merged = []
        processed = set()
        
        for i, item1 in enumerate(items):
            if item1.id in processed:
                continue
            
            similar_items = [item1]
            processed.add(item1.id)
            
            # Find similar items
            for j, item2 in enumerate(items[i+1:], i+1):
                if item2.id in processed:
                    continue
                
                if item1.embedding is not None and item2.embedding is not None:
                    similarity = np.dot(item1.embedding, item2.embedding)
                    if similarity > similarity_threshold:
                        similar_items.append(item2)
                        processed.add(item2.id)
            
            # Merge similar items
            if len(similar_items) > 1:
                merged_item = self._merge_items(similar_items)
                merged.append(merged_item)
            else:
                merged.append(item1)
        
        self.logger.info(f"Merged {len(items) - len(merged)} similar memories")
        return merged
    
    def _merge_items(self, items: List[MemoryItem]) -> MemoryItem:
        """Merge multiple memory items into one."""
        # Use the most recent item as base
        base_item = max(items, key=lambda x: x.last_accessed)
        
        # Combine content (simplified)
        if isinstance(base_item.content, str):
            contents = [item.content for item in items if isinstance(item.content, str)]
            merged_content = " | ".join(contents)
        else:
            merged_content = base_item.content
        
        # Combine tags
        merged_tags = set()
        for item in items:
            merged_tags.update(item.tags)
        
        # Combine metadata
        merged_metadata = {}
        for item in items:
            merged_metadata.update(item.metadata)
        
        # Calculate combined statistics
        total_access_count = sum(item.access_count for item in items)
        avg_decay_factor = sum(item.decay_factor for item in items) / len(items)
        
        # Create merged item
        merged_item = MemoryItem(
            id=str(uuid4()),
            content=merged_content,
            memory_type=base_item.memory_type,
            priority=max(item.priority for item in items),
            tags=merged_tags,
            metadata=merged_metadata,
            created_at=min(item.created_at for item in items),
            last_accessed=max(item.last_accessed for item in items),
            access_count=total_access_count,
            decay_factor=avg_decay_factor,
            embedding=base_item.embedding
        )
        
        return merged_item


class MemoryManager:
    """Main memory management system."""
    
    def __init__(self, config: MemoryConfig, store: Optional[MemoryStore] = None):
        self.config = config
        self.store = store or SQLiteMemoryStore(f"{config.storage_path}/memory.db")
        self.embedding = MemoryEmbedding(config.embedding_dimension) if config.use_embeddings else None
        self.consolidator = MemoryConsolidator(config)
        
        # In-memory caches for performance
        self.working_memory: deque = deque(maxlen=config.working_memory_capacity)
        self.recent_cache: Dict[str, MemoryItem] = {}
        
        self.logger = logging.getLogger(__name__)
        self.last_consolidation = datetime.now()
        self.last_cleanup = datetime.now()
    
    def store_memory(self, content: Any, 
                    memory_type: MemoryType = MemoryType.SHORT_TERM,
                    priority: MemoryPriority = MemoryPriority.MEDIUM,
                    tags: Optional[Set[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store new memory item."""
        # Create memory item
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            priority=priority,
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        # Generate embedding if enabled
        if self.embedding:
            item.embedding = self.embedding.generate_embedding(content)
        
        # Store in appropriate location
        if memory_type == MemoryType.WORKING:
            self.working_memory.append(item)
        else:
            self.store.store(item)
            
            # Add to recent cache
            self.recent_cache[item.id] = item
            
            # Limit cache size
            if len(self.recent_cache) > 1000:
                # Remove oldest items
                oldest_items = sorted(self.recent_cache.values(), key=lambda x: x.last_accessed)[:100]
                for old_item in oldest_items:
                    self.recent_cache.pop(old_item.id, None)
        
        self.logger.debug(f"Stored memory item {item.id} of type {memory_type.value}")
        
        # Check if consolidation is needed
        self._check_consolidation()
        
        return item.id
    
    def retrieve_memory(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve memory item by ID."""
        # Check working memory first
        for item in self.working_memory:
            if item.id == item_id:
                item.update_access()
                return item
        
        # Check recent cache
        if item_id in self.recent_cache:
            item = self.recent_cache[item_id]
            item.update_access()
            self.store.update(item)
            return item
        
        # Check persistent storage
        item = self.store.retrieve(item_id)
        if item:
            item.update_access()
            self.store.update(item)
            self.recent_cache[item_id] = item
        
        return item
    
    def search_memories(self, query: Union[str, MemoryQuery]) -> List[MemoryItem]:
        """Search for memories."""
        if isinstance(query, str):
            # Convert string to MemoryQuery
            query_obj = MemoryQuery(content=query)
        else:
            query_obj = query
        
        # Search persistent storage
        results = self.store.search(query_obj)
        
        # Search working memory
        working_results = []
        for item in self.working_memory:
            if self._matches_query(item, query_obj):
                working_results.append(item)
        
        # Combine results
        all_results = results + working_results
        
        # If using embeddings and query has content, do similarity search
        if self.embedding and query_obj.content:
            query_embedding = self.embedding.generate_embedding(query_obj.content)
            similar_items = self.embedding.find_similar_items(
                query_embedding, all_results, query_obj.similarity_threshold
            )
            # Sort by similarity and extract items
            all_results = [item for item, _ in similar_items]
        
        # Update access for retrieved items
        for item in all_results:
            item.update_access()
            if item.memory_type != MemoryType.WORKING:
                self.store.update(item)
        
        return all_results[:query_obj.limit]
    
    def _matches_query(self, item: MemoryItem, query: MemoryQuery) -> bool:
        """Check if item matches query criteria."""
        if query.memory_type and item.memory_type != query.memory_type:
            return False
        
        if query.priority and item.priority != query.priority:
            return False
        
        if query.tags and not query.tags.intersection(item.tags):
            return False
        
        if query.time_range:
            start, end = query.time_range
            if not (start <= item.created_at <= end):
                return False
        
        return True
    
    def update_memory(self, item_id: str, **updates) -> bool:
        """Update memory item."""
        item = self.retrieve_memory(item_id)
        if not item:
            return False
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        # Update embedding if content changed
        if 'content' in updates and self.embedding:
            item.embedding = self.embedding.generate_embedding(item.content)
        
        # Save changes
        if item.memory_type == MemoryType.WORKING:
            # Already updated in working memory
            pass
        else:
            self.store.update(item)
            self.recent_cache[item_id] = item
        
        return True
    
    def delete_memory(self, item_id: str) -> bool:
        """Delete memory item."""
        # Remove from working memory
        self.working_memory = deque(
            [item for item in self.working_memory if item.id != item_id],
            maxlen=self.config.working_memory_capacity
        )
        
        # Remove from cache
        self.recent_cache.pop(item_id, None)
        
        # Remove from persistent storage
        return self.store.delete(item_id)
    
    def consolidate_memories(self) -> int:
        """Manually trigger memory consolidation."""
        # Get short-term memories
        short_term_items = self.store.list_by_type(MemoryType.SHORT_TERM)
        
        # Consolidate
        consolidated = self.consolidator.consolidate_memories(short_term_items)
        
        # Update storage
        for item in consolidated:
            self.store.update(item)
        
        self.last_consolidation = datetime.now()
        return len(consolidated)
    
    def cleanup_memories(self) -> int:
        """Clean up old and irrelevant memories."""
        max_age = timedelta(days=self.config.max_age_days)
        deleted_count = self.store.cleanup(self.config.min_relevance_threshold, max_age)
        
        self.last_cleanup = datetime.now()
        self.logger.info(f"Cleaned up {deleted_count} old memories")
        
        return deleted_count
    
    def _check_consolidation(self) -> None:
        """Check if consolidation should be triggered."""
        time_since_last = (datetime.now() - self.last_consolidation).total_seconds()
        
        if time_since_last > self.config.consolidation_interval:
            self.consolidate_memories()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        storage_stats = self.store.get_stats()
        
        return {
            'working_memory_size': len(self.working_memory),
            'cache_size': len(self.recent_cache),
            'storage_stats': storage_stats,
            'last_consolidation': self.last_consolidation.isoformat(),
            'last_cleanup': self.last_cleanup.isoformat(),
            'config': {
                'short_term_capacity': self.config.short_term_capacity,
                'working_memory_capacity': self.config.working_memory_capacity,
                'long_term_capacity': self.config.long_term_capacity
            }
        }
    
    def export_memories(self, file_path: str, memory_type: Optional[MemoryType] = None) -> None:
        """Export memories to file."""
        if memory_type:
            items = self.store.list_by_type(memory_type)
        else:
            # Export all memories
            items = []
            for mem_type in MemoryType:
                items.extend(self.store.list_by_type(mem_type))
        
        # Convert to serializable format
        export_data = {
            'export_time': datetime.now().isoformat(),
            'memory_type': memory_type.value if memory_type else 'all',
            'items': [item.to_dict() for item in items]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(items)} memories to {file_path}")
    
    def import_memories(self, file_path: str) -> int:
        """Import memories from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        imported_count = 0
        for item_data in import_data['items']:
            item = MemoryItem.from_dict(item_data)
            self.store.store(item)
            imported_count += 1
        
        self.logger.info(f"Imported {imported_count} memories from {file_path}")
        return imported_count


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(config: Optional[MemoryConfig] = None) -> MemoryManager:
    """Get or create global memory manager instance."""
    global _memory_manager
    
    if _memory_manager is None or config is not None:
        _memory_manager = MemoryManager(config or MemoryConfig())
    
    return _memory_manager


def initialize_memory_manager(config: MemoryConfig) -> MemoryManager:
    """Initialize global memory manager with specific config."""
    global _memory_manager
    _memory_manager = MemoryManager(config)
    return _memory_manager


def shutdown_memory_manager() -> None:
    """Shutdown global memory manager."""
    global _memory_manager
    if _memory_manager:
        # Perform final consolidation and cleanup
        _memory_manager.consolidate_memories()
        _memory_manager.cleanup_memories()
    _memory_manager = None