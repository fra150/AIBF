"""Audit logging module for AI Bull Ford security layer.

Provides comprehensive audit logging, event tracking, and security monitoring.
"""

import os
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Union, Callable, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import time
import hashlib
import hmac
import secrets
from contextlib import contextmanager
import functools
import inspect


class EventType(Enum):
    """Audit event types."""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    TOKEN_GENERATED = "token_generated"
    TOKEN_REVOKED = "token_revoked"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"
    
    # Data events
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIGURATION_CHANGE = "configuration_change"
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ENCRYPTION_KEY_GENERATED = "encryption_key_generated"
    ENCRYPTION_KEY_ROTATED = "encryption_key_rotated"
    
    # API events
    API_CALL = "api_call"
    API_ERROR = "api_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # Model events
    MODEL_TRAINING_START = "model_training_start"
    MODEL_TRAINING_COMPLETE = "model_training_complete"
    MODEL_INFERENCE = "model_inference"
    MODEL_DEPLOYMENT = "model_deployment"
    
    # Custom events
    CUSTOM = "custom"


class EventSeverity(Enum):
    """Event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(Enum):
    """Event status."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    status: EventStatus = EventStatus.SUCCESS
    severity: EventSeverity = EventSeverity.LOW
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['status'] = self.status.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        data['tags'] = list(self.tags)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create event from dictionary."""
        data = data.copy()
        data['event_type'] = EventType(data['event_type'])
        data['status'] = EventStatus(data['status'])
        data['severity'] = EventSeverity(data['severity'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['tags'] = set(data.get('tags', []))
        return cls(**data)


@dataclass
class AuditFilter:
    """Audit event filter."""
    event_types: Optional[Set[EventType]] = None
    user_ids: Optional[Set[str]] = None
    severities: Optional[Set[EventSeverity]] = None
    statuses: Optional[Set[EventStatus]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    resources: Optional[Set[str]] = None
    tags: Optional[Set[str]] = None
    ip_addresses: Optional[Set[str]] = None
    correlation_ids: Optional[Set[str]] = None
    
    def matches(self, event: AuditEvent) -> bool:
        """Check if event matches filter."""
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        if self.user_ids and event.user_id not in self.user_ids:
            return False
        
        if self.severities and event.severity not in self.severities:
            return False
        
        if self.statuses and event.status not in self.statuses:
            return False
        
        if self.start_time and event.timestamp < self.start_time:
            return False
        
        if self.end_time and event.timestamp > self.end_time:
            return False
        
        if self.resources and event.resource not in self.resources:
            return False
        
        if self.tags and not self.tags.intersection(event.tags):
            return False
        
        if self.ip_addresses and event.ip_address not in self.ip_addresses:
            return False
        
        if self.correlation_ids and event.correlation_id not in self.correlation_ids:
            return False
        
        return True


class AuditStore(ABC):
    """Abstract audit event storage interface."""
    
    @abstractmethod
    def store_event(self, event: AuditEvent) -> bool:
        """Store an audit event."""
        pass
    
    @abstractmethod
    def get_events(self, 
                   filter_criteria: Optional[AuditFilter] = None,
                   limit: int = 1000,
                   offset: int = 0) -> List[AuditEvent]:
        """Get audit events."""
        pass
    
    @abstractmethod
    def count_events(self, filter_criteria: Optional[AuditFilter] = None) -> int:
        """Count audit events."""
        pass
    
    @abstractmethod
    def delete_events(self, filter_criteria: AuditFilter) -> int:
        """Delete audit events."""
        pass
    
    @abstractmethod
    def get_event_statistics(self, 
                           filter_criteria: Optional[AuditFilter] = None) -> Dict[str, Any]:
        """Get event statistics."""
        pass


class SQLiteAuditStore(AuditStore):
    """SQLite-based audit event storage."""
    
    def __init__(self, db_path: str):
        """Initialize SQLite audit store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger("security.audit.sqlite")
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    resource TEXT,
                    action TEXT,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT,
                    details TEXT,
                    tags TEXT,
                    correlation_id TEXT,
                    parent_event_id TEXT,
                    duration_ms INTEGER
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_correlation_id ON audit_events(correlation_id)")
            
            conn.commit()
    
    def store_event(self, event: AuditEvent) -> bool:
        """Store an audit event."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO audit_events (
                            event_id, event_type, timestamp, user_id, session_id,
                            ip_address, user_agent, resource, action, status,
                            severity, message, details, tags, correlation_id,
                            parent_event_id, duration_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.user_id,
                        event.session_id,
                        event.ip_address,
                        event.user_agent,
                        event.resource,
                        event.action,
                        event.status.value,
                        event.severity.value,
                        event.message,
                        json.dumps(event.details) if event.details else None,
                        json.dumps(list(event.tags)) if event.tags else None,
                        event.correlation_id,
                        event.parent_event_id,
                        event.duration_ms
                    ))
                    conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store audit event: {e}")
            return False
    
    def get_events(self, 
                   filter_criteria: Optional[AuditFilter] = None,
                   limit: int = 1000,
                   offset: int = 0) -> List[AuditEvent]:
        """Get audit events."""
        query = "SELECT * FROM audit_events"
        params = []
        
        if filter_criteria:
            conditions = []
            
            if filter_criteria.event_types:
                placeholders = ','.join('?' * len(filter_criteria.event_types))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend([et.value for et in filter_criteria.event_types])
            
            if filter_criteria.user_ids:
                placeholders = ','.join('?' * len(filter_criteria.user_ids))
                conditions.append(f"user_id IN ({placeholders})")
                params.extend(filter_criteria.user_ids)
            
            if filter_criteria.severities:
                placeholders = ','.join('?' * len(filter_criteria.severities))
                conditions.append(f"severity IN ({placeholders})")
                params.extend([s.value for s in filter_criteria.severities])
            
            if filter_criteria.statuses:
                placeholders = ','.join('?' * len(filter_criteria.statuses))
                conditions.append(f"status IN ({placeholders})")
                params.extend([s.value for s in filter_criteria.statuses])
            
            if filter_criteria.start_time:
                conditions.append("timestamp >= ?")
                params.append(filter_criteria.start_time.isoformat())
            
            if filter_criteria.end_time:
                conditions.append("timestamp <= ?")
                params.append(filter_criteria.end_time.isoformat())
            
            if filter_criteria.resources:
                placeholders = ','.join('?' * len(filter_criteria.resources))
                conditions.append(f"resource IN ({placeholders})")
                params.extend(filter_criteria.resources)
            
            if filter_criteria.ip_addresses:
                placeholders = ','.join('?' * len(filter_criteria.ip_addresses))
                conditions.append(f"ip_address IN ({placeholders})")
                params.extend(filter_criteria.ip_addresses)
            
            if filter_criteria.correlation_ids:
                placeholders = ','.join('?' * len(filter_criteria.correlation_ids))
                conditions.append(f"correlation_id IN ({placeholders})")
                params.extend(filter_criteria.correlation_ids)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        events = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    event = AuditEvent(
                        event_id=row['event_id'],
                        event_type=EventType(row['event_type']),
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        user_id=row['user_id'],
                        session_id=row['session_id'],
                        ip_address=row['ip_address'],
                        user_agent=row['user_agent'],
                        resource=row['resource'],
                        action=row['action'],
                        status=EventStatus(row['status']),
                        severity=EventSeverity(row['severity']),
                        message=row['message'],
                        details=json.loads(row['details']) if row['details'] else {},
                        tags=set(json.loads(row['tags'])) if row['tags'] else set(),
                        correlation_id=row['correlation_id'],
                        parent_event_id=row['parent_event_id'],
                        duration_ms=row['duration_ms']
                    )
                    events.append(event)
        
        except Exception as e:
            self.logger.error(f"Failed to get audit events: {e}")
        
        return events
    
    def count_events(self, filter_criteria: Optional[AuditFilter] = None) -> int:
        """Count audit events."""
        query = "SELECT COUNT(*) FROM audit_events"
        params = []
        
        if filter_criteria:
            conditions = []
            
            if filter_criteria.event_types:
                placeholders = ','.join('?' * len(filter_criteria.event_types))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend([et.value for et in filter_criteria.event_types])
            
            if filter_criteria.start_time:
                conditions.append("timestamp >= ?")
                params.append(filter_criteria.start_time.isoformat())
            
            if filter_criteria.end_time:
                conditions.append("timestamp <= ?")
                params.append(filter_criteria.end_time.isoformat())
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                return cursor.fetchone()[0]
        
        except Exception as e:
            self.logger.error(f"Failed to count audit events: {e}")
            return 0
    
    def delete_events(self, filter_criteria: AuditFilter) -> int:
        """Delete audit events."""
        query = "DELETE FROM audit_events"
        params = []
        conditions = []
        
        if filter_criteria.event_types:
            placeholders = ','.join('?' * len(filter_criteria.event_types))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend([et.value for et in filter_criteria.event_types])
        
        if filter_criteria.start_time:
            conditions.append("timestamp >= ?")
            params.append(filter_criteria.start_time.isoformat())
        
        if filter_criteria.end_time:
            conditions.append("timestamp <= ?")
            params.append(filter_criteria.end_time.isoformat())
        
        if not conditions:
            raise ValueError("Delete filter must specify at least one condition")
        
        query += " WHERE " + " AND ".join(conditions)
        
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(query, params)
                    deleted_count = cursor.rowcount
                    conn.commit()
                    return deleted_count
        
        except Exception as e:
            self.logger.error(f"Failed to delete audit events: {e}")
            return 0
    
    def get_event_statistics(self, 
                           filter_criteria: Optional[AuditFilter] = None) -> Dict[str, Any]:
        """Get event statistics."""
        base_query = "FROM audit_events"
        params = []
        
        if filter_criteria:
            conditions = []
            
            if filter_criteria.start_time:
                conditions.append("timestamp >= ?")
                params.append(filter_criteria.start_time.isoformat())
            
            if filter_criteria.end_time:
                conditions.append("timestamp <= ?")
                params.append(filter_criteria.end_time.isoformat())
            
            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)
        
        stats = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total events
                cursor = conn.execute(f"SELECT COUNT(*) {base_query}", params)
                stats['total_events'] = cursor.fetchone()[0]
                
                # Events by type
                cursor = conn.execute(f"SELECT event_type, COUNT(*) {base_query} GROUP BY event_type", params)
                stats['events_by_type'] = dict(cursor.fetchall())
                
                # Events by severity
                cursor = conn.execute(f"SELECT severity, COUNT(*) {base_query} GROUP BY severity", params)
                stats['events_by_severity'] = dict(cursor.fetchall())
                
                # Events by status
                cursor = conn.execute(f"SELECT status, COUNT(*) {base_query} GROUP BY status", params)
                stats['events_by_status'] = dict(cursor.fetchall())
                
                # Top users
                cursor = conn.execute(f"SELECT user_id, COUNT(*) {base_query} WHERE user_id IS NOT NULL GROUP BY user_id ORDER BY COUNT(*) DESC LIMIT 10", params)
                stats['top_users'] = dict(cursor.fetchall())
                
                # Top IP addresses
                cursor = conn.execute(f"SELECT ip_address, COUNT(*) {base_query} WHERE ip_address IS NOT NULL GROUP BY ip_address ORDER BY COUNT(*) DESC LIMIT 10", params)
                stats['top_ip_addresses'] = dict(cursor.fetchall())
        
        except Exception as e:
            self.logger.error(f"Failed to get event statistics: {e}")
        
        return stats


class FileAuditStore(AuditStore):
    """File-based audit event storage."""
    
    def __init__(self, log_directory: str, max_file_size: int = 100 * 1024 * 1024):
        """Initialize file audit store.
        
        Args:
            log_directory: Directory for log files
            max_file_size: Maximum file size before rotation
        """
        self.log_directory = Path(log_directory)
        self.max_file_size = max_file_size
        self.logger = logging.getLogger("security.audit.file")
        self._lock = threading.Lock()
        
        # Ensure directory exists
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self.current_file = self._get_current_log_file()
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.log_directory / f"audit_{timestamp}.jsonl"
    
    def _rotate_log_file(self) -> None:
        """Rotate log file if needed."""
        if self.current_file.exists() and self.current_file.stat().st_size > self.max_file_size:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_file = self.log_directory / f"audit_{timestamp}.jsonl"
            self.current_file.rename(rotated_file)
            self.current_file = self._get_current_log_file()
    
    def store_event(self, event: AuditEvent) -> bool:
        """Store an audit event."""
        try:
            with self._lock:
                self._rotate_log_file()
                
                with open(self.current_file, 'a', encoding='utf-8') as f:
                    json.dump(event.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store audit event: {e}")
            return False
    
    def get_events(self, 
                   filter_criteria: Optional[AuditFilter] = None,
                   limit: int = 1000,
                   offset: int = 0) -> List[AuditEvent]:
        """Get audit events."""
        events = []
        
        try:
            # Get all log files
            log_files = sorted(self.log_directory.glob("audit_*.jsonl"), reverse=True)
            
            current_offset = 0
            
            for log_file in log_files:
                if len(events) >= limit:
                    break
                
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(events) >= limit:
                            break
                        
                        try:
                            event_data = json.loads(line.strip())
                            event = AuditEvent.from_dict(event_data)
                            
                            if filter_criteria and not filter_criteria.matches(event):
                                continue
                            
                            if current_offset < offset:
                                current_offset += 1
                                continue
                            
                            events.append(event)
                            
                        except (json.JSONDecodeError, ValueError) as e:
                            self.logger.warning(f"Failed to parse event line: {e}")
                            continue
        
        except Exception as e:
            self.logger.error(f"Failed to get audit events: {e}")
        
        return events
    
    def count_events(self, filter_criteria: Optional[AuditFilter] = None) -> int:
        """Count audit events."""
        count = 0
        
        try:
            log_files = self.log_directory.glob("audit_*.jsonl")
            
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            event_data = json.loads(line.strip())
                            event = AuditEvent.from_dict(event_data)
                            
                            if filter_criteria and not filter_criteria.matches(event):
                                continue
                            
                            count += 1
                            
                        except (json.JSONDecodeError, ValueError):
                            continue
        
        except Exception as e:
            self.logger.error(f"Failed to count audit events: {e}")
        
        return count
    
    def delete_events(self, filter_criteria: AuditFilter) -> int:
        """Delete audit events (not supported for file store)."""
        raise NotImplementedError("Delete not supported for file-based audit store")
    
    def get_event_statistics(self, 
                           filter_criteria: Optional[AuditFilter] = None) -> Dict[str, Any]:
        """Get event statistics."""
        stats = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'events_by_status': {},
            'top_users': {},
            'top_ip_addresses': {}
        }
        
        try:
            log_files = self.log_directory.glob("audit_*.jsonl")
            
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            event_data = json.loads(line.strip())
                            event = AuditEvent.from_dict(event_data)
                            
                            if filter_criteria and not filter_criteria.matches(event):
                                continue
                            
                            stats['total_events'] += 1
                            
                            # Count by type
                            event_type = event.event_type.value
                            stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1
                            
                            # Count by severity
                            severity = event.severity.value
                            stats['events_by_severity'][severity] = stats['events_by_severity'].get(severity, 0) + 1
                            
                            # Count by status
                            status = event.status.value
                            stats['events_by_status'][status] = stats['events_by_status'].get(status, 0) + 1
                            
                            # Count by user
                            if event.user_id:
                                stats['top_users'][event.user_id] = stats['top_users'].get(event.user_id, 0) + 1
                            
                            # Count by IP
                            if event.ip_address:
                                stats['top_ip_addresses'][event.ip_address] = stats['top_ip_addresses'].get(event.ip_address, 0) + 1
                            
                        except (json.JSONDecodeError, ValueError):
                            continue
        
        except Exception as e:
            self.logger.error(f"Failed to get event statistics: {e}")
        
        # Sort top lists
        stats['top_users'] = dict(sorted(stats['top_users'].items(), key=lambda x: x[1], reverse=True)[:10])
        stats['top_ip_addresses'] = dict(sorted(stats['top_ip_addresses'].items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats


class AuditLogger:
    """Main audit logger."""
    
    def __init__(self, 
                 store: AuditStore,
                 async_logging: bool = True,
                 queue_size: int = 10000):
        """Initialize audit logger.
        
        Args:
            store: Audit event storage backend
            async_logging: Enable asynchronous logging
            queue_size: Maximum queue size for async logging
        """
        self.store = store
        self.async_logging = async_logging
        self.logger = logging.getLogger("security.audit")
        
        if async_logging:
            self.event_queue = queue.Queue(maxsize=queue_size)
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            self.shutdown_event = threading.Event()
        
        # Event integrity
        self.integrity_key = secrets.token_bytes(32)
    
    def _worker(self) -> None:
        """Background worker for async logging."""
        while not self.shutdown_event.is_set():
            try:
                event = self.event_queue.get(timeout=1.0)
                if event is None:  # Shutdown signal
                    break
                
                self.store.store_event(event)
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Failed to process audit event: {e}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"evt_{secrets.token_urlsafe(16)}"
    
    def _calculate_integrity_hash(self, event: AuditEvent) -> str:
        """Calculate integrity hash for event."""
        event_data = f"{event.event_id}{event.event_type.value}{event.timestamp.isoformat()}{event.user_id or ''}{event.resource or ''}"
        return hmac.new(self.integrity_key, event_data.encode(), hashlib.sha256).hexdigest()
    
    def log_event(self, 
                  event_type: EventType,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  resource: Optional[str] = None,
                  action: Optional[str] = None,
                  status: EventStatus = EventStatus.SUCCESS,
                  severity: EventSeverity = EventSeverity.LOW,
                  message: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  tags: Optional[Set[str]] = None,
                  correlation_id: Optional[str] = None,
                  parent_event_id: Optional[str] = None,
                  duration_ms: Optional[int] = None) -> str:
        """Log an audit event.
        
        Returns:
            Event ID
        """
        event_id = self._generate_event_id()
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            status=status,
            severity=severity,
            message=message,
            details=details or {},
            tags=tags or set(),
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            duration_ms=duration_ms
        )
        
        # Add integrity hash
        event.details['integrity_hash'] = self._calculate_integrity_hash(event)
        
        if self.async_logging:
            try:
                self.event_queue.put_nowait(event)
            except queue.Full:
                self.logger.error("Audit event queue is full, dropping event")
                # Fallback to synchronous logging
                self.store.store_event(event)
        else:
            self.store.store_event(event)
        
        return event_id
    
    def get_events(self, 
                   filter_criteria: Optional[AuditFilter] = None,
                   limit: int = 1000,
                   offset: int = 0) -> List[AuditEvent]:
        """Get audit events."""
        return self.store.get_events(filter_criteria, limit, offset)
    
    def count_events(self, filter_criteria: Optional[AuditFilter] = None) -> int:
        """Count audit events."""
        return self.store.count_events(filter_criteria)
    
    def get_statistics(self, filter_criteria: Optional[AuditFilter] = None) -> Dict[str, Any]:
        """Get audit statistics."""
        return self.store.get_event_statistics(filter_criteria)
    
    def verify_event_integrity(self, event: AuditEvent) -> bool:
        """Verify event integrity."""
        stored_hash = event.details.get('integrity_hash')
        if not stored_hash:
            return False
        
        # Temporarily remove hash for calculation
        temp_details = event.details.copy()
        del temp_details['integrity_hash']
        temp_event = AuditEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            user_id=event.user_id,
            resource=event.resource,
            details=temp_details
        )
        
        calculated_hash = self._calculate_integrity_hash(temp_event)
        return hmac.compare_digest(stored_hash, calculated_hash)
    
    def shutdown(self) -> None:
        """Shutdown audit logger."""
        if self.async_logging:
            self.shutdown_event.set()
            self.event_queue.put(None)  # Signal worker to stop
            self.worker_thread.join(timeout=5.0)


def audit_decorator(event_type: EventType, 
                   severity: EventSeverity = EventSeverity.LOW,
                   include_args: bool = False,
                   include_result: bool = False):
    """Decorator for automatic audit logging.
    
    Args:
        event_type: Type of audit event
        severity: Event severity
        include_args: Include function arguments in details
        include_result: Include function result in details
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Get audit logger from global context or create one
            audit_logger = getattr(wrapper, '_audit_logger', None)
            if not audit_logger:
                return func(*args, **kwargs)
            
            details = {}
            if include_args:
                # Get function signature
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                details['arguments'] = dict(bound_args.arguments)
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    details['result'] = str(result)[:1000]  # Limit size
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                audit_logger.log_event(
                    event_type=event_type,
                    resource=f"{func.__module__}.{func.__name__}",
                    action="function_call",
                    status=EventStatus.SUCCESS,
                    severity=severity,
                    details=details,
                    duration_ms=duration_ms
                )
                
                return result
                
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                details['error'] = str(e)
                
                audit_logger.log_event(
                    event_type=event_type,
                    resource=f"{func.__module__}.{func.__name__}",
                    action="function_call",
                    status=EventStatus.ERROR,
                    severity=EventSeverity.HIGH,
                    details=details,
                    duration_ms=duration_ms
                )
                
                raise
        
        return wrapper
    return decorator


@contextmanager
def audit_context(audit_logger: AuditLogger,
                 event_type: EventType,
                 resource: Optional[str] = None,
                 action: Optional[str] = None,
                 user_id: Optional[str] = None,
                 correlation_id: Optional[str] = None):
    """Context manager for audit logging.
    
    Args:
        audit_logger: Audit logger instance
        event_type: Type of audit event
        resource: Resource being accessed
        action: Action being performed
        user_id: User ID
        correlation_id: Correlation ID
    """
    start_time = time.time()
    event_id = None
    
    try:
        event_id = audit_logger.log_event(
            event_type=event_type,
            resource=resource,
            action=action,
            user_id=user_id,
            correlation_id=correlation_id,
            status=EventStatus.SUCCESS,
            message=f"Started {action or 'operation'}"
        )
        
        yield event_id
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        
        audit_logger.log_event(
            event_type=event_type,
            resource=resource,
            action=action,
            user_id=user_id,
            correlation_id=correlation_id,
            parent_event_id=event_id,
            status=EventStatus.ERROR,
            severity=EventSeverity.HIGH,
            message=f"Failed {action or 'operation'}: {str(e)}",
            duration_ms=duration_ms
        )
        
        raise
    
    finally:
        if event_id:
            duration_ms = int((time.time() - start_time) * 1000)
            
            audit_logger.log_event(
                event_type=event_type,
                resource=resource,
                action=action,
                user_id=user_id,
                correlation_id=correlation_id,
                parent_event_id=event_id,
                status=EventStatus.SUCCESS,
                message=f"Completed {action or 'operation'}",
                duration_ms=duration_ms
            )


# Global audit logger instance
_audit_logger = None


def get_audit_logger(store: Optional[AuditStore] = None) -> AuditLogger:
    """Get global audit logger instance.
    
    Args:
        store: Optional audit store (only used on first call)
        
    Returns:
        Audit logger instance
    """
    global _audit_logger
    if _audit_logger is None:
        if store is None:
            # Default to SQLite store
            store = SQLiteAuditStore("audit.db")
        _audit_logger = AuditLogger(store)
    return _audit_logger


def initialize_audit_logger(store: AuditStore) -> AuditLogger:
    """Initialize global audit logger.
    
    Args:
        store: Audit storage backend
        
    Returns:
        Audit logger instance
    """
    global _audit_logger
    _audit_logger = AuditLogger(store)
    return _audit_logger


def shutdown_audit_logger() -> None:
    """Shutdown global audit logger."""
    global _audit_logger
    if _audit_logger:
        _audit_logger.shutdown()
        _audit_logger = None