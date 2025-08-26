"""Authorization module for AI Bull Ford security layer.

Provides role-based access control (RBAC), permission management,
and resource access authorization.
"""

import json
from typing import Any, Dict, List, Optional, Set, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import re
from pathlib import Path
import sqlite3
import threading
from contextlib import contextmanager
from functools import wraps

from .authentication import User, UserRole


class PermissionType(Enum):
    """Types of permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    CREATE = "create"
    UPDATE = "update"
    MANAGE = "manage"


class ResourceType(Enum):
    """Types of resources."""
    USER = "user"
    MODEL = "model"
    DATASET = "dataset"
    PIPELINE = "pipeline"
    WORKFLOW = "workflow"
    API = "api"
    SYSTEM = "system"
    CONFIG = "config"
    LOG = "log"
    METRIC = "metric"


@dataclass
class Permission:
    """Permission definition."""
    permission_id: str
    name: str
    description: str
    resource_type: ResourceType
    permission_type: PermissionType
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert permission to dictionary."""
        return {
            'permission_id': self.permission_id,
            'name': self.name,
            'description': self.description,
            'resource_type': self.resource_type.value,
            'permission_type': self.permission_type.value,
            'conditions': self.conditions,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Permission':
        """Create permission from dictionary."""
        return cls(
            permission_id=data['permission_id'],
            name=data['name'],
            description=data['description'],
            resource_type=ResourceType(data['resource_type']),
            permission_type=PermissionType(data['permission_type']),
            conditions=data.get('conditions', {}),
            created_at=datetime.fromisoformat(data['created_at'])
        )


@dataclass
class Role:
    """Role definition with permissions."""
    role_id: str
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert role to dictionary."""
        return {
            'role_id': self.role_id,
            'name': self.name,
            'description': self.description,
            'permissions': list(self.permissions),
            'parent_roles': list(self.parent_roles),
            'is_system_role': self.is_system_role,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """Create role from dictionary."""
        return cls(
            role_id=data['role_id'],
            name=data['name'],
            description=data['description'],
            permissions=set(data.get('permissions', [])),
            parent_roles=set(data.get('parent_roles', [])),
            is_system_role=data.get('is_system_role', False),
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )


@dataclass
class AccessRequest:
    """Access request for authorization."""
    user_id: str
    resource_type: ResourceType
    resource_id: Optional[str]
    permission_type: PermissionType
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert access request to dictionary."""
        return {
            'user_id': self.user_id,
            'resource_type': self.resource_type.value,
            'resource_id': self.resource_id,
            'permission_type': self.permission_type.value,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AccessResult:
    """Result of access authorization."""
    granted: bool
    reason: str
    matched_permissions: List[str] = field(default_factory=list)
    conditions_met: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert access result to dictionary."""
        return {
            'granted': self.granted,
            'reason': self.reason,
            'matched_permissions': self.matched_permissions,
            'conditions_met': self.conditions_met,
            'timestamp': self.timestamp.isoformat()
        }


class PermissionStore(ABC):
    """Abstract permission storage interface."""
    
    @abstractmethod
    def create_permission(self, permission: Permission) -> bool:
        """Create a new permission."""
        pass
    
    @abstractmethod
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get permission by ID."""
        pass
    
    @abstractmethod
    def list_permissions(self, resource_type: Optional[ResourceType] = None) -> List[Permission]:
        """List permissions."""
        pass
    
    @abstractmethod
    def update_permission(self, permission: Permission) -> bool:
        """Update permission."""
        pass
    
    @abstractmethod
    def delete_permission(self, permission_id: str) -> bool:
        """Delete permission."""
        pass


class RoleStore(ABC):
    """Abstract role storage interface."""
    
    @abstractmethod
    def create_role(self, role: Role) -> bool:
        """Create a new role."""
        pass
    
    @abstractmethod
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        pass
    
    @abstractmethod
    def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name."""
        pass
    
    @abstractmethod
    def list_roles(self) -> List[Role]:
        """List all roles."""
        pass
    
    @abstractmethod
    def update_role(self, role: Role) -> bool:
        """Update role."""
        pass
    
    @abstractmethod
    def delete_role(self, role_id: str) -> bool:
        """Delete role."""
        pass
    
    @abstractmethod
    def assign_user_role(self, user_id: str, role_id: str) -> bool:
        """Assign role to user."""
        pass
    
    @abstractmethod
    def revoke_user_role(self, user_id: str, role_id: str) -> bool:
        """Revoke role from user."""
        pass
    
    @abstractmethod
    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get roles assigned to user."""
        pass


class SQLiteAuthorizationStore(PermissionStore, RoleStore):
    """SQLite-based authorization storage."""
    
    def __init__(self, db_path: str = "authorization.db"):
        """Initialize SQLite authorization store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Permissions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS permissions (
                    permission_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    resource_type TEXT NOT NULL,
                    permission_type TEXT NOT NULL,
                    conditions TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Roles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    role_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    is_system_role BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Role permissions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role_id TEXT,
                    permission_id TEXT,
                    PRIMARY KEY (role_id, permission_id),
                    FOREIGN KEY (role_id) REFERENCES roles (role_id),
                    FOREIGN KEY (permission_id) REFERENCES permissions (permission_id)
                )
            """)
            
            # Role hierarchy table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS role_hierarchy (
                    parent_role_id TEXT,
                    child_role_id TEXT,
                    PRIMARY KEY (parent_role_id, child_role_id),
                    FOREIGN KEY (parent_role_id) REFERENCES roles (role_id),
                    FOREIGN KEY (child_role_id) REFERENCES roles (role_id)
                )
            """)
            
            # User roles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_roles (
                    user_id TEXT,
                    role_id TEXT,
                    assigned_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, role_id),
                    FOREIGN KEY (role_id) REFERENCES roles (role_id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # Permission methods
    def create_permission(self, permission: Permission) -> bool:
        """Create a new permission."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO permissions 
                        (permission_id, name, description, resource_type, permission_type, conditions, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        permission.permission_id, permission.name, permission.description,
                        permission.resource_type.value, permission.permission_type.value,
                        json.dumps(permission.conditions), permission.created_at.isoformat()
                    ))
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get permission by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM permissions WHERE permission_id = ?", (permission_id,)
            ).fetchone()
            
            if row:
                return self._row_to_permission(row)
            return None
    
    def list_permissions(self, resource_type: Optional[ResourceType] = None) -> List[Permission]:
        """List permissions."""
        with self._get_connection() as conn:
            if resource_type:
                rows = conn.execute(
                    "SELECT * FROM permissions WHERE resource_type = ? ORDER BY name",
                    (resource_type.value,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM permissions ORDER BY name"
                ).fetchall()
            
            return [self._row_to_permission(row) for row in rows]
    
    def update_permission(self, permission: Permission) -> bool:
        """Update permission."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        UPDATE permissions SET
                        name = ?, description = ?, resource_type = ?, permission_type = ?, conditions = ?
                        WHERE permission_id = ?
                    """, (
                        permission.name, permission.description,
                        permission.resource_type.value, permission.permission_type.value,
                        json.dumps(permission.conditions), permission.permission_id
                    ))
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def delete_permission(self, permission_id: str) -> bool:
        """Delete permission."""
        with self._lock:
            with self._get_connection() as conn:
                # Remove from role permissions first
                conn.execute("DELETE FROM role_permissions WHERE permission_id = ?", (permission_id,))
                
                # Delete permission
                cursor = conn.execute("DELETE FROM permissions WHERE permission_id = ?", (permission_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    # Role methods
    def create_role(self, role: Role) -> bool:
        """Create a new role."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    # Create role
                    conn.execute("""
                        INSERT INTO roles 
                        (role_id, name, description, is_system_role, created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        role.role_id, role.name, role.description,
                        role.is_system_role, role.created_at.isoformat(),
                        json.dumps(role.metadata)
                    ))
                    
                    # Add permissions
                    for permission_id in role.permissions:
                        conn.execute(
                            "INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                            (role.role_id, permission_id)
                        )
                    
                    # Add parent roles
                    for parent_role_id in role.parent_roles:
                        conn.execute(
                            "INSERT INTO role_hierarchy (parent_role_id, child_role_id) VALUES (?, ?)",
                            (parent_role_id, role.role_id)
                        )
                    
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM roles WHERE role_id = ?", (role_id,)
            ).fetchone()
            
            if row:
                return self._row_to_role(row, conn)
            return None
    
    def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM roles WHERE name = ?", (name,)
            ).fetchone()
            
            if row:
                return self._row_to_role(row, conn)
            return None
    
    def list_roles(self) -> List[Role]:
        """List all roles."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM roles ORDER BY name").fetchall()
            return [self._row_to_role(row, conn) for row in rows]
    
    def update_role(self, role: Role) -> bool:
        """Update role."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    # Update role
                    conn.execute("""
                        UPDATE roles SET
                        name = ?, description = ?, is_system_role = ?, metadata = ?
                        WHERE role_id = ?
                    """, (
                        role.name, role.description, role.is_system_role,
                        json.dumps(role.metadata), role.role_id
                    ))
                    
                    # Update permissions
                    conn.execute("DELETE FROM role_permissions WHERE role_id = ?", (role.role_id,))
                    for permission_id in role.permissions:
                        conn.execute(
                            "INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                            (role.role_id, permission_id)
                        )
                    
                    # Update parent roles
                    conn.execute("DELETE FROM role_hierarchy WHERE child_role_id = ?", (role.role_id,))
                    for parent_role_id in role.parent_roles:
                        conn.execute(
                            "INSERT INTO role_hierarchy (parent_role_id, child_role_id) VALUES (?, ?)",
                            (parent_role_id, role.role_id)
                        )
                    
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def delete_role(self, role_id: str) -> bool:
        """Delete role."""
        with self._lock:
            with self._get_connection() as conn:
                # Remove role assignments
                conn.execute("DELETE FROM user_roles WHERE role_id = ?", (role_id,))
                
                # Remove role permissions
                conn.execute("DELETE FROM role_permissions WHERE role_id = ?", (role_id,))
                
                # Remove from hierarchy
                conn.execute("DELETE FROM role_hierarchy WHERE parent_role_id = ? OR child_role_id = ?", 
                           (role_id, role_id))
                
                # Delete role
                cursor = conn.execute("DELETE FROM roles WHERE role_id = ?", (role_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    def assign_user_role(self, user_id: str, role_id: str) -> bool:
        """Assign role to user."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute(
                        "INSERT INTO user_roles (user_id, role_id, assigned_at) VALUES (?, ?, ?)",
                        (user_id, role_id, datetime.now().isoformat())
                    )
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def revoke_user_role(self, user_id: str, role_id: str) -> bool:
        """Revoke role from user."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM user_roles WHERE user_id = ? AND role_id = ?",
                    (user_id, role_id)
                )
                conn.commit()
                return cursor.rowcount > 0
    
    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get roles assigned to user."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT r.* FROM roles r
                JOIN user_roles ur ON r.role_id = ur.role_id
                WHERE ur.user_id = ?
                ORDER BY r.name
            """, (user_id,)).fetchall()
            
            return [self._row_to_role(row, conn) for row in rows]
    
    def _row_to_permission(self, row) -> Permission:
        """Convert database row to Permission object."""
        return Permission(
            permission_id=row['permission_id'],
            name=row['name'],
            description=row['description'],
            resource_type=ResourceType(row['resource_type']),
            permission_type=PermissionType(row['permission_type']),
            conditions=json.loads(row['conditions']) if row['conditions'] else {},
            created_at=datetime.fromisoformat(row['created_at'])
        )
    
    def _row_to_role(self, row, conn) -> Role:
        """Convert database row to Role object."""
        role_id = row['role_id']
        
        # Get permissions
        permission_rows = conn.execute(
            "SELECT permission_id FROM role_permissions WHERE role_id = ?",
            (role_id,)
        ).fetchall()
        permissions = {row['permission_id'] for row in permission_rows}
        
        # Get parent roles
        parent_rows = conn.execute(
            "SELECT parent_role_id FROM role_hierarchy WHERE child_role_id = ?",
            (role_id,)
        ).fetchall()
        parent_roles = {row['parent_role_id'] for row in parent_rows}
        
        return Role(
            role_id=role_id,
            name=row['name'],
            description=row['description'],
            permissions=permissions,
            parent_roles=parent_roles,
            is_system_role=bool(row['is_system_role']),
            created_at=datetime.fromisoformat(row['created_at']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


class ConditionEvaluator:
    """Evaluates permission conditions."""
    
    def __init__(self):
        """Initialize condition evaluator."""
        self.logger = logging.getLogger("security.conditions")
    
    def evaluate(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate permission conditions against context.
        
        Args:
            conditions: Permission conditions
            context: Request context
            
        Returns:
            True if conditions are met
        """
        if not conditions:
            return True
        
        try:
            for condition_type, condition_value in conditions.items():
                if not self._evaluate_condition(condition_type, condition_value, context):
                    return False
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating conditions: {e}")
            return False
    
    def _evaluate_condition(self, condition_type: str, condition_value: Any, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition.
        
        Args:
            condition_type: Type of condition
            condition_value: Condition value
            context: Request context
            
        Returns:
            True if condition is met
        """
        if condition_type == "time_range":
            return self._evaluate_time_range(condition_value, context)
        elif condition_type == "ip_whitelist":
            return self._evaluate_ip_whitelist(condition_value, context)
        elif condition_type == "resource_owner":
            return self._evaluate_resource_owner(condition_value, context)
        elif condition_type == "custom":
            return self._evaluate_custom_condition(condition_value, context)
        else:
            self.logger.warning(f"Unknown condition type: {condition_type}")
            return True
    
    def _evaluate_time_range(self, time_range: Dict[str, str], context: Dict[str, Any]) -> bool:
        """Evaluate time range condition."""
        try:
            current_time = datetime.now().time()
            start_time = datetime.strptime(time_range['start'], '%H:%M').time()
            end_time = datetime.strptime(time_range['end'], '%H:%M').time()
            
            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:  # Overnight range
                return current_time >= start_time or current_time <= end_time
                
        except (KeyError, ValueError) as e:
            self.logger.error(f"Invalid time range condition: {e}")
            return False
    
    def _evaluate_ip_whitelist(self, ip_list: List[str], context: Dict[str, Any]) -> bool:
        """Evaluate IP whitelist condition."""
        client_ip = context.get('client_ip')
        if not client_ip:
            return False
        
        return client_ip in ip_list
    
    def _evaluate_resource_owner(self, owner_field: str, context: Dict[str, Any]) -> bool:
        """Evaluate resource owner condition."""
        user_id = context.get('user_id')
        resource_owner = context.get(owner_field)
        
        return user_id == resource_owner
    
    def _evaluate_custom_condition(self, condition_code: str, context: Dict[str, Any]) -> bool:
        """Evaluate custom condition (be very careful with this)."""
        # This is potentially dangerous - only use with trusted conditions
        # In production, consider using a safer expression evaluator
        try:
            # Create a restricted environment
            safe_globals = {
                '__builtins__': {},
                'context': context,
                'datetime': datetime,
                're': re
            }
            
            return bool(eval(condition_code, safe_globals))
            
        except Exception as e:
            self.logger.error(f"Error evaluating custom condition: {e}")
            return False


class PermissionManager:
    """Manages permissions."""
    
    def __init__(self, permission_store: PermissionStore):
        """Initialize permission manager.
        
        Args:
            permission_store: Permission storage backend
        """
        self.permission_store = permission_store
        self.logger = logging.getLogger("security.permissions")
    
    def create_permission(self, 
                         name: str,
                         description: str,
                         resource_type: ResourceType,
                         permission_type: PermissionType,
                         conditions: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Create a new permission.
        
        Args:
            name: Permission name
            description: Permission description
            resource_type: Resource type
            permission_type: Permission type
            conditions: Optional conditions
            
        Returns:
            Tuple of (success, message)
        """
        import secrets
        
        permission = Permission(
            permission_id=secrets.token_urlsafe(16),
            name=name,
            description=description,
            resource_type=resource_type,
            permission_type=permission_type,
            conditions=conditions or {}
        )
        
        success = self.permission_store.create_permission(permission)
        if success:
            self.logger.info(f"Permission created: {name}")
            return True, "Permission created successfully"
        else:
            return False, "Failed to create permission"
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get permission by ID."""
        return self.permission_store.get_permission(permission_id)
    
    def list_permissions(self, resource_type: Optional[ResourceType] = None) -> List[Permission]:
        """List permissions."""
        return self.permission_store.list_permissions(resource_type)
    
    def update_permission(self, permission: Permission) -> Tuple[bool, str]:
        """Update permission."""
        success = self.permission_store.update_permission(permission)
        if success:
            self.logger.info(f"Permission updated: {permission.name}")
            return True, "Permission updated successfully"
        else:
            return False, "Failed to update permission"
    
    def delete_permission(self, permission_id: str) -> Tuple[bool, str]:
        """Delete permission."""
        success = self.permission_store.delete_permission(permission_id)
        if success:
            self.logger.info(f"Permission deleted: {permission_id}")
            return True, "Permission deleted successfully"
        else:
            return False, "Failed to delete permission"


class RoleManager:
    """Manages roles and role assignments."""
    
    def __init__(self, role_store: RoleStore):
        """Initialize role manager.
        
        Args:
            role_store: Role storage backend
        """
        self.role_store = role_store
        self.logger = logging.getLogger("security.roles")
    
    def create_role(self, 
                   name: str,
                   description: str,
                   permissions: Optional[Set[str]] = None,
                   parent_roles: Optional[Set[str]] = None,
                   is_system_role: bool = False) -> Tuple[bool, str]:
        """Create a new role.
        
        Args:
            name: Role name
            description: Role description
            permissions: Permission IDs
            parent_roles: Parent role IDs
            is_system_role: Whether this is a system role
            
        Returns:
            Tuple of (success, message)
        """
        import secrets
        
        role = Role(
            role_id=secrets.token_urlsafe(16),
            name=name,
            description=description,
            permissions=permissions or set(),
            parent_roles=parent_roles or set(),
            is_system_role=is_system_role
        )
        
        success = self.role_store.create_role(role)
        if success:
            self.logger.info(f"Role created: {name}")
            return True, "Role created successfully"
        else:
            return False, "Failed to create role"
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        return self.role_store.get_role(role_id)
    
    def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name."""
        return self.role_store.get_role_by_name(name)
    
    def list_roles(self) -> List[Role]:
        """List all roles."""
        return self.role_store.list_roles()
    
    def update_role(self, role: Role) -> Tuple[bool, str]:
        """Update role."""
        success = self.role_store.update_role(role)
        if success:
            self.logger.info(f"Role updated: {role.name}")
            return True, "Role updated successfully"
        else:
            return False, "Failed to update role"
    
    def delete_role(self, role_id: str) -> Tuple[bool, str]:
        """Delete role."""
        success = self.role_store.delete_role(role_id)
        if success:
            self.logger.info(f"Role deleted: {role_id}")
            return True, "Role deleted successfully"
        else:
            return False, "Failed to delete role"
    
    def assign_user_role(self, user_id: str, role_id: str) -> Tuple[bool, str]:
        """Assign role to user."""
        success = self.role_store.assign_user_role(user_id, role_id)
        if success:
            self.logger.info(f"Role {role_id} assigned to user {user_id}")
            return True, "Role assigned successfully"
        else:
            return False, "Failed to assign role"
    
    def revoke_user_role(self, user_id: str, role_id: str) -> Tuple[bool, str]:
        """Revoke role from user."""
        success = self.role_store.revoke_user_role(user_id, role_id)
        if success:
            self.logger.info(f"Role {role_id} revoked from user {user_id}")
            return True, "Role revoked successfully"
        else:
            return False, "Failed to revoke role"
    
    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get roles assigned to user."""
        return self.role_store.get_user_roles(user_id)
    
    def get_effective_permissions(self, user_id: str) -> Set[str]:
        """Get all effective permissions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permission IDs
        """
        permissions = set()
        roles = self.get_user_roles(user_id)
        
        # Collect permissions from all roles (including inherited)
        for role in roles:
            permissions.update(role.permissions)
            permissions.update(self._get_inherited_permissions(role))
        
        return permissions
    
    def _get_inherited_permissions(self, role: Role) -> Set[str]:
        """Get permissions inherited from parent roles.
        
        Args:
            role: Role to get inherited permissions for
            
        Returns:
            Set of inherited permission IDs
        """
        inherited_permissions = set()
        
        for parent_role_id in role.parent_roles:
            parent_role = self.role_store.get_role(parent_role_id)
            if parent_role:
                inherited_permissions.update(parent_role.permissions)
                inherited_permissions.update(self._get_inherited_permissions(parent_role))
        
        return inherited_permissions


class AuthorizationManager:
    """Main authorization manager."""
    
    def __init__(self,
                 permission_store: Optional[PermissionStore] = None,
                 role_store: Optional[RoleStore] = None):
        """Initialize authorization manager.
        
        Args:
            permission_store: Permission storage backend
            role_store: Role storage backend
        """
        # Use SQLite stores by default
        auth_store = SQLiteAuthorizationStore()
        
        self.permission_store = permission_store or auth_store
        self.role_store = role_store or auth_store
        
        self.permission_manager = PermissionManager(self.permission_store)
        self.role_manager = RoleManager(self.role_store)
        self.condition_evaluator = ConditionEvaluator()
        
        self.logger = logging.getLogger("security.authorization")
        
        # Initialize default permissions and roles
        self._initialize_defaults()
    
    def authorize(self, request: AccessRequest) -> AccessResult:
        """Authorize an access request.
        
        Args:
            request: Access request
            
        Returns:
            Authorization result
        """
        try:
            # Get user's effective permissions
            user_permissions = self.role_manager.get_effective_permissions(request.user_id)
            
            # Find matching permissions
            matching_permissions = []
            conditions_met = True
            
            for permission_id in user_permissions:
                permission = self.permission_store.get_permission(permission_id)
                if not permission:
                    continue
                
                # Check if permission matches the request
                if (permission.resource_type == request.resource_type and
                    permission.permission_type == request.permission_type):
                    
                    # Evaluate conditions
                    if self.condition_evaluator.evaluate(permission.conditions, request.context):
                        matching_permissions.append(permission_id)
                    else:
                        conditions_met = False
            
            # Grant access if any matching permission found
            granted = len(matching_permissions) > 0
            
            if granted:
                reason = f"Access granted via permissions: {', '.join(matching_permissions)}"
            else:
                if not user_permissions:
                    reason = "No permissions assigned to user"
                elif not conditions_met:
                    reason = "Permission conditions not met"
                else:
                    reason = "No matching permissions found"
            
            result = AccessResult(
                granted=granted,
                reason=reason,
                matched_permissions=matching_permissions,
                conditions_met=conditions_met
            )
            
            self.logger.info(f"Authorization result for user {request.user_id}: {granted} - {reason}")
            return result
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return AccessResult(
                granted=False,
                reason=f"Authorization error: {str(e)}"
            )
    
    def check_permission(self, 
                        user_id: str,
                        resource_type: ResourceType,
                        permission_type: PermissionType,
                        resource_id: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user has permission for a resource.
        
        Args:
            user_id: User ID
            resource_type: Resource type
            permission_type: Permission type
            resource_id: Optional resource ID
            context: Optional context
            
        Returns:
            True if permission granted
        """
        request = AccessRequest(
            user_id=user_id,
            resource_type=resource_type,
            permission_type=permission_type,
            resource_id=resource_id,
            context=context or {}
        )
        
        result = self.authorize(request)
        return result.granted
    
    def _initialize_defaults(self) -> None:
        """Initialize default permissions and roles."""
        try:
            # Create default permissions if they don't exist
            default_permissions = [
                ("user.read", "Read user information", ResourceType.USER, PermissionType.READ),
                ("user.write", "Modify user information", ResourceType.USER, PermissionType.WRITE),
                ("user.delete", "Delete users", ResourceType.USER, PermissionType.DELETE),
                ("model.read", "Read models", ResourceType.MODEL, PermissionType.READ),
                ("model.write", "Modify models", ResourceType.MODEL, PermissionType.WRITE),
                ("model.execute", "Execute models", ResourceType.MODEL, PermissionType.EXECUTE),
                ("system.admin", "System administration", ResourceType.SYSTEM, PermissionType.ADMIN),
            ]
            
            for name, desc, resource_type, permission_type in default_permissions:
                existing = None
                for perm in self.permission_store.list_permissions():
                    if perm.name == name:
                        existing = perm
                        break
                
                if not existing:
                    self.permission_manager.create_permission(name, desc, resource_type, permission_type)
            
            # Create default roles if they don't exist
            admin_role = self.role_manager.get_role_by_name("admin")
            if not admin_role:
                # Get all permission IDs
                all_permissions = {p.permission_id for p in self.permission_store.list_permissions()}
                self.role_manager.create_role(
                    "admin", 
                    "System administrator with full access",
                    all_permissions,
                    is_system_role=True
                )
            
            user_role = self.role_manager.get_role_by_name("user")
            if not user_role:
                # Get basic user permissions
                user_permissions = set()
                for perm in self.permission_store.list_permissions():
                    if perm.name in ["user.read", "model.read", "model.execute"]:
                        user_permissions.add(perm.permission_id)
                
                self.role_manager.create_role(
                    "user",
                    "Standard user with basic access",
                    user_permissions,
                    is_system_role=True
                )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize defaults: {e}")


def require_permission(resource_type: ResourceType, permission_type: PermissionType):
    """Decorator to require permission for a function.
    
    Args:
        resource_type: Required resource type
        permission_type: Required permission type
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is a simplified example - in practice, you'd need to
            # extract user context from the request/session
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None)
            
            if not user_id:
                raise PermissionError("User not authenticated")
            
            # Get authorization manager from somewhere (dependency injection, global, etc.)
            auth_manager = kwargs.get('auth_manager')
            if not auth_manager:
                raise RuntimeError("Authorization manager not available")
            
            # Check permission
            if not auth_manager.check_permission(user_id, resource_type, permission_type):
                raise PermissionError(f"Permission denied: {resource_type.value}.{permission_type.value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator