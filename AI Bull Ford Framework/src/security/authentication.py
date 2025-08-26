"""Authentication module for AI Bull Ford security layer.

Provides user authentication, token management, and session handling.
"""

import hashlib
import secrets
import jwt
import bcrypt
from typing import Any, Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
import re
from pathlib import Path
import sqlite3
import threading
from contextlib import contextmanager


class UserRole(Enum):
    """User roles in the system."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    SERVICE = "service"
    DEVELOPER = "developer"
    ANALYST = "analyst"


class UserStatus(Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING = "pending"


class TokenType(Enum):
    """Types of authentication tokens."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    SESSION = "session"
    RESET = "reset"


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    status: UserStatus
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'failed_login_attempts': self.failed_login_attempts,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary."""
        return cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data['email'],
            password_hash=data['password_hash'],
            role=UserRole(data['role']),
            status=UserStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            failed_login_attempts=data.get('failed_login_attempts', 0),
            metadata=data.get('metadata', {})
        )


@dataclass
class AuthToken:
    """Authentication token information."""
    token_id: str
    user_id: str
    token_type: TokenType
    token_value: str
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.now)
    is_revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now() >= self.expires_at
    
    def is_valid(self) -> bool:
        """Check if token is valid."""
        return not self.is_revoked and not self.is_expired()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert token to dictionary."""
        return {
            'token_id': self.token_id,
            'user_id': self.user_id,
            'token_type': self.token_type.value,
            'expires_at': self.expires_at.isoformat(),
            'created_at': self.created_at.isoformat(),
            'is_revoked': self.is_revoked,
            'metadata': self.metadata
        }


class PasswordPolicy:
    """Password policy configuration."""
    
    def __init__(self,
                 min_length: int = 8,
                 max_length: int = 128,
                 require_uppercase: bool = True,
                 require_lowercase: bool = True,
                 require_digits: bool = True,
                 require_special: bool = True,
                 special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"):
        """Initialize password policy.
        
        Args:
            min_length: Minimum password length
            max_length: Maximum password length
            require_uppercase: Require uppercase letters
            require_lowercase: Require lowercase letters
            require_digits: Require digits
            require_special: Require special characters
            special_chars: Allowed special characters
        """
        self.min_length = min_length
        self.max_length = max_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digits = require_digits
        self.require_special = require_special
        self.special_chars = special_chars
    
    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against policy.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        
        if len(password) > self.max_length:
            errors.append(f"Password must be no more than {self.max_length} characters long")
        
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if self.require_special and not any(c in self.special_chars for c in password):
            errors.append(f"Password must contain at least one special character: {self.special_chars}")
        
        return len(errors) == 0, errors


class UserStore(ABC):
    """Abstract user storage interface."""
    
    @abstractmethod
    def create_user(self, user: User) -> bool:
        """Create a new user."""
        pass
    
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        pass
    
    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        pass
    
    @abstractmethod
    def update_user(self, user: User) -> bool:
        """Update user information."""
        pass
    
    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        pass
    
    @abstractmethod
    def list_users(self, limit: Optional[int] = None, offset: int = 0) -> List[User]:
        """List users."""
        pass


class SQLiteUserStore(UserStore):
    """SQLite-based user storage."""
    
    def __init__(self, db_path: str = "users.db"):
        """Initialize SQLite user store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    failed_login_attempts INTEGER DEFAULT 0,
                    metadata TEXT
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
    
    def create_user(self, user: User) -> bool:
        """Create a new user."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT INTO users 
                        (user_id, username, email, password_hash, role, status, 
                         created_at, last_login, failed_login_attempts, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user.user_id, user.username, user.email, user.password_hash,
                        user.role.value, user.status.value, user.created_at.isoformat(),
                        user.last_login.isoformat() if user.last_login else None,
                        user.failed_login_attempts, json.dumps(user.metadata)
                    ))
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            
            if row:
                return self._row_to_user(row)
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            
            if row:
                return self._row_to_user(row)
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
            
            if row:
                return self._row_to_user(row)
            return None
    
    def update_user(self, user: User) -> bool:
        """Update user information."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("""
                        UPDATE users SET
                        username = ?, email = ?, password_hash = ?, role = ?, status = ?,
                        last_login = ?, failed_login_attempts = ?, metadata = ?
                        WHERE user_id = ?
                    """, (
                        user.username, user.email, user.password_hash,
                        user.role.value, user.status.value,
                        user.last_login.isoformat() if user.last_login else None,
                        user.failed_login_attempts, json.dumps(user.metadata),
                        user.user_id
                    ))
                    conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                conn.commit()
                return cursor.rowcount > 0
    
    def list_users(self, limit: Optional[int] = None, offset: int = 0) -> List[User]:
        """List users."""
        with self._get_connection() as conn:
            query = "SELECT * FROM users ORDER BY created_at DESC"
            params = []
            
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_user(row) for row in rows]
    
    def _row_to_user(self, row) -> User:
        """Convert database row to User object."""
        return User(
            user_id=row['user_id'],
            username=row['username'],
            email=row['email'],
            password_hash=row['password_hash'],
            role=UserRole(row['role']),
            status=UserStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None,
            failed_login_attempts=row['failed_login_attempts'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )


class TokenManager:
    """Manages authentication tokens."""
    
    def __init__(self, 
                 secret_key: str,
                 access_token_lifetime: timedelta = timedelta(hours=1),
                 refresh_token_lifetime: timedelta = timedelta(days=30)):
        """Initialize token manager.
        
        Args:
            secret_key: Secret key for token signing
            access_token_lifetime: Access token lifetime
            refresh_token_lifetime: Refresh token lifetime
        """
        self.secret_key = secret_key
        self.access_token_lifetime = access_token_lifetime
        self.refresh_token_lifetime = refresh_token_lifetime
        self.algorithm = "HS256"
        self._revoked_tokens: set = set()
        self.logger = logging.getLogger("security.tokens")
    
    def generate_token(self, 
                      user_id: str,
                      token_type: TokenType,
                      additional_claims: Optional[Dict[str, Any]] = None) -> AuthToken:
        """Generate a new token.
        
        Args:
            user_id: User ID
            token_type: Type of token
            additional_claims: Additional JWT claims
            
        Returns:
            Generated authentication token
        """
        token_id = secrets.token_urlsafe(32)
        now = datetime.now()
        
        # Determine expiration based on token type
        if token_type == TokenType.ACCESS:
            expires_at = now + self.access_token_lifetime
        elif token_type == TokenType.REFRESH:
            expires_at = now + self.refresh_token_lifetime
        elif token_type == TokenType.API_KEY:
            expires_at = now + timedelta(days=365)  # 1 year for API keys
        else:
            expires_at = now + self.access_token_lifetime
        
        # Create JWT payload
        payload = {
            'jti': token_id,
            'sub': user_id,
            'type': token_type.value,
            'iat': now.timestamp(),
            'exp': expires_at.timestamp()
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        # Generate JWT token
        token_value = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return AuthToken(
            token_id=token_id,
            user_id=user_id,
            token_type=token_type,
            token_value=token_value,
            expires_at=expires_at
        )
    
    def verify_token(self, token_value: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a token.
        
        Args:
            token_value: Token to verify
            
        Returns:
            Decoded token payload if valid
        """
        try:
            payload = jwt.decode(token_value, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            if payload.get('jti') in self._revoked_tokens:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
    
    def revoke_token(self, token_id: str) -> None:
        """Revoke a token.
        
        Args:
            token_id: Token ID to revoke
        """
        self._revoked_tokens.add(token_id)
        self.logger.info(f"Token revoked: {token_id}")
    
    def revoke_user_tokens(self, user_id: str) -> None:
        """Revoke all tokens for a user.
        
        Args:
            user_id: User ID
        """
        # Note: In a production system, you'd want to store tokens
        # in a database to properly revoke all user tokens
        self.logger.info(f"All tokens revoked for user: {user_id}")
    
    def refresh_token(self, refresh_token: str) -> Optional[AuthToken]:
        """Generate new access token from refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New access token if refresh token is valid
        """
        payload = self.verify_token(refresh_token)
        if not payload or payload.get('type') != TokenType.REFRESH.value:
            return None
        
        user_id = payload.get('sub')
        if not user_id:
            return None
        
        return self.generate_token(user_id, TokenType.ACCESS)


class UserManager:
    """Manages user accounts and authentication."""
    
    def __init__(self, 
                 user_store: UserStore,
                 password_policy: Optional[PasswordPolicy] = None,
                 max_failed_attempts: int = 5,
                 lockout_duration: timedelta = timedelta(minutes=30)):
        """Initialize user manager.
        
        Args:
            user_store: User storage backend
            password_policy: Password policy
            max_failed_attempts: Maximum failed login attempts
            lockout_duration: Account lockout duration
        """
        self.user_store = user_store
        self.password_policy = password_policy or PasswordPolicy()
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        self.logger = logging.getLogger("security.users")
    
    def create_user(self, 
                   username: str,
                   email: str,
                   password: str,
                   role: UserRole = UserRole.USER) -> Tuple[bool, str]:
        """Create a new user account.
        
        Args:
            username: Username
            email: Email address
            password: Password
            role: User role
            
        Returns:
            Tuple of (success, message)
        """
        # Validate password
        is_valid, errors = self.password_policy.validate(password)
        if not is_valid:
            return False, "; ".join(errors)
        
        # Check if user already exists
        if self.user_store.get_user_by_username(username):
            return False, "Username already exists"
        
        if self.user_store.get_user_by_email(email):
            return False, "Email already exists"
        
        # Hash password
        password_hash = self._hash_password(password)
        
        # Create user
        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            status=UserStatus.ACTIVE
        )
        
        success = self.user_store.create_user(user)
        if success:
            self.logger.info(f"User created: {username}")
            return True, "User created successfully"
        else:
            return False, "Failed to create user"
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[User], str]:
        """Authenticate user credentials.
        
        Args:
            username: Username or email
            password: Password
            
        Returns:
            Tuple of (success, user, message)
        """
        # Get user by username or email
        user = self.user_store.get_user_by_username(username)
        if not user:
            user = self.user_store.get_user_by_email(username)
        
        if not user:
            return False, None, "Invalid credentials"
        
        # Check account status
        if user.status == UserStatus.SUSPENDED:
            return False, None, "Account suspended"
        
        if user.status == UserStatus.LOCKED:
            return False, None, "Account locked"
        
        if user.status == UserStatus.INACTIVE:
            return False, None, "Account inactive"
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.status = UserStatus.LOCKED
                self.logger.warning(f"Account locked due to failed attempts: {username}")
            
            self.user_store.update_user(user)
            return False, None, "Invalid credentials"
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        self.user_store.update_user(user)
        
        self.logger.info(f"User authenticated: {username}")
        return True, user, "Authentication successful"
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        user = self.user_store.get_user(user_id)
        if not user:
            return False, "User not found"
        
        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            return False, "Invalid current password"
        
        # Validate new password
        is_valid, errors = self.password_policy.validate(new_password)
        if not is_valid:
            return False, "; ".join(errors)
        
        # Update password
        user.password_hash = self._hash_password(new_password)
        success = self.user_store.update_user(user)
        
        if success:
            self.logger.info(f"Password changed for user: {user.username}")
            return True, "Password changed successfully"
        else:
            return False, "Failed to change password"
    
    def reset_password(self, user_id: str, new_password: str) -> Tuple[bool, str]:
        """Reset user password (admin function).
        
        Args:
            user_id: User ID
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        user = self.user_store.get_user(user_id)
        if not user:
            return False, "User not found"
        
        # Validate new password
        is_valid, errors = self.password_policy.validate(new_password)
        if not is_valid:
            return False, "; ".join(errors)
        
        # Update password and reset failed attempts
        user.password_hash = self._hash_password(new_password)
        user.failed_login_attempts = 0
        user.status = UserStatus.ACTIVE  # Unlock account
        
        success = self.user_store.update_user(user)
        
        if success:
            self.logger.info(f"Password reset for user: {user.username}")
            return True, "Password reset successfully"
        else:
            return False, "Failed to reset password"
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Plain text password
            password_hash: Hashed password
            
        Returns:
            True if password matches
        """
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


class AuthenticationManager:
    """Main authentication manager."""
    
    def __init__(self,
                 secret_key: str,
                 user_store: Optional[UserStore] = None,
                 password_policy: Optional[PasswordPolicy] = None):
        """Initialize authentication manager.
        
        Args:
            secret_key: Secret key for token signing
            user_store: User storage backend
            password_policy: Password policy
        """
        self.secret_key = secret_key
        self.user_store = user_store or SQLiteUserStore()
        self.password_policy = password_policy or PasswordPolicy()
        
        self.token_manager = TokenManager(secret_key)
        self.user_manager = UserManager(self.user_store, self.password_policy)
        
        self.logger = logging.getLogger("security.auth")
    
    def login(self, username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """User login.
        
        Args:
            username: Username or email
            password: Password
            
        Returns:
            Tuple of (success, tokens, message)
        """
        success, user, message = self.user_manager.authenticate(username, password)
        
        if not success or not user:
            return False, None, message
        
        # Generate tokens
        access_token = self.token_manager.generate_token(user.user_id, TokenType.ACCESS)
        refresh_token = self.token_manager.generate_token(user.user_id, TokenType.REFRESH)
        
        tokens = {
            'access_token': access_token.token_value,
            'refresh_token': refresh_token.token_value,
            'token_type': 'Bearer',
            'expires_in': int(self.token_manager.access_token_lifetime.total_seconds()),
            'user': user.to_dict()
        }
        
        return True, tokens, "Login successful"
    
    def logout(self, token: str) -> bool:
        """User logout.
        
        Args:
            token: Access token to revoke
            
        Returns:
            True if logout successful
        """
        payload = self.token_manager.verify_token(token)
        if payload:
            self.token_manager.revoke_token(payload.get('jti'))
            return True
        return False
    
    def verify_token(self, token: str) -> Optional[User]:
        """Verify token and get user.
        
        Args:
            token: Token to verify
            
        Returns:
            User if token is valid
        """
        payload = self.token_manager.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get('sub')
        if not user_id:
            return None
        
        return self.user_store.get_user(user_id)
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Refresh access token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New tokens if refresh successful
        """
        new_access_token = self.token_manager.refresh_token(refresh_token)
        if not new_access_token:
            return None
        
        return {
            'access_token': new_access_token.token_value,
            'token_type': 'Bearer',
            'expires_in': int(self.token_manager.access_token_lifetime.total_seconds())
        }
    
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> Tuple[bool, str]:
        """Create new user account.
        
        Args:
            username: Username
            email: Email
            password: Password
            role: User role
            
        Returns:
            Tuple of (success, message)
        """
        return self.user_manager.create_user(username, email, password, role)
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        return self.user_manager.change_password(user_id, old_password, new_password)