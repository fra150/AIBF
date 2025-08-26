"""Common API components for AI Bull Ford.

This module provides shared components used across all API types including:
- Authentication and authorization
- Rate limiting and throttling
- Request/response validation
- Error handling and responses
- Middleware and decorators
- API versioning
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from uuid import uuid4

import hashlib
import hmac
import jwt
from pydantic import BaseModel, ValidationError


class APIVersion(Enum):
    """API version enumeration."""
    V1 = "v1"
    V2 = "v2"
    BETA = "beta"
    ALPHA = "alpha"


class AuthMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    BEARER = "bearer"
    CUSTOM = "custom"


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ErrorCode(Enum):
    """API error codes."""
    # Authentication errors
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    
    # Validation errors
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    
    # Server errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    
    # Resource errors
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    GONE = "GONE"


@dataclass
class APIError:
    """API error representation."""
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error': {
                'code': self.code.value,
                'message': self.message,
                'details': self.details,
                'timestamp': self.timestamp.isoformat(),
                'request_id': self.request_id
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class APIResponse:
    """Standard API response."""
    success: bool
    data: Optional[Any] = None
    error: Optional[APIError] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'success': self.success,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
        
        if self.success and self.data is not None:
            result['data'] = self.data
        elif not self.success and self.error:
            result.update(self.error.to_dict())
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def success_response(cls, data: Any = None, metadata: Dict[str, Any] = None) -> 'APIResponse':
        """Create success response."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {}
        )
    
    @classmethod
    def error_response(cls, error: APIError, metadata: Dict[str, Any] = None) -> 'APIResponse':
        """Create error response."""
        return cls(
            success=False,
            error=error,
            metadata=metadata or {}
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    
    # Advanced settings
    sliding_window: bool = True
    per_user: bool = True
    per_ip: bool = True
    
    # Penalties
    penalty_duration: timedelta = timedelta(minutes=5)
    progressive_penalty: bool = True


@dataclass
class AuthConfig:
    """Authentication configuration."""
    method: AuthMethod = AuthMethod.JWT
    secret_key: str = ""
    algorithm: str = "HS256"
    token_expiry: timedelta = timedelta(hours=24)
    
    # JWT specific
    issuer: str = "ai-bull-ford"
    audience: str = "api"
    
    # API key specific
    api_key_header: str = "X-API-Key"
    api_key_query_param: str = "api_key"
    
    # OAuth2 specific
    oauth2_provider: str = ""
    oauth2_client_id: str = ""
    oauth2_client_secret: str = ""


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        
        # Storage for rate limiting data
        self.user_requests: Dict[str, deque] = defaultdict(deque)
        self.ip_requests: Dict[str, deque] = defaultdict(deque)
        self.penalties: Dict[str, datetime] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, identifier: str, ip_address: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed."""
        current_time = datetime.now()
        
        # Check penalties
        if identifier in self.penalties:
            if current_time < self.penalties[identifier]:
                return False, {
                    'reason': 'penalty_active',
                    'retry_after': (self.penalties[identifier] - current_time).total_seconds()
                }
            else:
                del self.penalties[identifier]
        
        # Check user-based limits
        if self.config.per_user:
            allowed, info = self._check_user_limits(identifier, current_time)
            if not allowed:
                self._apply_penalty(identifier)
                return False, info
        
        # Check IP-based limits
        if self.config.per_ip and ip_address:
            allowed, info = self._check_ip_limits(ip_address, current_time)
            if not allowed:
                self._apply_penalty(ip_address)
                return False, info
        
        # Record request
        self._record_request(identifier, ip_address, current_time)
        
        return True, {'remaining': self._get_remaining_requests(identifier)}
    
    def _check_user_limits(self, user_id: str, current_time: datetime) -> Tuple[bool, Dict[str, Any]]:
        """Check user-specific rate limits."""
        requests = self.user_requests[user_id]
        
        # Clean old requests
        self._clean_old_requests(requests, current_time)
        
        # Check limits
        minute_requests = sum(1 for req_time in requests if current_time - req_time <= timedelta(minutes=1))
        hour_requests = sum(1 for req_time in requests if current_time - req_time <= timedelta(hours=1))
        day_requests = sum(1 for req_time in requests if current_time - req_time <= timedelta(days=1))
        
        if minute_requests >= self.config.requests_per_minute:
            return False, {
                'reason': 'minute_limit_exceeded',
                'limit': self.config.requests_per_minute,
                'current': minute_requests,
                'retry_after': 60
            }
        
        if hour_requests >= self.config.requests_per_hour:
            return False, {
                'reason': 'hour_limit_exceeded',
                'limit': self.config.requests_per_hour,
                'current': hour_requests,
                'retry_after': 3600
            }
        
        if day_requests >= self.config.requests_per_day:
            return False, {
                'reason': 'day_limit_exceeded',
                'limit': self.config.requests_per_day,
                'current': day_requests,
                'retry_after': 86400
            }
        
        return True, {}
    
    def _check_ip_limits(self, ip_address: str, current_time: datetime) -> Tuple[bool, Dict[str, Any]]:
        """Check IP-specific rate limits."""
        requests = self.ip_requests[ip_address]
        
        # Clean old requests
        self._clean_old_requests(requests, current_time)
        
        # Check burst limit
        recent_requests = sum(1 for req_time in requests if current_time - req_time <= timedelta(seconds=10))
        
        if recent_requests >= self.config.burst_limit:
            return False, {
                'reason': 'burst_limit_exceeded',
                'limit': self.config.burst_limit,
                'current': recent_requests,
                'retry_after': 10
            }
        
        return True, {}
    
    def _clean_old_requests(self, requests: deque, current_time: datetime) -> None:
        """Clean old requests from deque."""
        cutoff_time = current_time - timedelta(days=1)
        
        while requests and requests[0] < cutoff_time:
            requests.popleft()
    
    def _record_request(self, user_id: str, ip_address: str, current_time: datetime) -> None:
        """Record a new request."""
        if self.config.per_user:
            self.user_requests[user_id].append(current_time)
        
        if self.config.per_ip and ip_address:
            self.ip_requests[ip_address].append(current_time)
    
    def _apply_penalty(self, identifier: str) -> None:
        """Apply penalty for rate limit violation."""
        if self.config.progressive_penalty:
            # Increase penalty for repeated violations
            current_penalty = self.penalties.get(identifier, datetime.now())
            if current_penalty > datetime.now():
                # Double the penalty
                penalty_duration = self.config.penalty_duration * 2
            else:
                penalty_duration = self.config.penalty_duration
        else:
            penalty_duration = self.config.penalty_duration
        
        self.penalties[identifier] = datetime.now() + penalty_duration
        self.logger.warning(f"Applied rate limit penalty to {identifier} for {penalty_duration}")
    
    def _get_remaining_requests(self, user_id: str) -> Dict[str, int]:
        """Get remaining requests for user."""
        if not self.config.per_user or user_id not in self.user_requests:
            return {
                'minute': self.config.requests_per_minute,
                'hour': self.config.requests_per_hour,
                'day': self.config.requests_per_day
            }
        
        requests = self.user_requests[user_id]
        current_time = datetime.now()
        
        minute_requests = sum(1 for req_time in requests if current_time - req_time <= timedelta(minutes=1))
        hour_requests = sum(1 for req_time in requests if current_time - req_time <= timedelta(hours=1))
        day_requests = sum(1 for req_time in requests if current_time - req_time <= timedelta(days=1))
        
        return {
            'minute': max(0, self.config.requests_per_minute - minute_requests),
            'hour': max(0, self.config.requests_per_hour - hour_requests),
            'day': max(0, self.config.requests_per_day - day_requests)
        }


class Authenticator:
    """Authentication handler."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage for API keys and tokens
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.revoked_tokens: Set[str] = set()
    
    def authenticate(self, request_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate request."""
        try:
            if self.config.method == AuthMethod.JWT:
                return self._authenticate_jwt(request_data)
            elif self.config.method == AuthMethod.API_KEY:
                return self._authenticate_api_key(request_data)
            elif self.config.method == AuthMethod.BEARER:
                return self._authenticate_bearer(request_data)
            else:
                return False, None
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False, None
    
    def _authenticate_jwt(self, request_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using JWT token."""
        # Extract token from Authorization header
        auth_header = request_data.get('headers', {}).get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return False, None
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        if token in self.revoked_tokens:
            return False, None
        
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience
            )
            
            # Check expiration
            if 'exp' in payload and datetime.fromtimestamp(payload['exp']) < datetime.now():
                return False, None
            
            return True, payload
            
        except jwt.InvalidTokenError:
            return False, None
    
    def _authenticate_api_key(self, request_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using API key."""
        # Check header
        api_key = request_data.get('headers', {}).get(self.config.api_key_header)
        
        # Check query parameter if not in header
        if not api_key:
            api_key = request_data.get('query_params', {}).get(self.config.api_key_query_param)
        
        if not api_key or api_key not in self.api_keys:
            return False, None
        
        key_info = self.api_keys[api_key]
        
        # Check if key is active
        if not key_info.get('active', True):
            return False, None
        
        # Check expiration
        if 'expires_at' in key_info and key_info['expires_at'] < datetime.now():
            return False, None
        
        return True, key_info
    
    def _authenticate_bearer(self, request_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Authenticate using Bearer token."""
        auth_header = request_data.get('headers', {}).get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return False, None
        
        token = auth_header[7:]
        
        # Custom bearer token validation logic
        # This would typically involve checking against a database or external service
        return self._validate_bearer_token(token)
    
    def _validate_bearer_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate bearer token (placeholder implementation)."""
        # Placeholder - implement actual token validation
        return False, None
    
    def generate_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token."""
        payload = {
            'user_id': user_data.get('user_id'),
            'username': user_data.get('username'),
            'roles': user_data.get('roles', []),
            'iat': datetime.now(),
            'exp': datetime.now() + self.config.token_expiry,
            'iss': self.config.issuer,
            'aud': self.config.audience
        }
        
        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        self.revoked_tokens.add(token)
        return True
    
    def add_api_key(self, key: str, user_data: Dict[str, Any], 
                   expires_at: Optional[datetime] = None) -> bool:
        """Add API key."""
        self.api_keys[key] = {
            'user_id': user_data.get('user_id'),
            'username': user_data.get('username'),
            'roles': user_data.get('roles', []),
            'created_at': datetime.now(),
            'expires_at': expires_at,
            'active': True
        }
        
        self.logger.info(f"Added API key for user: {user_data.get('username')}")
        return True
    
    def revoke_api_key(self, key: str) -> bool:
        """Revoke API key."""
        if key in self.api_keys:
            self.api_keys[key]['active'] = False
            self.logger.info(f"Revoked API key: {key[:8]}...")
            return True
        return False


class RequestValidator:
    """Request validation handler."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.schemas: Dict[str, BaseModel] = {}
    
    def register_schema(self, endpoint: str, schema: BaseModel) -> None:
        """Register validation schema for endpoint."""
        self.schemas[endpoint] = schema
        self.logger.info(f"Registered validation schema for {endpoint}")
    
    def validate_request(self, endpoint: str, data: Any) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Validate request data."""
        if endpoint not in self.schemas:
            return True, data, None  # No validation required
        
        schema = self.schemas[endpoint]
        
        try:
            if isinstance(data, dict):
                validated_data = schema(**data)
            else:
                validated_data = schema(data)
            
            return True, validated_data.dict(), None
            
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                error_details.append({
                    'field': '.'.join(str(x) for x in error['loc']),
                    'message': error['msg'],
                    'type': error['type']
                })
            
            return False, None, f"Validation failed: {error_details}"
        
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"


def require_auth(auth_config: AuthConfig):
    """Decorator for requiring authentication."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract request data (implementation depends on framework)
            request_data = kwargs.get('request_data', {})
            
            authenticator = Authenticator(auth_config)
            is_authenticated, user_data = authenticator.authenticate(request_data)
            
            if not is_authenticated:
                error = APIError(
                    code=ErrorCode.UNAUTHORIZED,
                    message="Authentication required"
                )
                return APIResponse.error_response(error)
            
            # Add user data to kwargs
            kwargs['user_data'] = user_data
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit(rate_config: RateLimitConfig):
    """Decorator for rate limiting."""
    limiter = RateLimiter(rate_config)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user and IP information
            user_data = kwargs.get('user_data', {})
            request_data = kwargs.get('request_data', {})
            
            user_id = user_data.get('user_id', 'anonymous')
            ip_address = request_data.get('ip_address', '')
            
            is_allowed, limit_info = limiter.is_allowed(user_id, ip_address)
            
            if not is_allowed:
                error = APIError(
                    code=ErrorCode.RATE_LIMIT_EXCEEDED,
                    message="Rate limit exceeded",
                    details=limit_info
                )
                return APIResponse.error_response(error)
            
            # Add rate limit info to response metadata
            response = func(*args, **kwargs)
            if isinstance(response, APIResponse):
                response.metadata.update({
                    'rate_limit': limit_info
                })
            
            return response
        
        return wrapper
    return decorator


def validate_request(schema: BaseModel):
    """Decorator for request validation."""
    validator = RequestValidator()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            request_data = kwargs.get('request_data', {})
            data = request_data.get('data', {})
            
            is_valid, validated_data, error_message = validator.validate_request(
                func.__name__, data
            )
            
            if not is_valid:
                error = APIError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=error_message or "Request validation failed"
                )
                return APIResponse.error_response(error)
            
            # Replace data with validated data
            kwargs['validated_data'] = validated_data
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def handle_api_errors(func: Callable) -> Callable:
    """Decorator for handling API errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        
        except ValidationError as e:
            error = APIError(
                code=ErrorCode.INVALID_REQUEST,
                message="Validation error",
                details={'validation_errors': e.errors()}
            )
            return APIResponse.error_response(error)
        
        except PermissionError:
            error = APIError(
                code=ErrorCode.FORBIDDEN,
                message="Insufficient permissions"
            )
            return APIResponse.error_response(error)
        
        except FileNotFoundError:
            error = APIError(
                code=ErrorCode.NOT_FOUND,
                message="Resource not found"
            )
            return APIResponse.error_response(error)
        
        except TimeoutError:
            error = APIError(
                code=ErrorCode.TIMEOUT,
                message="Request timeout"
            )
            return APIResponse.error_response(error)
        
        except Exception as e:
            logging.getLogger(__name__).error(f"Unhandled API error: {e}")
            error = APIError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Internal server error"
            )
            return APIResponse.error_response(error)
    
    return wrapper


class APIMiddleware:
    """Base API middleware class."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request."""
        return request_data
    
    def process_response(self, response: APIResponse) -> APIResponse:
        """Process outgoing response."""
        return response


class CORSMiddleware(APIMiddleware):
    """CORS middleware for handling cross-origin requests."""
    
    def __init__(self, allowed_origins: List[str] = None, 
                 allowed_methods: List[str] = None,
                 allowed_headers: List[str] = None):
        super().__init__("CORS")
        
        self.allowed_origins = allowed_origins or ['*']
        self.allowed_methods = allowed_methods or ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        self.allowed_headers = allowed_headers or ['Content-Type', 'Authorization']
    
    def process_response(self, response: APIResponse) -> APIResponse:
        """Add CORS headers to response."""
        cors_headers = {
            'Access-Control-Allow-Origin': ', '.join(self.allowed_origins),
            'Access-Control-Allow-Methods': ', '.join(self.allowed_methods),
            'Access-Control-Allow-Headers': ', '.join(self.allowed_headers),
            'Access-Control-Max-Age': '86400'
        }
        
        response.metadata.update({'cors_headers': cors_headers})
        return response


class LoggingMiddleware(APIMiddleware):
    """Logging middleware for request/response logging."""
    
    def __init__(self, log_requests: bool = True, log_responses: bool = True):
        super().__init__("Logging")
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log incoming request."""
        if self.log_requests:
            self.logger.info(f"API Request: {request_data.get('method')} {request_data.get('path')}")
        
        request_data['start_time'] = time.time()
        return request_data
    
    def process_response(self, response: APIResponse) -> APIResponse:
        """Log outgoing response."""
        if self.log_responses:
            self.logger.info(f"API Response: {response.success} - {response.request_id}")
        
        return response


# Global instances
_rate_limiter: Optional[RateLimiter] = None
_authenticator: Optional[Authenticator] = None
_validator: Optional[RequestValidator] = None


def initialize_api_common(auth_config: AuthConfig, rate_config: RateLimitConfig) -> None:
    """Initialize common API components."""
    global _rate_limiter, _authenticator, _validator
    
    _rate_limiter = RateLimiter(rate_config)
    _authenticator = Authenticator(auth_config)
    _validator = RequestValidator()
    
    logging.getLogger(__name__).info("Initialized API common components")


def get_rate_limiter() -> Optional[RateLimiter]:
    """Get global rate limiter instance."""
    return _rate_limiter


def get_authenticator() -> Optional[Authenticator]:
    """Get global authenticator instance."""
    return _authenticator


def get_validator() -> Optional[RequestValidator]:
    """Get global validator instance."""
    return _validator


def shutdown_api_common() -> None:
    """Shutdown common API components."""
    global _rate_limiter, _authenticator, _validator
    
    _rate_limiter = None
    _authenticator = None
    _validator = None
    
    logging.getLogger(__name__).info("Shutdown API common components")