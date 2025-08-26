"""REST API module for AI Bull Ford.

This module provides comprehensive REST API capabilities including:
- RESTful endpoints with automatic routing
- HTTP client for external API calls
- Request/response handling with validation
- OpenAPI documentation generation
- Async/sync support
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Type
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from pydantic import BaseModel, ValidationError

from .common import (
    APIError, APIResponse, HTTPMethod, ErrorCode,
    require_auth, rate_limit, validate_request, handle_api_errors
)


class ContentType(Enum):
    """Supported content types."""
    JSON = "application/json"
    XML = "application/xml"
    FORM = "application/x-www-form-urlencoded"
    MULTIPART = "multipart/form-data"
    TEXT = "text/plain"
    HTML = "text/html"


class ResponseFormat(Enum):
    """Response format options."""
    JSON = "json"
    XML = "xml"
    TEXT = "text"
    BINARY = "binary"


@dataclass
class EndpointConfig:
    """Configuration for API endpoints."""
    path: str
    method: HTTPMethod
    handler: Callable
    auth_required: bool = False
    rate_limit_per_minute: Optional[int] = None
    request_schema: Optional[Type[BaseModel]] = None
    response_schema: Optional[Type[BaseModel]] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False


@dataclass
class RequestContext:
    """Context information for API requests."""
    method: HTTPMethod
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, Any]
    body: Optional[Any] = None
    user_id: Optional[str] = None
    client_ip: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: f"req_{datetime.now().timestamp()}")


@dataclass
class ClientConfig:
    """Configuration for REST client."""
    base_url: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)
    auth_token: Optional[str] = None
    verify_ssl: bool = True
    follow_redirects: bool = True


class APIEndpoint:
    """Represents a REST API endpoint."""
    
    def __init__(self, config: EndpointConfig):
        self.config = config
        self._middleware: List[Callable] = []
        
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to the endpoint."""
        self._middleware.append(middleware)
        
    def apply_decorators(self, handler: Callable) -> Callable:
        """Apply decorators based on endpoint configuration."""
        decorated_handler = handler
        
        # Apply validation decorator
        if self.config.request_schema:
            decorated_handler = validate_request(self.config.request_schema)(decorated_handler)
            
        # Apply rate limiting
        if self.config.rate_limit_per_minute:
            decorated_handler = rate_limit(self.config.rate_limit_per_minute)(decorated_handler)
            
        # Apply authentication
        if self.config.auth_required:
            decorated_handler = require_auth(decorated_handler)
            
        # Apply error handling
        decorated_handler = handle_api_errors(decorated_handler)
        
        return decorated_handler
        
    async def handle_request(self, context: RequestContext) -> APIResponse:
        """Handle incoming request."""
        try:
            # Apply middleware
            for middleware in self._middleware:
                context = await middleware(context)
                
            # Get decorated handler
            handler = self.apply_decorators(self.config.handler)
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(context)
            else:
                result = handler(context)
                
            # Format response
            if isinstance(result, APIResponse):
                return result
            else:
                return APIResponse(
                    success=True,
                    data=result,
                    message="Request processed successfully"
                )
                
        except ValidationError as e:
            return APIResponse(
                success=False,
                error_code=ErrorCode.VALIDATION_ERROR,
                message="Validation failed",
                details=e.errors()
            )
        except Exception as e:
            return APIResponse(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                message=str(e)
            )


class RequestHandler:
    """Handles HTTP request processing."""
    
    def __init__(self):
        self.endpoints: Dict[str, Dict[HTTPMethod, APIEndpoint]] = {}
        self._global_middleware: List[Callable] = []
        
    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """Register an API endpoint."""
        path = endpoint.config.path
        method = endpoint.config.method
        
        if path not in self.endpoints:
            self.endpoints[path] = {}
            
        self.endpoints[path][method] = endpoint
        
    def add_global_middleware(self, middleware: Callable) -> None:
        """Add global middleware."""
        self._global_middleware.append(middleware)
        
    def route(self, path: str, method: HTTPMethod = HTTPMethod.GET, **kwargs):
        """Decorator for registering endpoints."""
        def decorator(handler: Callable):
            config = EndpointConfig(
                path=path,
                method=method,
                handler=handler,
                **kwargs
            )
            endpoint = APIEndpoint(config)
            self.register_endpoint(endpoint)
            return handler
        return decorator
        
    async def handle_request(self, method: str, path: str, **kwargs) -> APIResponse:
        """Handle incoming HTTP request."""
        try:
            http_method = HTTPMethod(method.upper())
        except ValueError:
            return APIResponse(
                success=False,
                error_code=ErrorCode.METHOD_NOT_ALLOWED,
                message=f"Method {method} not allowed"
            )
            
        # Find endpoint
        if path not in self.endpoints or http_method not in self.endpoints[path]:
            return APIResponse(
                success=False,
                error_code=ErrorCode.NOT_FOUND,
                message=f"Endpoint {method} {path} not found"
            )
            
        endpoint = self.endpoints[path][http_method]
        
        # Create request context
        context = RequestContext(
            method=http_method,
            path=path,
            **kwargs
        )
        
        # Apply global middleware
        for middleware in self._global_middleware:
            context = await middleware(context)
            
        # Handle request
        return await endpoint.handle_request(context)
        
    def get_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "AI Bull Ford API",
                "version": "1.0.0",
                "description": "REST API for AI Bull Ford system"
            },
            "paths": {}
        }
        
        for path, methods in self.endpoints.items():
            spec["paths"][path] = {}
            for method, endpoint in methods.items():
                spec["paths"][path][method.value.lower()] = {
                    "summary": endpoint.config.description,
                    "tags": endpoint.config.tags,
                    "deprecated": endpoint.config.deprecated
                }
                
        return spec


class RESTClient:
    """HTTP client for making REST API calls."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None
        
    def _get_session(self) -> requests.Session:
        """Get or create synchronous session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.config.headers)
            if self.config.auth_token:
                self._session.headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return self._session
        
    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create asynchronous session."""
        if self._async_session is None:
            headers = self.config.headers.copy()
            if self.config.auth_token:
                headers["Authorization"] = f"Bearer {self.config.auth_token}"
                
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._async_session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(verify_ssl=self.config.verify_ssl)
            )
        return self._async_session
        
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return urljoin(self.config.base_url, endpoint)
        
    def _handle_response(self, response: requests.Response, format_type: ResponseFormat) -> Any:
        """Handle response based on format type."""
        response.raise_for_status()
        
        if format_type == ResponseFormat.JSON:
            return response.json()
        elif format_type == ResponseFormat.TEXT:
            return response.text
        elif format_type == ResponseFormat.BINARY:
            return response.content
        else:
            return response.text
            
    async def _handle_async_response(self, response: aiohttp.ClientResponse, format_type: ResponseFormat) -> Any:
        """Handle async response based on format type."""
        response.raise_for_status()
        
        if format_type == ResponseFormat.JSON:
            return await response.json()
        elif format_type == ResponseFormat.TEXT:
            return await response.text()
        elif format_type == ResponseFormat.BINARY:
            return await response.read()
        else:
            return await response.text()
            
    def request(
        self,
        method: HTTPMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        format_type: ResponseFormat = ResponseFormat.JSON
    ) -> Any:
        """Make synchronous HTTP request."""
        session = self._get_session()
        url = self._build_url(endpoint)
        
        request_headers = headers or {}
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = session.request(
                    method=method.value,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl,
                    allow_redirects=self.config.follow_redirects
                )
                
                return self._handle_response(response, format_type)
                
            except requests.RequestException as e:
                if attempt == self.config.max_retries:
                    raise APIError(
                        error_code=ErrorCode.EXTERNAL_API_ERROR,
                        message=f"Request failed after {self.config.max_retries} retries: {str(e)}"
                    )
                    
                # Wait before retry
                import time
                time.sleep(self.config.retry_delay * (attempt + 1))
                
    async def async_request(
        self,
        method: HTTPMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        format_type: ResponseFormat = ResponseFormat.JSON
    ) -> Any:
        """Make asynchronous HTTP request."""
        session = await self._get_async_session()
        url = self._build_url(endpoint)
        
        request_headers = headers or {}
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with session.request(
                    method=method.value,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    allow_redirects=self.config.follow_redirects
                ) as response:
                    return await self._handle_async_response(response, format_type)
                    
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries:
                    raise APIError(
                        error_code=ErrorCode.EXTERNAL_API_ERROR,
                        message=f"Request failed after {self.config.max_retries} retries: {str(e)}"
                    )
                    
                # Wait before retry
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                
    def get(self, endpoint: str, **kwargs) -> Any:
        """Make GET request."""
        return self.request(HTTPMethod.GET, endpoint, **kwargs)
        
    def post(self, endpoint: str, **kwargs) -> Any:
        """Make POST request."""
        return self.request(HTTPMethod.POST, endpoint, **kwargs)
        
    def put(self, endpoint: str, **kwargs) -> Any:
        """Make PUT request."""
        return self.request(HTTPMethod.PUT, endpoint, **kwargs)
        
    def delete(self, endpoint: str, **kwargs) -> Any:
        """Make DELETE request."""
        return self.request(HTTPMethod.DELETE, endpoint, **kwargs)
        
    async def async_get(self, endpoint: str, **kwargs) -> Any:
        """Make async GET request."""
        return await self.async_request(HTTPMethod.GET, endpoint, **kwargs)
        
    async def async_post(self, endpoint: str, **kwargs) -> Any:
        """Make async POST request."""
        return await self.async_request(HTTPMethod.POST, endpoint, **kwargs)
        
    async def async_put(self, endpoint: str, **kwargs) -> Any:
        """Make async PUT request."""
        return await self.async_request(HTTPMethod.PUT, endpoint, **kwargs)
        
    async def async_delete(self, endpoint: str, **kwargs) -> Any:
        """Make async DELETE request."""
        return await self.async_request(HTTPMethod.DELETE, endpoint, **kwargs)
        
    def close(self) -> None:
        """Close synchronous session."""
        if self._session:
            self._session.close()
            self._session = None
            
    async def async_close(self) -> None:
        """Close asynchronous session."""
        if self._async_session:
            await self._async_session.close()
            self._async_session = None


# Global instances
_request_handler: Optional[RequestHandler] = None
_rest_client: Optional[RESTClient] = None


def get_request_handler() -> RequestHandler:
    """Get global request handler instance."""
    global _request_handler
    if _request_handler is None:
        _request_handler = RequestHandler()
    return _request_handler


def get_rest_client(config: Optional[ClientConfig] = None) -> RESTClient:
    """Get global REST client instance."""
    global _rest_client
    if _rest_client is None:
        if config is None:
            config = ClientConfig(base_url="http://localhost:8000")
        _rest_client = RESTClient(config)
    return _rest_client


def initialize_rest_api(client_config: Optional[ClientConfig] = None) -> None:
    """Initialize REST API components."""
    global _request_handler, _rest_client
    
    _request_handler = RequestHandler()
    
    if client_config is None:
        client_config = ClientConfig(base_url="http://localhost:8000")
    _rest_client = RESTClient(client_config)


def shutdown_rest_api() -> None:
    """Shutdown REST API components."""
    global _request_handler, _rest_client
    
    if _rest_client:
        _rest_client.close()
        _rest_client = None
        
    _request_handler = None


async def async_shutdown_rest_api() -> None:
    """Async shutdown REST API components."""
    global _request_handler, _rest_client
    
    if _rest_client:
        await _rest_client.async_close()
        _rest_client = None
        
    _request_handler = None