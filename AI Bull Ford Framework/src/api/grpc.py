"""gRPC API module for AI Bull Ford.

This module provides comprehensive gRPC capabilities including:
- High-performance RPC services
- Protocol buffer message handling
- Streaming support (client, server, bidirectional)
- Service discovery and load balancing
- Authentication and interceptors
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Iterator, Union

import grpc
from grpc import aio as aio_grpc
from grpc_reflection.v1alpha import reflection
from google.protobuf import descriptor_pool, message_factory
from google.protobuf.message import Message

from .common import APIError, ErrorCode, AuthConfig


class ServiceType(Enum):
    """gRPC service types."""
    UNARY = "unary"
    CLIENT_STREAMING = "client_streaming"
    SERVER_STREAMING = "server_streaming"
    BIDIRECTIONAL_STREAMING = "bidirectional_streaming"


class CompressionType(Enum):
    """gRPC compression types."""
    NONE = grpc.Compression.NoCompression
    GZIP = grpc.Compression.Gzip
    DEFLATE = grpc.Compression.Deflate


@dataclass
class ServiceConfig:
    """Configuration for gRPC service."""
    name: str
    version: str
    description: str = ""
    methods: Dict[str, ServiceType] = field(default_factory=dict)
    auth_required: bool = False
    rate_limit: Optional[int] = None
    timeout: float = 30.0
    compression: CompressionType = CompressionType.NONE
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ServerConfig:
    """Configuration for gRPC server."""
    host: str = "localhost"
    port: int = 50051
    max_workers: int = 10
    max_receive_message_length: int = 4 * 1024 * 1024  # 4MB
    max_send_message_length: int = 4 * 1024 * 1024  # 4MB
    compression: CompressionType = CompressionType.GZIP
    enable_reflection: bool = True
    auth_config: Optional[AuthConfig] = None
    ssl_credentials: Optional[grpc.ServerCredentials] = None
    interceptors: List[grpc.ServerInterceptor] = field(default_factory=list)


@dataclass
class ClientConfig:
    """Configuration for gRPC client."""
    target: str
    timeout: float = 30.0
    max_receive_message_length: int = 4 * 1024 * 1024  # 4MB
    max_send_message_length: int = 4 * 1024 * 1024  # 4MB
    compression: CompressionType = CompressionType.GZIP
    auth_token: Optional[str] = None
    ssl_credentials: Optional[grpc.ChannelCredentials] = None
    interceptors: List[grpc.ClientInterceptor] = field(default_factory=list)
    retry_policy: Optional[Dict[str, Any]] = None


class GRPCService(ABC):
    """Abstract base class for gRPC services."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self._methods: Dict[str, Callable] = {}
        
    @abstractmethod
    def register_methods(self) -> None:
        """Register service methods."""
        pass
    
    def add_method(self, name: str, handler: Callable, service_type: ServiceType) -> None:
        """Add method to service."""
        self._methods[name] = handler
        self.config.methods[name] = service_type
    
    def get_method(self, name: str) -> Optional[Callable]:
        """Get method handler."""
        return self._methods.get(name)
    
    def get_service_descriptor(self) -> Dict[str, Any]:
        """Get service descriptor."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "description": self.config.description,
            "methods": {name: stype.value for name, stype in self.config.methods.items()},
            "metadata": self.config.metadata
        }


class AuthInterceptor(grpc.ServerInterceptor):
    """Authentication interceptor for gRPC."""
    
    def __init__(self, auth_config: AuthConfig):
        self.auth_config = auth_config
    
    def intercept_service(self, continuation, handler_call_details):
        """Intercept service calls for authentication."""
        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata)
        
        # Check for authorization header
        auth_header = metadata.get('authorization')
        if not auth_header:
            return self._unauthenticated_response()
        
        # Validate token
        try:
            token = auth_header.replace('Bearer ', '')
            # TODO: Implement actual token validation
            if not self._validate_token(token):
                return self._unauthenticated_response()
        except Exception:
            return self._unauthenticated_response()
        
        return continuation(handler_call_details)
    
    def _validate_token(self, token: str) -> bool:
        """Validate authentication token."""
        # Basic validation - implement actual logic
        return len(token) > 0
    
    def _unauthenticated_response(self):
        """Return unauthenticated response."""
        def abort(ignored_request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Authentication required')
        return grpc.unary_unary_rpc_method_handler(abort)


class LoggingInterceptor(grpc.ServerInterceptor):
    """Logging interceptor for gRPC."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def intercept_service(self, continuation, handler_call_details):
        """Intercept service calls for logging."""
        start_time = datetime.now()
        
        def log_wrapper(behavior, request_streaming, response_streaming):
            def new_behavior(request_or_iterator, context):
                try:
                    response = behavior(request_or_iterator, context)
                    duration = (datetime.now() - start_time).total_seconds()
                    
                    self.logger.info(
                        f"gRPC call: {handler_call_details.method} "
                        f"duration: {duration:.3f}s status: OK"
                    )
                    return response
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.error(
                        f"gRPC call: {handler_call_details.method} "
                        f"duration: {duration:.3f}s error: {str(e)}"
                    )
                    raise
            return new_behavior
        
        handler = continuation(handler_call_details)
        if handler:
            return grpc.ServerInterceptor.intercept_service(
                self, lambda hcd: handler, handler_call_details
            )
        return handler


class RateLimitInterceptor(grpc.ServerInterceptor):
    """Rate limiting interceptor for gRPC."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, List[datetime]] = {}
    
    def intercept_service(self, continuation, handler_call_details):
        """Intercept service calls for rate limiting."""
        # Extract client IP from metadata
        metadata = dict(handler_call_details.invocation_metadata)
        client_ip = metadata.get('x-forwarded-for', 'unknown')
        
        # Check rate limit
        if not self._check_rate_limit(client_ip):
            def rate_limited(ignored_request, context):
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, 'Rate limit exceeded')
            return grpc.unary_unary_rpc_method_handler(rate_limited)
        
        return continuation(handler_call_details)
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limit."""
        now = datetime.now()
        minute_ago = now.timestamp() - 60
        
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        # Remove old requests
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if req_time.timestamp() > minute_ago
        ]
        
        # Check limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.request_counts[client_ip].append(now)
        return True


class GRPCServer:
    """gRPC server implementation."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.services: Dict[str, GRPCService] = {}
        self._server: Optional[aio_grpc.Server] = None
        
    def add_service(self, service: GRPCService) -> None:
        """Add service to server."""
        self.services[service.config.name] = service
        service.register_methods()
    
    def _create_interceptors(self) -> List[grpc.ServerInterceptor]:
        """Create server interceptors."""
        interceptors = [LoggingInterceptor()]
        
        if self.config.auth_config:
            interceptors.append(AuthInterceptor(self.config.auth_config))
        
        # Add custom interceptors
        interceptors.extend(self.config.interceptors)
        
        return interceptors
    
    async def start(self) -> None:
        """Start gRPC server."""
        # Create server
        interceptors = self._create_interceptors()
        self._server = aio_grpc.server(
            interceptors=interceptors,
            options=[
                ('grpc.max_receive_message_length', self.config.max_receive_message_length),
                ('grpc.max_send_message_length', self.config.max_send_message_length),
                ('grpc.default_compression_algorithm', self.config.compression.value),
            ]
        )
        
        # Add services
        for service in self.services.values():
            # TODO: Add actual service registration with protobuf
            pass
        
        # Enable reflection if configured
        if self.config.enable_reflection:
            service_names = [service.config.name for service in self.services.values()]
            reflection.enable_server_reflection(service_names, self._server)
        
        # Add port
        listen_addr = f'{self.config.host}:{self.config.port}'
        if self.config.ssl_credentials:
            self._server.add_secure_port(listen_addr, self.config.ssl_credentials)
        else:
            self._server.add_insecure_port(listen_addr)
        
        # Start server
        await self._server.start()
        print(f"gRPC server started on {listen_addr}")
    
    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop gRPC server."""
        if self._server:
            await self._server.stop(grace_period)
            self._server = None
            print("gRPC server stopped")
    
    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self._server:
            await self._server.wait_for_termination()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "server_config": {
                "host": self.config.host,
                "port": self.config.port,
                "max_workers": self.config.max_workers,
                "compression": self.config.compression.name
            },
            "services": {
                name: service.get_service_descriptor()
                for name, service in self.services.items()
            }
        }


class GRPCClient:
    """gRPC client implementation."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self._channel: Optional[aio_grpc.Channel] = None
        self._stubs: Dict[str, Any] = {}
    
    async def connect(self) -> None:
        """Connect to gRPC server."""
        try:
            # Create channel options
            options = [
                ('grpc.max_receive_message_length', self.config.max_receive_message_length),
                ('grpc.max_send_message_length', self.config.max_send_message_length),
                ('grpc.default_compression_algorithm', self.config.compression.value),
            ]
            
            # Add retry policy if configured
            if self.config.retry_policy:
                options.append(('grpc.service_config', self.config.retry_policy))
            
            # Create channel
            if self.config.ssl_credentials:
                self._channel = aio_grpc.secure_channel(
                    self.config.target,
                    self.config.ssl_credentials,
                    options=options
                )
            else:
                self._channel = aio_grpc.insecure_channel(
                    self.config.target,
                    options=options
                )
            
            # Wait for channel to be ready
            await self._channel.channel_ready()
            
        except Exception as e:
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Failed to connect to gRPC server: {str(e)}"
            )
    
    async def disconnect(self) -> None:
        """Disconnect from gRPC server."""
        if self._channel:
            await self._channel.close()
            self._channel = None
        self._stubs.clear()
    
    def create_stub(self, stub_class: type, service_name: str) -> Any:
        """Create service stub."""
        if not self._channel:
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message="Not connected to gRPC server"
            )
        
        if service_name not in self._stubs:
            self._stubs[service_name] = stub_class(self._channel)
        
        return self._stubs[service_name]
    
    def _get_metadata(self) -> List[tuple]:
        """Get request metadata."""
        metadata = []
        
        if self.config.auth_token:
            metadata.append(('authorization', f'Bearer {self.config.auth_token}'))
        
        return metadata
    
    async def unary_call(
        self,
        stub: Any,
        method_name: str,
        request: Message,
        timeout: Optional[float] = None
    ) -> Message:
        """Make unary RPC call."""
        try:
            method = getattr(stub, method_name)
            metadata = self._get_metadata()
            
            response = await method(
                request,
                metadata=metadata,
                timeout=timeout or self.config.timeout
            )
            
            return response
            
        except grpc.RpcError as e:
            raise APIError(
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                message=f"gRPC call failed: {e.details()}"
            )
    
    async def server_streaming_call(
        self,
        stub: Any,
        method_name: str,
        request: Message,
        timeout: Optional[float] = None
    ) -> AsyncIterator[Message]:
        """Make server streaming RPC call."""
        try:
            method = getattr(stub, method_name)
            metadata = self._get_metadata()
            
            async for response in method(
                request,
                metadata=metadata,
                timeout=timeout or self.config.timeout
            ):
                yield response
                
        except grpc.RpcError as e:
            raise APIError(
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                message=f"gRPC streaming call failed: {e.details()}"
            )
    
    async def client_streaming_call(
        self,
        stub: Any,
        method_name: str,
        request_iterator: AsyncIterator[Message],
        timeout: Optional[float] = None
    ) -> Message:
        """Make client streaming RPC call."""
        try:
            method = getattr(stub, method_name)
            metadata = self._get_metadata()
            
            response = await method(
                request_iterator,
                metadata=metadata,
                timeout=timeout or self.config.timeout
            )
            
            return response
            
        except grpc.RpcError as e:
            raise APIError(
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                message=f"gRPC client streaming call failed: {e.details()}"
            )
    
    async def bidirectional_streaming_call(
        self,
        stub: Any,
        method_name: str,
        request_iterator: AsyncIterator[Message],
        timeout: Optional[float] = None
    ) -> AsyncIterator[Message]:
        """Make bidirectional streaming RPC call."""
        try:
            method = getattr(stub, method_name)
            metadata = self._get_metadata()
            
            async for response in method(
                request_iterator,
                metadata=metadata,
                timeout=timeout or self.config.timeout
            ):
                yield response
                
        except grpc.RpcError as e:
            raise APIError(
                error_code=ErrorCode.EXTERNAL_API_ERROR,
                message=f"gRPC bidirectional streaming call failed: {e.details()}"
            )


class ServiceRegistry:
    """Registry for gRPC services."""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
    
    def register_service(
        self,
        name: str,
        version: str,
        endpoint: str,
        methods: Dict[str, ServiceType],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a service."""
        self.services[name] = {
            "version": version,
            "endpoint": endpoint,
            "methods": methods,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat()
        }
    
    def unregister_service(self, name: str) -> None:
        """Unregister a service."""
        if name in self.services:
            del self.services[name]
    
    def get_service(self, name: str) -> Optional[Dict[str, Any]]:
        """Get service information."""
        return self.services.get(name)
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all registered services."""
        return self.services.copy()
    
    def find_services_by_method(self, method_name: str) -> List[str]:
        """Find services that provide a specific method."""
        matching_services = []
        for service_name, service_info in self.services.items():
            if method_name in service_info.get("methods", {}):
                matching_services.append(service_name)
        return matching_services


# Global instances
_grpc_server: Optional[GRPCServer] = None
_grpc_client: Optional[GRPCClient] = None
_service_registry: Optional[ServiceRegistry] = None


def get_grpc_server(config: Optional[ServerConfig] = None) -> GRPCServer:
    """Get global gRPC server instance."""
    global _grpc_server
    if _grpc_server is None:
        if config is None:
            config = ServerConfig()
        _grpc_server = GRPCServer(config)
    return _grpc_server


def get_grpc_client(config: Optional[ClientConfig] = None) -> GRPCClient:
    """Get global gRPC client instance."""
    global _grpc_client
    if _grpc_client is None:
        if config is None:
            config = ClientConfig(target="localhost:50051")
        _grpc_client = GRPCClient(config)
    return _grpc_client


def get_service_registry() -> ServiceRegistry:
    """Get global service registry instance."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry


def initialize_grpc_api(
    server_config: Optional[ServerConfig] = None,
    client_config: Optional[ClientConfig] = None
) -> None:
    """Initialize gRPC API components."""
    global _grpc_server, _grpc_client, _service_registry
    
    if server_config is None:
        server_config = ServerConfig()
    _grpc_server = GRPCServer(server_config)
    
    if client_config is None:
        client_config = ClientConfig(target="localhost:50051")
    _grpc_client = GRPCClient(client_config)
    
    _service_registry = ServiceRegistry()


async def shutdown_grpc_api() -> None:
    """Shutdown gRPC API components."""
    global _grpc_server, _grpc_client, _service_registry
    
    if _grpc_server:
        await _grpc_server.stop()
        _grpc_server = None
    
    if _grpc_client:
        await _grpc_client.disconnect()
        _grpc_client = None
    
    _service_registry = None