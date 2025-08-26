"""API layer for AI Bull Ford.

This module provides comprehensive API capabilities including:
- REST API with OpenAPI/Swagger support
- WebSocket for real-time communication
- gRPC for high-performance services
- GraphQL for flexible queries
- Common utilities (auth, rate limiting, validation)
"""

# Common API components
from .common import (
    # Enums
    APIVersion,
    AuthMethod,
    HTTPMethod,
    ErrorCode,
    
    # Data classes
    APIError,
    APIResponse,
    RateLimitConfig,
    AuthConfig,
    
    # Core classes
    RateLimiter,
    Authenticator,
    RequestValidator,
    APIMiddleware,
    CORSMiddleware,
    LoggingMiddleware,
    
    # Decorators
    require_auth,
    rate_limit,
    validate_request,
    handle_api_errors,
    
    # Global functions
    initialize_api_common,
    get_rate_limiter,
    get_authenticator,
    get_validator,
    shutdown_api_common
)

# REST API components
from .rest import (
    # Enums
    ContentType,
    ResponseFormat,
    
    # Data classes
    EndpointConfig,
    RequestContext,
    ClientConfig as RESTClientConfig,
    
    # Core classes
    APIEndpoint,
    RequestHandler,
    RESTClient,
    
    # Global functions
    get_request_handler,
    get_rest_client,
    initialize_rest_api,
    shutdown_rest_api,
    async_shutdown_rest_api
)

# WebSocket components
from .websocket import (
    # Enums
    ConnectionState,
    MessageType,
    
    # Data classes
    WebSocketMessage,
    ConnectionInfo,
    WebSocketConfig,
    
    # Core classes
    MessageHandler,
    DefaultMessageHandler,
    ConnectionManager,
    WebSocketServer,
    WebSocketClient,
    
    # Global functions
    get_websocket_server,
    get_websocket_client,
    initialize_websocket_api,
    shutdown_websocket_api
)

# gRPC components
from .grpc import (
    # Enums
    ServiceType,
    CompressionType,
    
    # Data classes
    ServiceConfig,
    ServerConfig,
    ClientConfig,
    
    # Core classes
    GRPCService,
    AuthInterceptor,
    LoggingInterceptor,
    RateLimitInterceptor,
    GRPCServer,
    GRPCClient,
    ServiceRegistry,
    
    # Global functions
    get_grpc_server,
    get_grpc_client,
    get_service_registry,
    initialize_grpc_api,
    shutdown_grpc_api
)

# GraphQL components
from .graphql import (
    # Enums
    OperationType,
    DirectiveLocation,
    
    # Data classes
    QueryInfo,
    ResolverContext,
    SubscriptionInfo,
    GraphQLConfig,
    
    # Core classes
    GraphQLResolver,
    QueryComplexityAnalyzer,
    SubscriptionManager,
    AuthDirective,
    RateLimitDirective,
    GraphQLExecutor,
    SchemaBuilder,
    GraphQLServer,
    
    # Global functions
    get_graphql_server,
    initialize_graphql_api,
    shutdown_graphql_api
)

__all__ = [
    # Common API components
    "APIVersion",
    "AuthMethod",
    "HTTPMethod",
    "ErrorCode",
    "APIError",
    "APIResponse",
    "RateLimitConfig",
    "AuthConfig",
    "RateLimiter",
    "Authenticator",
    "RequestValidator",
    "APIMiddleware",
    "CORSMiddleware",
    "LoggingMiddleware",
    "require_auth",
    "rate_limit",
    "validate_request",
    "handle_api_errors",
    "initialize_api_common",
    "get_rate_limiter",
    "get_authenticator",
    "get_validator",
    "shutdown_api_common",
    
    # REST API components
    "ContentType",
    "ResponseFormat",
    "EndpointConfig",
    "RequestContext",
    "RESTClientConfig",
    "APIEndpoint",
    "RequestHandler",
    "RESTClient",
    "get_request_handler",
    "get_rest_client",
    "initialize_rest_api",
    "shutdown_rest_api",
    "async_shutdown_rest_api",
    
    # WebSocket components
    "ConnectionState",
    "MessageType",
    "WebSocketMessage",
    "ConnectionInfo",
    "WebSocketConfig",
    "MessageHandler",
    "DefaultMessageHandler",
    "ConnectionManager",
    "WebSocketServer",
    "WebSocketClient",
    "get_websocket_server",
    "get_websocket_client",
    "initialize_websocket_api",
    "shutdown_websocket_api",
    
    # gRPC components
    "ServiceType",
    "CompressionType",
    "ServiceConfig",
    "ServerConfig",
    "ClientConfig",
    "GRPCService",
    "AuthInterceptor",
    "LoggingInterceptor",
    "RateLimitInterceptor",
    "GRPCServer",
    "GRPCClient",
    "ServiceRegistry",
    "get_grpc_server",
    "get_grpc_client",
    "get_service_registry",
    "initialize_grpc_api",
    "shutdown_grpc_api",
    
    # GraphQL components
    "OperationType",
    "DirectiveLocation",
    "QueryInfo",
    "ResolverContext",
    "SubscriptionInfo",
    "GraphQLConfig",
    "GraphQLResolver",
    "QueryComplexityAnalyzer",
    "SubscriptionManager",
    "AuthDirective",
    "RateLimitDirective",
    "GraphQLExecutor",
    "SchemaBuilder",
    "GraphQLServer",
    "get_graphql_server",
    "initialize_graphql_api",
    "shutdown_graphql_api"
]