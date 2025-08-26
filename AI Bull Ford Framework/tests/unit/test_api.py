"""Unit tests for API module components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import json
from typing import Dict, Any, List

# Import API modules
from api.rest import (
    RESTConfig, APIEndpoint, RequestHandler, ResponseHandler,
    RESTServer, create_rest_app
)
from api.websocket import (
    WebSocketConfig, WebSocketHandler, WebSocketManager,
    ConnectionManager, MessageRouter, WebSocketServer
)
from api.grpc import (
    GRPCConfig, GRPCServicer, GRPCServer, ClientConfig,
    create_grpc_server, create_grpc_client
)
from api.graphql import (
    GraphQLConfig, GraphQLSchema, QueryResolver, MutationResolver,
    GraphQLServer, create_graphql_app
)
from api.common import (
    APIResponse, APIError, RateLimiter, AuthenticationMiddleware,
    CORSMiddleware, LoggingMiddleware, ValidationMiddleware
)


class TestRESTAPI:
    """Test cases for REST API components."""
    
    def test_rest_config(self):
        """Test RESTConfig creation and validation."""
        config = RESTConfig(
            host="localhost",
            port=8000,
            debug=True,
            cors_enabled=True,
            rate_limit_enabled=True,
            max_requests_per_minute=100
        )
        
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.debug is True
        assert config.cors_enabled is True
        assert config.rate_limit_enabled is True
        assert config.max_requests_per_minute == 100
    
    def test_api_endpoint(self):
        """Test APIEndpoint creation."""
        endpoint = APIEndpoint(
            path="/api/v1/models",
            method="GET",
            handler=lambda: {"models": []},
            description="Get all models",
            tags=["models"]
        )
        
        assert endpoint.path == "/api/v1/models"
        assert endpoint.method == "GET"
        assert callable(endpoint.handler)
        assert endpoint.description == "Get all models"
        assert endpoint.tags == ["models"]
    
    def test_request_handler(self):
        """Test RequestHandler functionality."""
        handler = RequestHandler()
        
        # Test request validation
        valid_data = {"name": "test", "value": 123}
        assert handler.validate_request(valid_data) is True
        
        # Test request preprocessing
        processed = handler.preprocess_request(valid_data)
        assert isinstance(processed, dict)
    
    def test_response_handler(self):
        """Test ResponseHandler functionality."""
        handler = ResponseHandler()
        
        # Test successful response
        data = {"result": "success"}
        response = handler.create_response(data, status_code=200)
        
        assert isinstance(response, APIResponse)
        assert response.status_code == 200
        assert response.data == data
        
        # Test error response
        error_response = handler.create_error_response(
            "Invalid input", status_code=400
        )
        
        assert isinstance(error_response, APIError)
        assert error_response.status_code == 400
        assert "Invalid input" in error_response.message
    
    def test_rest_server_creation(self):
        """Test REST server creation."""
        config = RESTConfig(host="localhost", port=8000)
        server = RESTServer(config)
        
        assert server.config == config
        assert server.app is not None
        
        # Test endpoint registration
        endpoint = APIEndpoint(
            path="/test",
            method="GET",
            handler=lambda: {"message": "test"}
        )
        
        server.register_endpoint(endpoint)
        assert len(server.endpoints) == 1
    
    def test_create_rest_app(self):
        """Test REST app creation function."""
        config = RESTConfig()
        app = create_rest_app(config)
        
        assert app is not None
        
        # Test with TestClient
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "status" in response.json()


class TestWebSocketAPI:
    """Test cases for WebSocket API components."""
    
    def test_websocket_config(self):
        """Test WebSocketConfig creation."""
        config = WebSocketConfig(
            host="localhost",
            port=8001,
            max_connections=100,
            heartbeat_interval=30,
            message_queue_size=1000
        )
        
        assert config.host == "localhost"
        assert config.port == 8001
        assert config.max_connections == 100
        assert config.heartbeat_interval == 30
        assert config.message_queue_size == 1000
    
    @pytest.mark.asyncio
    async def test_websocket_handler(self):
        """Test WebSocketHandler functionality."""
        handler = WebSocketHandler()
        
        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        mock_websocket.receive_text = AsyncMock(return_value='{"type": "ping"}')
        mock_websocket.send_text = AsyncMock()
        
        # Test message handling
        message = '{"type": "test", "data": "hello"}'
        result = await handler.handle_message(mock_websocket, message)
        
        assert result is not None
    
    def test_connection_manager(self):
        """Test ConnectionManager functionality."""
        manager = ConnectionManager()
        
        # Mock connections
        conn1 = Mock()
        conn2 = Mock()
        
        # Test connection management
        manager.add_connection("user1", conn1)
        manager.add_connection("user2", conn2)
        
        assert manager.get_connection("user1") == conn1
        assert manager.get_connection("user2") == conn2
        assert len(manager.get_all_connections()) == 2
        
        # Test connection removal
        manager.remove_connection("user1")
        assert manager.get_connection("user1") is None
        assert len(manager.get_all_connections()) == 1
    
    @pytest.mark.asyncio
    async def test_message_router(self):
        """Test MessageRouter functionality."""
        router = MessageRouter()
        
        # Register message handler
        @router.route("test_message")
        async def handle_test(data):
            return {"response": "handled", "data": data}
        
        # Test message routing
        message = {"type": "test_message", "payload": {"key": "value"}}
        result = await router.route_message(message)
        
        assert result["response"] == "handled"
        assert result["data"]["key"] == "value"
    
    def test_websocket_server(self):
        """Test WebSocketServer creation."""
        config = WebSocketConfig()
        server = WebSocketServer(config)
        
        assert server.config == config
        assert isinstance(server.connection_manager, ConnectionManager)
        assert isinstance(server.message_router, MessageRouter)


class TestGRPCAPI:
    """Test cases for gRPC API components."""
    
    def test_grpc_config(self):
        """Test GRPCConfig creation."""
        config = GRPCConfig(
            host="localhost",
            port=50051,
            max_workers=10,
            max_message_length=4 * 1024 * 1024,
            compression="gzip"
        )
        
        assert config.host == "localhost"
        assert config.port == 50051
        assert config.max_workers == 10
        assert config.max_message_length == 4 * 1024 * 1024
        assert config.compression == "gzip"
    
    def test_client_config(self):
        """Test ClientConfig creation."""
        config = ClientConfig(
            server_address="localhost:50051",
            timeout=30,
            retry_attempts=3,
            compression="gzip"
        )
        
        assert config.server_address == "localhost:50051"
        assert config.timeout == 30
        assert config.retry_attempts == 3
        assert config.compression == "gzip"
    
    def test_grpc_servicer(self):
        """Test GRPCServicer base functionality."""
        servicer = GRPCServicer()
        
        # Test service registration
        assert hasattr(servicer, 'register_service')
        assert callable(servicer.register_service)
        
        # Test method registration
        def test_method(request, context):
            return {"result": "success"}
        
        servicer.register_method("TestMethod", test_method)
        assert "TestMethod" in servicer.methods
    
    @patch('grpc.server')
    def test_create_grpc_server(self, mock_grpc_server):
        """Test gRPC server creation."""
        config = GRPCConfig()
        mock_server = Mock()
        mock_grpc_server.return_value = mock_server
        
        server = create_grpc_server(config)
        
        mock_grpc_server.assert_called_once()
        assert server == mock_server
    
    @patch('grpc.insecure_channel')
    def test_create_grpc_client(self, mock_channel):
        """Test gRPC client creation."""
        config = ClientConfig(server_address="localhost:50051")
        mock_channel_instance = Mock()
        mock_channel.return_value = mock_channel_instance
        
        client = create_grpc_client(config)
        
        mock_channel.assert_called_once_with("localhost:50051")
        assert client == mock_channel_instance


class TestGraphQLAPI:
    """Test cases for GraphQL API components."""
    
    def test_graphql_config(self):
        """Test GraphQLConfig creation."""
        config = GraphQLConfig(
            schema_path="schema.graphql",
            introspection_enabled=True,
            playground_enabled=True,
            max_query_depth=10,
            max_query_complexity=1000
        )
        
        assert config.schema_path == "schema.graphql"
        assert config.introspection_enabled is True
        assert config.playground_enabled is True
        assert config.max_query_depth == 10
        assert config.max_query_complexity == 1000
    
    def test_query_resolver(self):
        """Test QueryResolver functionality."""
        resolver = QueryResolver()
        
        # Register query resolver
        @resolver.field("hello")
        def resolve_hello(obj, info, name="World"):
            return f"Hello, {name}!"
        
        # Test resolver registration
        assert "hello" in resolver.resolvers
        assert callable(resolver.resolvers["hello"])
        
        # Test resolver execution
        result = resolver.resolvers["hello"](None, None, name="GraphQL")
        assert result == "Hello, GraphQL!"
    
    def test_mutation_resolver(self):
        """Test MutationResolver functionality."""
        resolver = MutationResolver()
        
        # Register mutation resolver
        @resolver.field("createUser")
        def resolve_create_user(obj, info, input):
            return {"id": "123", "name": input["name"]}
        
        # Test resolver registration
        assert "createUser" in resolver.resolvers
        assert callable(resolver.resolvers["createUser"])
        
        # Test resolver execution
        result = resolver.resolvers["createUser"](None, None, {"name": "John"})
        assert result["id"] == "123"
        assert result["name"] == "John"
    
    def test_graphql_schema(self):
        """Test GraphQLSchema creation."""
        schema = GraphQLSchema()
        
        # Test schema definition
        type_defs = """
            type Query {
                hello(name: String): String
            }
            
            type Mutation {
                createUser(input: UserInput!): User
            }
            
            input UserInput {
                name: String!
            }
            
            type User {
                id: ID!
                name: String!
            }
        """
        
        schema.set_type_definitions(type_defs)
        assert schema.type_definitions == type_defs
    
    def test_create_graphql_app(self):
        """Test GraphQL app creation."""
        config = GraphQLConfig()
        app = create_graphql_app(config)
        
        assert app is not None


class TestCommonComponents:
    """Test cases for common API components."""
    
    def test_api_response(self):
        """Test APIResponse creation."""
        data = {"message": "success"}
        response = APIResponse(
            data=data,
            status_code=200,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.data == data
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.success is True
    
    def test_api_error(self):
        """Test APIError creation."""
        error = APIError(
            message="Invalid request",
            status_code=400,
            error_code="INVALID_REQUEST",
            details={"field": "name is required"}
        )
        
        assert error.message == "Invalid request"
        assert error.status_code == 400
        assert error.error_code == "INVALID_REQUEST"
        assert error.details["field"] == "name is required"
        assert error.success is False
    
    def test_rate_limiter(self):
        """Test RateLimiter functionality."""
        limiter = RateLimiter(
            max_requests=5,
            time_window=60  # 5 requests per minute
        )
        
        client_id = "test_client"
        
        # Test within limit
        for i in range(5):
            assert limiter.is_allowed(client_id) is True
        
        # Test exceeding limit
        assert limiter.is_allowed(client_id) is False
        
        # Test reset
        limiter.reset_client(client_id)
        assert limiter.is_allowed(client_id) is True
    
    def test_authentication_middleware(self):
        """Test AuthenticationMiddleware."""
        middleware = AuthenticationMiddleware(
            secret_key="test_secret",
            algorithm="HS256"
        )
        
        # Test token generation
        payload = {"user_id": "123", "role": "user"}
        token = middleware.generate_token(payload)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Test token validation
        decoded = middleware.validate_token(token)
        assert decoded["user_id"] == "123"
        assert decoded["role"] == "user"
    
    def test_cors_middleware(self):
        """Test CORSMiddleware."""
        middleware = CORSMiddleware(
            allowed_origins=["http://localhost:3000"],
            allowed_methods=["GET", "POST"],
            allowed_headers=["Content-Type", "Authorization"]
        )
        
        # Test CORS headers
        headers = middleware.get_cors_headers("http://localhost:3000")
        
        assert "Access-Control-Allow-Origin" in headers
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert "Access-Control-Allow-Methods" in headers
        assert "Access-Control-Allow-Headers" in headers
    
    def test_logging_middleware(self):
        """Test LoggingMiddleware."""
        middleware = LoggingMiddleware(
            log_requests=True,
            log_responses=True,
            log_level="INFO"
        )
        
        # Test request logging
        request_data = {
            "method": "GET",
            "path": "/api/test",
            "headers": {"User-Agent": "test"}
        }
        
        log_entry = middleware.log_request(request_data)
        assert log_entry["method"] == "GET"
        assert log_entry["path"] == "/api/test"
        assert "timestamp" in log_entry
    
    def test_validation_middleware(self):
        """Test ValidationMiddleware."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        middleware = ValidationMiddleware(schema)
        
        # Test valid data
        valid_data = {"name": "John", "age": 30}
        assert middleware.validate(valid_data) is True
        
        # Test invalid data
        invalid_data = {"age": 30}  # missing required 'name'
        assert middleware.validate(invalid_data) is False
        
        # Test validation errors
        errors = middleware.get_validation_errors(invalid_data)
        assert len(errors) > 0
        assert any("name" in error for error in errors)


# Integration tests for API module
class TestAPIIntegration:
    """Integration tests for API module components."""
    
    def test_rest_websocket_integration(self):
        """Test REST and WebSocket integration."""
        # Create REST config
        rest_config = RESTConfig(port=8000)
        rest_server = RESTServer(rest_config)
        
        # Create WebSocket config
        ws_config = WebSocketConfig(port=8001)
        ws_server = WebSocketServer(ws_config)
        
        # Test that both servers can coexist
        assert rest_server.config.port != ws_server.config.port
        assert rest_server.app is not None
        assert ws_server.connection_manager is not None
    
    @patch('grpc.server')
    def test_grpc_rest_integration(self, mock_grpc_server):
        """Test gRPC and REST integration."""
        # Create REST server
        rest_config = RESTConfig(port=8000)
        rest_server = RESTServer(rest_config)
        
        # Create gRPC server
        grpc_config = GRPCConfig(port=50051)
        mock_server = Mock()
        mock_grpc_server.return_value = mock_server
        grpc_server = create_grpc_server(grpc_config)
        
        # Test that both servers use different ports
        assert rest_config.port != grpc_config.port
        assert rest_server.app is not None
        assert grpc_server == mock_server
    
    def test_middleware_chain(self):
        """Test middleware chain integration."""
        # Create middleware components
        auth_middleware = AuthenticationMiddleware("secret")
        cors_middleware = CORSMiddleware(["*"])
        logging_middleware = LoggingMiddleware()
        rate_limiter = RateLimiter(100, 60)
        
        # Test middleware chain
        middlewares = [
            auth_middleware,
            cors_middleware,
            logging_middleware,
            rate_limiter
        ]
        
        assert len(middlewares) == 4
        assert all(hasattr(mw, '__class__') for mw in middlewares)
    
    def test_api_response_serialization(self):
        """Test API response serialization across different formats."""
        data = {"users": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]}
        
        # Test JSON serialization
        response = APIResponse(data=data, status_code=200)
        json_data = response.to_json()
        
        assert isinstance(json_data, str)
        parsed = json.loads(json_data)
        assert parsed["data"] == data
        assert parsed["status_code"] == 200
        
        # Test dictionary conversion
        dict_data = response.to_dict()
        assert isinstance(dict_data, dict)
        assert dict_data["data"] == data
        assert dict_data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__])