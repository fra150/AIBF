"""GraphQL API module for AI Bull Ford.

This module provides comprehensive GraphQL capabilities including:
- Schema definition and type system
- Query, mutation, and subscription resolvers
- Real-time subscriptions with WebSocket support
- Authentication and authorization
- Query complexity analysis and rate limiting
- Introspection and playground
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, AsyncIterator

import graphql
from graphql import (
    GraphQLSchema, GraphQLObjectType, GraphQLField, GraphQLString,
    GraphQLInt, GraphQLFloat, GraphQLBoolean, GraphQLList, GraphQLNonNull,
    GraphQLArgument, GraphQLInputObjectType, GraphQLEnumType,
    build_schema, execute, subscribe, validate, parse
)
from graphql.execution import ExecutionResult
from graphql.error import GraphQLError

from .common import APIError, ErrorCode, AuthConfig


class OperationType(Enum):
    """GraphQL operation types."""
    QUERY = "query"
    MUTATION = "mutation"
    SUBSCRIPTION = "subscription"


class DirectiveLocation(Enum):
    """GraphQL directive locations."""
    QUERY = "QUERY"
    MUTATION = "MUTATION"
    SUBSCRIPTION = "SUBSCRIPTION"
    FIELD = "FIELD"
    FRAGMENT_DEFINITION = "FRAGMENT_DEFINITION"
    FRAGMENT_SPREAD = "FRAGMENT_SPREAD"
    INLINE_FRAGMENT = "INLINE_FRAGMENT"
    VARIABLE_DEFINITION = "VARIABLE_DEFINITION"
    SCHEMA = "SCHEMA"
    SCALAR = "SCALAR"
    OBJECT = "OBJECT"
    FIELD_DEFINITION = "FIELD_DEFINITION"
    ARGUMENT_DEFINITION = "ARGUMENT_DEFINITION"
    INTERFACE = "INTERFACE"
    UNION = "UNION"
    ENUM = "ENUM"
    ENUM_VALUE = "ENUM_VALUE"
    INPUT_OBJECT = "INPUT_OBJECT"
    INPUT_FIELD_DEFINITION = "INPUT_FIELD_DEFINITION"


@dataclass
class QueryInfo:
    """Information about a GraphQL query."""
    operation_name: Optional[str]
    operation_type: OperationType
    query: str
    variables: Dict[str, Any] = field(default_factory=dict)
    complexity: int = 0
    depth: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    ip_address: Optional[str] = None


@dataclass
class ResolverContext:
    """Context passed to GraphQL resolvers."""
    user_id: Optional[str] = None
    user_roles: List[str] = field(default_factory=list)
    request_id: str = ""
    ip_address: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    auth_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubscriptionInfo:
    """Information about a GraphQL subscription."""
    id: str
    query: str
    variables: Dict[str, Any]
    context: ResolverContext
    created_at: datetime = field(default_factory=datetime.now)
    last_event: Optional[datetime] = None
    event_count: int = 0


@dataclass
class GraphQLConfig:
    """Configuration for GraphQL API."""
    schema_path: Optional[str] = None
    enable_introspection: bool = True
    enable_playground: bool = True
    max_query_complexity: int = 1000
    max_query_depth: int = 15
    timeout: float = 30.0
    auth_config: Optional[AuthConfig] = None
    rate_limit_per_minute: int = 100
    enable_subscriptions: bool = True
    subscription_keepalive: float = 30.0
    custom_scalars: Dict[str, Any] = field(default_factory=dict)
    middleware: List[Callable] = field(default_factory=list)


class GraphQLResolver(ABC):
    """Abstract base class for GraphQL resolvers."""
    
    @abstractmethod
    async def resolve(self, parent: Any, info: Any, **kwargs) -> Any:
        """Resolve field value."""
        pass


class QueryComplexityAnalyzer:
    """Analyzer for GraphQL query complexity."""
    
    def __init__(self, max_complexity: int = 1000, max_depth: int = 15):
        self.max_complexity = max_complexity
        self.max_depth = max_depth
    
    def analyze(self, document: Any, variables: Dict[str, Any] = None) -> Dict[str, int]:
        """Analyze query complexity and depth."""
        complexity = self._calculate_complexity(document, variables or {})
        depth = self._calculate_depth(document)
        
        return {
            "complexity": complexity,
            "depth": depth,
            "max_complexity": self.max_complexity,
            "max_depth": self.max_depth
        }
    
    def validate(self, document: Any, variables: Dict[str, Any] = None) -> List[GraphQLError]:
        """Validate query complexity and depth."""
        errors = []
        analysis = self.analyze(document, variables)
        
        if analysis["complexity"] > self.max_complexity:
            errors.append(GraphQLError(
                f"Query complexity {analysis['complexity']} exceeds maximum {self.max_complexity}"
            ))
        
        if analysis["depth"] > self.max_depth:
            errors.append(GraphQLError(
                f"Query depth {analysis['depth']} exceeds maximum {self.max_depth}"
            ))
        
        return errors
    
    def _calculate_complexity(self, document: Any, variables: Dict[str, Any]) -> int:
        """Calculate query complexity."""
        # Simplified complexity calculation
        # In a real implementation, this would traverse the AST
        # and calculate based on field weights and multiplicative factors
        complexity = 0
        
        # Count selections (simplified)
        query_str = str(document)
        complexity += query_str.count('{')
        complexity += query_str.count('[')
        
        return complexity
    
    def _calculate_depth(self, document: Any) -> int:
        """Calculate query depth."""
        # Simplified depth calculation
        # In a real implementation, this would traverse the AST
        query_str = str(document)
        return query_str.count('{')


class SubscriptionManager:
    """Manager for GraphQL subscriptions."""
    
    def __init__(self):
        self.subscriptions: Dict[str, SubscriptionInfo] = {}
        self.event_streams: Dict[str, AsyncIterator] = {}
        self._next_id = 1
    
    def create_subscription(
        self,
        query: str,
        variables: Dict[str, Any],
        context: ResolverContext
    ) -> str:
        """Create a new subscription."""
        subscription_id = f"sub_{self._next_id}"
        self._next_id += 1
        
        self.subscriptions[subscription_id] = SubscriptionInfo(
            id=subscription_id,
            query=query,
            variables=variables,
            context=context
        )
        
        return subscription_id
    
    def remove_subscription(self, subscription_id: str) -> None:
        """Remove a subscription."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
        
        if subscription_id in self.event_streams:
            del self.event_streams[subscription_id]
    
    def get_subscription(self, subscription_id: str) -> Optional[SubscriptionInfo]:
        """Get subscription information."""
        return self.subscriptions.get(subscription_id)
    
    async def publish_event(self, event_type: str, data: Any) -> None:
        """Publish event to relevant subscriptions."""
        for sub_id, sub_info in self.subscriptions.items():
            # Check if subscription is interested in this event type
            if self._matches_subscription(sub_info, event_type):
                sub_info.last_event = datetime.now()
                sub_info.event_count += 1
                
                # Send event to subscription
                if sub_id in self.event_streams:
                    try:
                        await self.event_streams[sub_id].asend(data)
                    except Exception as e:
                        logging.error(f"Failed to send event to subscription {sub_id}: {e}")
                        self.remove_subscription(sub_id)
    
    def _matches_subscription(self, subscription: SubscriptionInfo, event_type: str) -> bool:
        """Check if subscription matches event type."""
        # Simplified matching - in real implementation,
        # this would parse the subscription query and match fields
        return event_type.lower() in subscription.query.lower()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get subscription statistics."""
        active_subs = len(self.subscriptions)
        total_events = sum(sub.event_count for sub in self.subscriptions.values())
        
        return {
            "active_subscriptions": active_subs,
            "total_events_sent": total_events,
            "subscriptions": {
                sub_id: {
                    "created_at": sub.created_at.isoformat(),
                    "last_event": sub.last_event.isoformat() if sub.last_event else None,
                    "event_count": sub.event_count
                }
                for sub_id, sub in self.subscriptions.items()
            }
        }


class AuthDirective:
    """Authentication directive for GraphQL."""
    
    def __init__(self, auth_config: AuthConfig):
        self.auth_config = auth_config
    
    def __call__(self, resolver: Callable) -> Callable:
        """Decorator for authentication."""
        async def wrapper(parent, info, **kwargs):
            context = info.context
            
            if not context.auth_token:
                raise GraphQLError("Authentication required")
            
            # Validate token
            if not self._validate_token(context.auth_token):
                raise GraphQLError("Invalid authentication token")
            
            return await resolver(parent, info, **kwargs)
        
        return wrapper
    
    def _validate_token(self, token: str) -> bool:
        """Validate authentication token."""
        # Basic validation - implement actual logic
        return len(token) > 0


class RateLimitDirective:
    """Rate limiting directive for GraphQL."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, List[datetime]] = {}
    
    def __call__(self, resolver: Callable) -> Callable:
        """Decorator for rate limiting."""
        async def wrapper(parent, info, **kwargs):
            context = info.context
            client_id = context.user_id or context.ip_address or "anonymous"
            
            if not self._check_rate_limit(client_id):
                raise GraphQLError("Rate limit exceeded")
            
            return await resolver(parent, info, **kwargs)
        
        return wrapper
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = datetime.now()
        minute_ago = now.timestamp() - 60
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
        
        # Remove old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time.timestamp() > minute_ago
        ]
        
        # Check limit
        if len(self.request_counts[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.request_counts[client_id].append(now)
        return True


class GraphQLExecutor:
    """GraphQL query executor."""
    
    def __init__(self, schema: GraphQLSchema, config: GraphQLConfig):
        self.schema = schema
        self.config = config
        self.complexity_analyzer = QueryComplexityAnalyzer(
            config.max_query_complexity,
            config.max_query_depth
        )
        self.subscription_manager = SubscriptionManager()
    
    async def execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        context: Optional[ResolverContext] = None
    ) -> ExecutionResult:
        """Execute GraphQL query."""
        try:
            # Parse query
            document = parse(query)
            
            # Validate syntax
            validation_errors = validate(self.schema, document)
            if validation_errors:
                return ExecutionResult(errors=validation_errors)
            
            # Analyze complexity
            complexity_errors = self.complexity_analyzer.validate(document, variables)
            if complexity_errors:
                return ExecutionResult(errors=complexity_errors)
            
            # Create query info
            query_info = QueryInfo(
                operation_name=operation_name,
                operation_type=self._get_operation_type(document),
                query=query,
                variables=variables or {},
                user_id=context.user_id if context else None,
                ip_address=context.ip_address if context else None
            )
            
            # Execute query
            result = await execute(
                self.schema,
                document,
                variable_values=variables,
                operation_name=operation_name,
                context_value=context
            )
            
            return result
            
        except Exception as e:
            return ExecutionResult(errors=[GraphQLError(str(e))])
    
    async def execute_subscription(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        context: Optional[ResolverContext] = None
    ) -> AsyncIterator[ExecutionResult]:
        """Execute GraphQL subscription."""
        try:
            # Parse query
            document = parse(query)
            
            # Validate syntax
            validation_errors = validate(self.schema, document)
            if validation_errors:
                yield ExecutionResult(errors=validation_errors)
                return
            
            # Create subscription
            subscription_id = self.subscription_manager.create_subscription(
                query, variables or {}, context or ResolverContext()
            )
            
            # Execute subscription
            async for result in subscribe(
                self.schema,
                document,
                variable_values=variables,
                operation_name=operation_name,
                context_value=context
            ):
                yield result
                
        except Exception as e:
            yield ExecutionResult(errors=[GraphQLError(str(e))])
    
    def _get_operation_type(self, document: Any) -> OperationType:
        """Get operation type from document."""
        # Simplified operation type detection
        query_str = str(document).lower()
        if "mutation" in query_str:
            return OperationType.MUTATION
        elif "subscription" in query_str:
            return OperationType.SUBSCRIPTION
        else:
            return OperationType.QUERY
    
    def get_schema_sdl(self) -> str:
        """Get schema definition language representation."""
        # This would return the SDL representation of the schema
        return "# GraphQL Schema\n# TODO: Implement SDL generation"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "schema_info": {
                "types_count": len(self.schema.type_map),
                "introspection_enabled": self.config.enable_introspection
            },
            "complexity_limits": {
                "max_complexity": self.config.max_query_complexity,
                "max_depth": self.config.max_query_depth
            },
            "subscriptions": self.subscription_manager.get_stats()
        }


class SchemaBuilder:
    """Builder for GraphQL schemas."""
    
    def __init__(self):
        self.types: Dict[str, Any] = {}
        self.resolvers: Dict[str, Dict[str, Callable]] = {}
        self.directives: Dict[str, Any] = {}
    
    def add_type(self, name: str, type_def: Any) -> None:
        """Add type to schema."""
        self.types[name] = type_def
    
    def add_resolver(self, type_name: str, field_name: str, resolver: Callable) -> None:
        """Add resolver for field."""
        if type_name not in self.resolvers:
            self.resolvers[type_name] = {}
        self.resolvers[type_name][field_name] = resolver
    
    def add_directive(self, name: str, directive: Any) -> None:
        """Add directive to schema."""
        self.directives[name] = directive
    
    def build_from_sdl(self, sdl: str) -> GraphQLSchema:
        """Build schema from SDL string."""
        try:
            schema = build_schema(sdl)
            
            # Add resolvers
            for type_name, field_resolvers in self.resolvers.items():
                if type_name in schema.type_map:
                    type_obj = schema.type_map[type_name]
                    if hasattr(type_obj, 'fields'):
                        for field_name, resolver in field_resolvers.items():
                            if field_name in type_obj.fields:
                                type_obj.fields[field_name].resolve = resolver
            
            return schema
            
        except Exception as e:
            raise APIError(
                error_code=ErrorCode.VALIDATION_ERROR,
                message=f"Failed to build schema from SDL: {str(e)}"
            )
    
    def build_programmatic(self) -> GraphQLSchema:
        """Build schema programmatically."""
        try:
            # Create query type
            query_fields = {}
            if "Query" in self.resolvers:
                for field_name, resolver in self.resolvers["Query"].items():
                    query_fields[field_name] = GraphQLField(
                        GraphQLString,  # Simplified - should determine actual type
                        resolve=resolver
                    )
            
            query_type = GraphQLObjectType(
                name="Query",
                fields=query_fields or {"hello": GraphQLField(GraphQLString)}
            )
            
            # Create mutation type
            mutation_type = None
            if "Mutation" in self.resolvers:
                mutation_fields = {}
                for field_name, resolver in self.resolvers["Mutation"].items():
                    mutation_fields[field_name] = GraphQLField(
                        GraphQLString,  # Simplified
                        resolve=resolver
                    )
                
                mutation_type = GraphQLObjectType(
                    name="Mutation",
                    fields=mutation_fields
                )
            
            # Create subscription type
            subscription_type = None
            if "Subscription" in self.resolvers:
                subscription_fields = {}
                for field_name, resolver in self.resolvers["Subscription"].items():
                    subscription_fields[field_name] = GraphQLField(
                        GraphQLString,  # Simplified
                        resolve=resolver
                    )
                
                subscription_type = GraphQLObjectType(
                    name="Subscription",
                    fields=subscription_fields
                )
            
            return GraphQLSchema(
                query=query_type,
                mutation=mutation_type,
                subscription=subscription_type
            )
            
        except Exception as e:
            raise APIError(
                error_code=ErrorCode.VALIDATION_ERROR,
                message=f"Failed to build schema programmatically: {str(e)}"
            )


class GraphQLServer:
    """GraphQL server implementation."""
    
    def __init__(self, config: GraphQLConfig):
        self.config = config
        self.schema: Optional[GraphQLSchema] = None
        self.executor: Optional[GraphQLExecutor] = None
        self.schema_builder = SchemaBuilder()
    
    def set_schema(self, schema: GraphQLSchema) -> None:
        """Set GraphQL schema."""
        self.schema = schema
        self.executor = GraphQLExecutor(schema, self.config)
    
    def load_schema_from_file(self, file_path: str) -> None:
        """Load schema from SDL file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sdl = f.read()
            
            schema = self.schema_builder.build_from_sdl(sdl)
            self.set_schema(schema)
            
        except Exception as e:
            raise APIError(
                error_code=ErrorCode.FILE_ERROR,
                message=f"Failed to load schema from file: {str(e)}"
            )
    
    def add_resolver(self, type_name: str, field_name: str, resolver: Callable) -> None:
        """Add resolver to schema builder."""
        self.schema_builder.add_resolver(type_name, field_name, resolver)
    
    async def execute(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        context: Optional[ResolverContext] = None
    ) -> Dict[str, Any]:
        """Execute GraphQL operation."""
        if not self.executor:
            raise APIError(
                error_code=ErrorCode.CONFIGURATION_ERROR,
                message="No schema configured"
            )
        
        result = await self.executor.execute_query(
            query, variables, operation_name, context
        )
        
        return {
            "data": result.data,
            "errors": [str(error) for error in result.errors] if result.errors else None
        }
    
    async def subscribe(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        context: Optional[ResolverContext] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute GraphQL subscription."""
        if not self.executor:
            raise APIError(
                error_code=ErrorCode.CONFIGURATION_ERROR,
                message="No schema configured"
            )
        
        async for result in self.executor.execute_subscription(
            query, variables, operation_name, context
        ):
            yield {
                "data": result.data,
                "errors": [str(error) for error in result.errors] if result.errors else None
            }
    
    def get_schema_sdl(self) -> str:
        """Get schema SDL."""
        if not self.executor:
            return "# No schema configured"
        return self.executor.get_schema_sdl()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        base_stats = {
            "config": {
                "introspection_enabled": self.config.enable_introspection,
                "playground_enabled": self.config.enable_playground,
                "subscriptions_enabled": self.config.enable_subscriptions,
                "max_complexity": self.config.max_query_complexity,
                "max_depth": self.config.max_query_depth
            }
        }
        
        if self.executor:
            base_stats.update(self.executor.get_stats())
        
        return base_stats


# Global instances
_graphql_server: Optional[GraphQLServer] = None


def get_graphql_server(config: Optional[GraphQLConfig] = None) -> GraphQLServer:
    """Get global GraphQL server instance."""
    global _graphql_server
    if _graphql_server is None:
        if config is None:
            config = GraphQLConfig()
        _graphql_server = GraphQLServer(config)
    return _graphql_server


def initialize_graphql_api(config: Optional[GraphQLConfig] = None) -> None:
    """Initialize GraphQL API components."""
    global _graphql_server
    
    if config is None:
        config = GraphQLConfig()
    _graphql_server = GraphQLServer(config)


def shutdown_graphql_api() -> None:
    """Shutdown GraphQL API components."""
    global _graphql_server
    _graphql_server = None