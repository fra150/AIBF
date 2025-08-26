"""WebSocket API module for AI Bull Ford.

This module provides comprehensive WebSocket capabilities including:
- Real-time bidirectional communication
- Connection management and pooling
- Message routing and broadcasting
- Authentication and authorization
- Automatic reconnection and heartbeat
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
from weakref import WeakSet

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .common import APIError, ErrorCode, AuthConfig


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(Enum):
    """WebSocket message types."""
    TEXT = "text"
    BINARY = "binary"
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    BROADCAST = "broadcast"
    DIRECT = "direct"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: MessageType
    data: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None
    channel: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps({
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "channel": self.channel,
            "metadata": self.metadata
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data["type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            sender_id=data.get("sender_id"),
            recipient_id=data.get("recipient_id"),
            channel=data.get("channel"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    id: str
    websocket: websockets.WebSocketServerProtocol
    state: ConnectionState
    user_id: Optional[str] = None
    channels: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        """Check if connection is active."""
        return self.state == ConnectionState.CONNECTED


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket server."""
    host: str = "localhost"
    port: int = 8765
    path: str = "/ws"
    max_connections: int = 1000
    heartbeat_interval: float = 30.0
    connection_timeout: float = 60.0
    max_message_size: int = 1024 * 1024  # 1MB
    auth_required: bool = False
    auth_config: Optional[AuthConfig] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    compression: bool = True


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle_message(self, connection: ConnectionInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle incoming message."""
        pass
    
    @abstractmethod
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if handler can handle message type."""
        pass


class DefaultMessageHandler(MessageHandler):
    """Default message handler implementation."""
    
    def __init__(self):
        self._handlers: Dict[MessageType, Callable] = {
            MessageType.PING: self._handle_ping,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.AUTH: self._handle_auth,
            MessageType.SUBSCRIBE: self._handle_subscribe,
            MessageType.UNSUBSCRIBE: self._handle_unsubscribe
        }
    
    async def handle_message(self, connection: ConnectionInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle incoming message."""
        handler = self._handlers.get(message.type)
        if handler:
            return await handler(connection, message)
        return None
    
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if handler can handle message type."""
        return message_type in self._handlers
    
    async def _handle_ping(self, connection: ConnectionInfo, message: WebSocketMessage) -> WebSocketMessage:
        """Handle ping message."""
        return WebSocketMessage(
            type=MessageType.PONG,
            data=message.data,
            recipient_id=message.sender_id
        )
    
    async def _handle_heartbeat(self, connection: ConnectionInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle heartbeat message."""
        connection.last_heartbeat = datetime.now()
        return None
    
    async def _handle_auth(self, connection: ConnectionInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle authentication message."""
        # Basic auth implementation
        token = message.data.get("token")
        if token:
            # TODO: Implement actual token validation
            connection.user_id = message.data.get("user_id")
            connection.metadata["authenticated"] = True
            return WebSocketMessage(
                type=MessageType.AUTH,
                data={"status": "authenticated", "user_id": connection.user_id}
            )
        else:
            return WebSocketMessage(
                type=MessageType.ERROR,
                data={"error": "Authentication failed"}
            )
    
    async def _handle_subscribe(self, connection: ConnectionInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle channel subscription."""
        channel = message.data.get("channel")
        if channel:
            connection.channels.add(channel)
            return WebSocketMessage(
                type=MessageType.SUBSCRIBE,
                data={"status": "subscribed", "channel": channel}
            )
        return None
    
    async def _handle_unsubscribe(self, connection: ConnectionInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Handle channel unsubscription."""
        channel = message.data.get("channel")
        if channel and channel in connection.channels:
            connection.channels.remove(channel)
            return WebSocketMessage(
                type=MessageType.UNSUBSCRIBE,
                data={"status": "unsubscribed", "channel": channel}
            )
        return None


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.connections: Dict[str, ConnectionInfo] = {}
        self.channels: Dict[str, Set[str]] = {}  # channel -> connection_ids
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def add_connection(self, websocket: websockets.WebSocketServerProtocol) -> ConnectionInfo:
        """Add new connection."""
        if len(self.connections) >= self.config.max_connections:
            raise APIError(
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                message="Maximum connections exceeded"
            )
        
        connection_id = str(uuid.uuid4())
        connection = ConnectionInfo(
            id=connection_id,
            websocket=websocket,
            state=ConnectionState.CONNECTED
        )
        
        self.connections[connection_id] = connection
        return connection
    
    async def remove_connection(self, connection_id: str) -> None:
        """Remove connection."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            
            # Remove from channels
            for channel in connection.channels:
                if channel in self.channels:
                    self.channels[channel].discard(connection_id)
                    if not self.channels[channel]:
                        del self.channels[channel]
            
            # Remove from user connections
            if connection.user_id and connection.user_id in self.user_connections:
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
            
            # Update state and remove
            connection.state = ConnectionState.DISCONNECTED
            del self.connections[connection_id]
    
    async def subscribe_to_channel(self, connection_id: str, channel: str) -> None:
        """Subscribe connection to channel."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.channels.add(channel)
            
            if channel not in self.channels:
                self.channels[channel] = set()
            self.channels[channel].add(connection_id)
    
    async def unsubscribe_from_channel(self, connection_id: str, channel: str) -> None:
        """Unsubscribe connection from channel."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.channels.discard(channel)
            
            if channel in self.channels:
                self.channels[channel].discard(connection_id)
                if not self.channels[channel]:
                    del self.channels[channel]
    
    async def broadcast_to_channel(self, channel: str, message: WebSocketMessage) -> int:
        """Broadcast message to all connections in channel."""
        sent_count = 0
        if channel in self.channels:
            for connection_id in self.channels[channel].copy():
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        return sent_count
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """Send message to all connections of a user."""
        sent_count = 0
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id].copy():
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        return sent_count
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific connection."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        if not connection.is_active:
            return False
        
        try:
            await connection.websocket.send(message.to_json())
            return True
        except (ConnectionClosed, WebSocketException):
            await self.remove_connection(connection_id)
            return False
    
    async def broadcast_to_all(self, message: WebSocketMessage) -> int:
        """Broadcast message to all connections."""
        sent_count = 0
        for connection_id in list(self.connections.keys()):
            if await self.send_to_connection(connection_id, message):
                sent_count += 1
        return sent_count
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.connections),
            "active_connections": sum(1 for c in self.connections.values() if c.is_active),
            "total_channels": len(self.channels),
            "total_users": len(self.user_connections)
        }
    
    async def start_heartbeat(self) -> None:
        """Start heartbeat monitoring."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def stop_heartbeat(self) -> None:
        """Stop heartbeat monitoring."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                current_time = datetime.now()
                timeout_threshold = current_time.timestamp() - self.config.connection_timeout
                
                # Check for timed out connections
                timed_out_connections = []
                for connection_id, connection in self.connections.items():
                    if connection.last_heartbeat:
                        if connection.last_heartbeat.timestamp() < timeout_threshold:
                            timed_out_connections.append(connection_id)
                    elif connection.connected_at.timestamp() < timeout_threshold:
                        timed_out_connections.append(connection_id)
                
                # Remove timed out connections
                for connection_id in timed_out_connections:
                    await self.remove_connection(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat error: {e}")


class WebSocketServer:
    """WebSocket server implementation."""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.connection_manager = ConnectionManager(config)
        self.message_handlers: List[MessageHandler] = [DefaultMessageHandler()]
        self._server: Optional[websockets.WebSocketServer] = None
    
    def add_message_handler(self, handler: MessageHandler) -> None:
        """Add custom message handler."""
        self.message_handlers.append(handler)
    
    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """Handle new WebSocket connection."""
        connection = None
        try:
            connection = await self.connection_manager.add_connection(websocket)
            
            # Send welcome message
            welcome_message = WebSocketMessage(
                type=MessageType.TEXT,
                data={"status": "connected", "connection_id": connection.id}
            )
            await self.connection_manager.send_to_connection(connection.id, welcome_message)
            
            # Handle messages
            async for raw_message in websocket:
                try:
                    message = WebSocketMessage.from_json(raw_message)
                    message.sender_id = connection.id
                    
                    # Process message through handlers
                    response = await self._process_message(connection, message)
                    if response:
                        await self.connection_manager.send_to_connection(connection.id, response)
                        
                except json.JSONDecodeError:
                    error_message = WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"error": "Invalid JSON format"}
                    )
                    await self.connection_manager.send_to_connection(connection.id, error_message)
                except Exception as e:
                    error_message = WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"error": str(e)}
                    )
                    await self.connection_manager.send_to_connection(connection.id, error_message)
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            if connection:
                await self.connection_manager.remove_connection(connection.id)
    
    async def _process_message(self, connection: ConnectionInfo, message: WebSocketMessage) -> Optional[WebSocketMessage]:
        """Process message through handlers."""
        for handler in self.message_handlers:
            if handler.can_handle(message.type):
                response = await handler.handle_message(connection, message)
                if response:
                    return response
        
        # Handle routing messages
        if message.type == MessageType.BROADCAST and message.channel:
            await self.connection_manager.broadcast_to_channel(message.channel, message)
        elif message.type == MessageType.DIRECT and message.recipient_id:
            await self.connection_manager.send_to_connection(message.recipient_id, message)
        
        return None
    
    async def start(self) -> None:
        """Start WebSocket server."""
        self._server = await websockets.serve(
            self.handle_connection,
            self.config.host,
            self.config.port,
            path=self.config.path,
            max_size=self.config.max_message_size,
            compression="deflate" if self.config.compression else None
        )
        
        await self.connection_manager.start_heartbeat()
        print(f"WebSocket server started on ws://{self.config.host}:{self.config.port}{self.config.path}")
    
    async def stop(self) -> None:
        """Stop WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        
        await self.connection_manager.stop_heartbeat()
        print("WebSocket server stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "server_config": {
                "host": self.config.host,
                "port": self.config.port,
                "path": self.config.path,
                "max_connections": self.config.max_connections
            },
            "connections": self.connection_manager.get_connection_stats()
        }


class WebSocketClient:
    """WebSocket client implementation."""
    
    def __init__(self, uri: str, auto_reconnect: bool = True):
        self.uri = uri
        self.auto_reconnect = auto_reconnect
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.state = ConnectionState.DISCONNECTED
        self.message_handlers: Dict[MessageType, Callable] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
    
    def on_message(self, message_type: MessageType):
        """Decorator for registering message handlers."""
        def decorator(func: Callable):
            self.message_handlers[message_type] = func
            return func
        return decorator
    
    async def connect(self) -> None:
        """Connect to WebSocket server."""
        try:
            self.state = ConnectionState.CONNECTING
            self.websocket = await websockets.connect(self.uri)
            self.state = ConnectionState.CONNECTED
            
            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Failed to connect: {str(e)}"
            )
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self.state = ConnectionState.DISCONNECTING
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.state = ConnectionState.DISCONNECTED
    
    async def send_message(self, message: WebSocketMessage) -> None:
        """Send message to server."""
        if self.state != ConnectionState.CONNECTED or not self.websocket:
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message="Not connected to server"
            )
        
        try:
            await self.websocket.send(message.to_json())
        except (ConnectionClosed, WebSocketException) as e:
            self.state = ConnectionState.ERROR
            if self.auto_reconnect:
                self._reconnect_task = asyncio.create_task(self._reconnect())
            raise APIError(
                error_code=ErrorCode.CONNECTION_ERROR,
                message=f"Failed to send message: {str(e)}"
            )
    
    async def _receive_loop(self) -> None:
        """Message receiving loop."""
        try:
            while self.state == ConnectionState.CONNECTED and self.websocket:
                try:
                    raw_message = await self.websocket.recv()
                    message = WebSocketMessage.from_json(raw_message)
                    
                    # Handle message
                    handler = self.message_handlers.get(message.type)
                    if handler:
                        await handler(message)
                        
                except json.JSONDecodeError:
                    print("Received invalid JSON message")
                except (ConnectionClosed, WebSocketException):
                    break
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Receive loop error: {e}")
        finally:
            self.state = ConnectionState.ERROR
            if self.auto_reconnect:
                self._reconnect_task = asyncio.create_task(self._reconnect())
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect."""
        max_retries = 5
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(retry_delay * (attempt + 1))
                await self.connect()
                print("Reconnected successfully")
                return
            except Exception as e:
                print(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        print("Failed to reconnect after maximum retries")


# Global instances
_websocket_server: Optional[WebSocketServer] = None
_websocket_client: Optional[WebSocketClient] = None


def get_websocket_server(config: Optional[WebSocketConfig] = None) -> WebSocketServer:
    """Get global WebSocket server instance."""
    global _websocket_server
    if _websocket_server is None:
        if config is None:
            config = WebSocketConfig()
        _websocket_server = WebSocketServer(config)
    return _websocket_server


def get_websocket_client(uri: str) -> WebSocketClient:
    """Get global WebSocket client instance."""
    global _websocket_client
    if _websocket_client is None:
        _websocket_client = WebSocketClient(uri)
    return _websocket_client


def initialize_websocket_api(server_config: Optional[WebSocketConfig] = None) -> None:
    """Initialize WebSocket API components."""
    global _websocket_server
    
    if server_config is None:
        server_config = WebSocketConfig()
    _websocket_server = WebSocketServer(server_config)


async def shutdown_websocket_api() -> None:
    """Shutdown WebSocket API components."""
    global _websocket_server, _websocket_client
    
    if _websocket_server:
        await _websocket_server.stop()
        _websocket_server = None
    
    if _websocket_client:
        await _websocket_client.disconnect()
        _websocket_client = None