"""Multi-agent system module for AI Bull Ford.

This module provides comprehensive multi-agent capabilities including:
- Agent coordination and communication
- Task distribution and load balancing
- Consensus mechanisms and voting
- Agent discovery and registration
- Collaborative problem solving
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Awaitable
from uuid import uuid4

import numpy as np

from .planning import Agent, Task, Plan, AgentConfig, ExecutionStatus, TaskPriority


class CommunicationProtocol(Enum):
    """Communication protocols for agents."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"


class MessageType(Enum):
    """Types of messages between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    COORDINATION = "coordination"
    NEGOTIATION = "negotiation"
    VOTE = "vote"
    CONSENSUS = "consensus"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    CAPABILITY_ANNOUNCEMENT = "capability_announcement"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentRole(Enum):
    """Roles that agents can play in the system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    MONITOR = "monitor"
    BROKER = "broker"
    FACILITATOR = "facilitator"
    LEADER = "leader"
    FOLLOWER = "follower"


class ConsensusAlgorithm(Enum):
    """Consensus algorithms for decision making."""
    MAJORITY_VOTE = "majority_vote"
    UNANIMOUS = "unanimous"
    WEIGHTED_VOTE = "weighted_vote"
    RAFT = "raft"
    PBFT = "pbft"
    PAXOS = "paxos"


@dataclass
class Message:
    """Message between agents."""
    id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    receiver_id: str = ""  # Empty for broadcast
    message_type: MessageType = MessageType.STATUS_UPDATE
    protocol: CommunicationProtocol = CommunicationProtocol.DIRECT
    
    # Content
    content: Dict[str, Any] = field(default_factory=dict)
    payload: Optional[bytes] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    ttl: Optional[timedelta] = None  # Time to live
    reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Routing
    route: List[str] = field(default_factory=list)
    hop_count: int = 0
    max_hops: int = 10
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        return datetime.now() - self.timestamp > self.ttl
    
    def add_hop(self, agent_id: str) -> bool:
        """Add hop to message route."""
        if self.hop_count >= self.max_hops:
            return False
        
        self.route.append(agent_id)
        self.hop_count += 1
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type.value,
            'protocol': self.protocol.value,
            'content': self.content,
            'payload': self.payload.hex() if self.payload else None,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'ttl': self.ttl.total_seconds() if self.ttl else None,
            'reply_to': self.reply_to,
            'conversation_id': self.conversation_id,
            'route': self.route,
            'hop_count': self.hop_count,
            'max_hops': self.max_hops
        }


@dataclass
class AgentInfo:
    """Information about an agent in the system."""
    id: str
    name: str
    role: AgentRole
    capabilities: Set[str] = field(default_factory=set)
    status: str = "active"
    
    # Network information
    address: str = ""
    port: int = 0
    
    # Performance metrics
    load: float = 0.0
    success_rate: float = 1.0
    response_time: float = 0.0
    
    # Metadata
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """Check if agent is available for tasks."""
        return (
            self.status == "active" and
            self.load < 0.9 and
            (datetime.now() - self.last_seen).total_seconds() < 60
        )
    
    def update_metrics(self, load: float, success_rate: float, response_time: float) -> None:
        """Update performance metrics."""
        self.load = max(0.0, min(1.0, load))
        self.success_rate = max(0.0, min(1.0, success_rate))
        self.response_time = max(0.0, response_time)
        self.last_seen = datetime.now()


@dataclass
class TaskAllocation:
    """Task allocation information."""
    task_id: str
    agent_id: str
    allocated_at: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Allocation metadata
    allocation_reason: str = ""
    confidence: float = 1.0
    alternatives: List[str] = field(default_factory=list)


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self):
        self.subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.message_queue: deque = deque()
        self.message_history: List[Message] = []
        self.max_history = 1000
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'average_latency': 0.0
        }
    
    def subscribe(self, message_type: MessageType, handler: Callable[[Message], None]) -> None:
        """Subscribe to messages of a specific type."""
        self.subscribers[message_type].append(handler)
        self.logger.debug(f"Subscribed handler to {message_type.value}")
    
    def unsubscribe(self, message_type: MessageType, handler: Callable[[Message], None]) -> None:
        """Unsubscribe from messages."""
        if handler in self.subscribers[message_type]:
            self.subscribers[message_type].remove(handler)
            self.logger.debug(f"Unsubscribed handler from {message_type.value}")
    
    def send_message(self, message: Message) -> bool:
        """Send message through the bus."""
        if message.is_expired():
            self.stats['messages_dropped'] += 1
            return False
        
        self.message_queue.append(message)
        self.stats['messages_sent'] += 1
        
        # Process message immediately if synchronous
        if message.protocol == CommunicationProtocol.DIRECT:
            self._process_message(message)
        
        return True
    
    def _process_message(self, message: Message) -> None:
        """Process a message by notifying subscribers."""
        handlers = self.subscribers.get(message.message_type, [])
        
        for handler in handlers:
            try:
                handler(message)
                self.stats['messages_received'] += 1
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")
        
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
    
    def process_queue(self) -> None:
        """Process queued messages."""
        while self.message_queue:
            message = self.message_queue.popleft()
            if not message.is_expired():
                self._process_message(message)
            else:
                self.stats['messages_dropped'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return self.stats.copy()


class AgentRegistry:
    """Registry for managing agents in the system."""
    
    def __init__(self):
        self.agents: Dict[str, AgentInfo] = {}
        self.capabilities_index: Dict[str, Set[str]] = defaultdict(set)
        self.role_index: Dict[AgentRole, Set[str]] = defaultdict(set)
        
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, agent_info: AgentInfo) -> bool:
        """Register an agent."""
        try:
            self.agents[agent_info.id] = agent_info
            
            # Update indices
            for capability in agent_info.capabilities:
                self.capabilities_index[capability].add(agent_info.id)
            
            self.role_index[agent_info.role].add(agent_info.id)
            
            self.logger.info(f"Registered agent {agent_info.id} ({agent_info.name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_info.id}: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id not in self.agents:
            return False
        
        agent_info = self.agents[agent_id]
        
        # Remove from indices
        for capability in agent_info.capabilities:
            self.capabilities_index[capability].discard(agent_id)
        
        self.role_index[agent_info.role].discard(agent_id)
        
        # Remove from registry
        del self.agents[agent_id]
        
        self.logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information."""
        return self.agents.get(agent_id)
    
    def find_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Find agents with specific capability."""
        agent_ids = self.capabilities_index.get(capability, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def find_agents_by_role(self, role: AgentRole) -> List[AgentInfo]:
        """Find agents with specific role."""
        agent_ids = self.role_index.get(role, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def get_available_agents(self) -> List[AgentInfo]:
        """Get all available agents."""
        return [agent for agent in self.agents.values() if agent.is_available()]
    
    def update_agent_status(self, agent_id: str, status: str, 
                           load: Optional[float] = None,
                           success_rate: Optional[float] = None,
                           response_time: Optional[float] = None) -> bool:
        """Update agent status and metrics."""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        agent.status = status
        agent.last_seen = datetime.now()
        
        if load is not None or success_rate is not None or response_time is not None:
            agent.update_metrics(
                load if load is not None else agent.load,
                success_rate if success_rate is not None else agent.success_rate,
                response_time if response_time is not None else agent.response_time
            )
        
        return True
    
    def cleanup_stale_agents(self, timeout: timedelta = timedelta(minutes=5)) -> int:
        """Remove agents that haven't been seen recently."""
        current_time = datetime.now()
        stale_agents = []
        
        for agent_id, agent in self.agents.items():
            if current_time - agent.last_seen > timeout:
                stale_agents.append(agent_id)
        
        for agent_id in stale_agents:
            self.unregister_agent(agent_id)
        
        return len(stale_agents)


class TaskDistributor:
    """Distributes tasks among available agents."""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.allocations: Dict[str, TaskAllocation] = {}
        self.task_queue: deque = deque()
        
        self.logger = logging.getLogger(__name__)
        
        # Distribution strategies
        self.strategies = {
            'round_robin': self._round_robin_allocation,
            'load_balanced': self._load_balanced_allocation,
            'capability_based': self._capability_based_allocation,
            'performance_based': self._performance_based_allocation
        }
        
        self.current_strategy = 'load_balanced'
        self._round_robin_index = 0
    
    def distribute_task(self, task: Task, strategy: Optional[str] = None) -> Optional[str]:
        """Distribute task to an appropriate agent."""
        strategy = strategy or self.current_strategy
        
        if strategy not in self.strategies:
            self.logger.error(f"Unknown distribution strategy: {strategy}")
            return None
        
        agent_id = self.strategies[strategy](task)
        
        if agent_id:
            allocation = TaskAllocation(
                task_id=task.id,
                agent_id=agent_id,
                priority=task.priority,
                allocation_reason=f"Allocated using {strategy} strategy"
            )
            
            self.allocations[task.id] = allocation
            self.logger.info(f"Allocated task {task.id} to agent {agent_id}")
        
        return agent_id
    
    def _round_robin_allocation(self, task: Task) -> Optional[str]:
        """Round-robin task allocation."""
        available_agents = self.registry.get_available_agents()
        
        if not available_agents:
            return None
        
        agent = available_agents[self._round_robin_index % len(available_agents)]
        self._round_robin_index += 1
        
        return agent.id
    
    def _load_balanced_allocation(self, task: Task) -> Optional[str]:
        """Load-balanced task allocation."""
        available_agents = self.registry.get_available_agents()
        
        if not available_agents:
            return None
        
        # Find agent with lowest load
        best_agent = min(available_agents, key=lambda a: a.load)
        return best_agent.id
    
    def _capability_based_allocation(self, task: Task) -> Optional[str]:
        """Capability-based task allocation."""
        required_capabilities = task.metadata.get('required_capabilities', set())
        
        if not required_capabilities:
            return self._load_balanced_allocation(task)
        
        # Find agents with required capabilities
        suitable_agents = []
        for capability in required_capabilities:
            agents = self.registry.find_agents_by_capability(capability)
            suitable_agents.extend([a for a in agents if a.is_available()])
        
        if not suitable_agents:
            return None
        
        # Choose agent with best capability match and lowest load
        def score_agent(agent: AgentInfo) -> float:
            capability_score = len(required_capabilities.intersection(agent.capabilities)) / len(required_capabilities)
            load_score = 1.0 - agent.load
            return capability_score * 0.7 + load_score * 0.3
        
        best_agent = max(suitable_agents, key=score_agent)
        return best_agent.id
    
    def _performance_based_allocation(self, task: Task) -> Optional[str]:
        """Performance-based task allocation."""
        available_agents = self.registry.get_available_agents()
        
        if not available_agents:
            return None
        
        # Score agents based on performance metrics
        def score_agent(agent: AgentInfo) -> float:
            success_score = agent.success_rate
            speed_score = 1.0 / max(1.0, agent.response_time)
            load_score = 1.0 - agent.load
            
            return success_score * 0.4 + speed_score * 0.3 + load_score * 0.3
        
        best_agent = max(available_agents, key=score_agent)
        return best_agent.id
    
    def get_task_allocation(self, task_id: str) -> Optional[TaskAllocation]:
        """Get allocation information for a task."""
        return self.allocations.get(task_id)
    
    def complete_task_allocation(self, task_id: str) -> bool:
        """Mark task allocation as complete."""
        if task_id in self.allocations:
            del self.allocations[task_id]
            return True
        return False
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        agent_loads = defaultdict(int)
        
        for allocation in self.allocations.values():
            agent_loads[allocation.agent_id] += 1
        
        return {
            'active_allocations': len(self.allocations),
            'agent_loads': dict(agent_loads),
            'current_strategy': self.current_strategy
        }


class ConsensusManager:
    """Manages consensus and voting among agents."""
    
    def __init__(self, registry: AgentRegistry, message_bus: MessageBus):
        self.registry = registry
        self.message_bus = message_bus
        self.logger = logging.getLogger(__name__)
        
        # Active consensus sessions
        self.consensus_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Subscribe to vote messages
        self.message_bus.subscribe(MessageType.VOTE, self._handle_vote)
        self.message_bus.subscribe(MessageType.CONSENSUS, self._handle_consensus)
    
    def initiate_consensus(self, proposal_id: str, proposal: Dict[str, Any], 
                          algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY_VOTE,
                          participants: Optional[List[str]] = None,
                          timeout: timedelta = timedelta(minutes=5)) -> str:
        """Initiate a consensus process."""
        session_id = str(uuid4())
        
        # Determine participants
        if participants is None:
            participants = [agent.id for agent in self.registry.get_available_agents()]
        
        # Create consensus session
        session = {
            'id': session_id,
            'proposal_id': proposal_id,
            'proposal': proposal,
            'algorithm': algorithm,
            'participants': set(participants),
            'votes': {},
            'status': 'active',
            'created_at': datetime.now(),
            'timeout': timeout,
            'result': None
        }
        
        self.consensus_sessions[session_id] = session
        
        # Send consensus request to participants
        for participant_id in participants:
            message = Message(
                sender_id="consensus_manager",
                receiver_id=participant_id,
                message_type=MessageType.CONSENSUS,
                content={
                    'session_id': session_id,
                    'proposal_id': proposal_id,
                    'proposal': proposal,
                    'algorithm': algorithm.value,
                    'action': 'request_vote'
                }
            )
            
            self.message_bus.send_message(message)
        
        self.logger.info(f"Initiated consensus session {session_id} for proposal {proposal_id}")
        return session_id
    
    def _handle_vote(self, message: Message) -> None:
        """Handle vote message."""
        content = message.content
        session_id = content.get('session_id')
        vote = content.get('vote')
        
        if session_id not in self.consensus_sessions:
            return
        
        session = self.consensus_sessions[session_id]
        
        if message.sender_id in session['participants']:
            session['votes'][message.sender_id] = vote
            self.logger.debug(f"Received vote from {message.sender_id} for session {session_id}")
            
            # Check if consensus is reached
            self._check_consensus(session_id)
    
    def _handle_consensus(self, message: Message) -> None:
        """Handle consensus message."""
        content = message.content
        action = content.get('action')
        
        if action == 'request_vote':
            # This would be handled by individual agents
            pass
        elif action == 'result':
            # Consensus result notification
            session_id = content.get('session_id')
            result = content.get('result')
            self.logger.info(f"Consensus result for session {session_id}: {result}")
    
    def _check_consensus(self, session_id: str) -> None:
        """Check if consensus has been reached."""
        session = self.consensus_sessions[session_id]
        algorithm = session['algorithm']
        votes = session['votes']
        participants = session['participants']
        
        # Check if all participants have voted
        if len(votes) < len(participants):
            return
        
        result = None
        
        if algorithm == ConsensusAlgorithm.MAJORITY_VOTE:
            result = self._majority_vote(votes)
        elif algorithm == ConsensusAlgorithm.UNANIMOUS:
            result = self._unanimous_vote(votes)
        elif algorithm == ConsensusAlgorithm.WEIGHTED_VOTE:
            result = self._weighted_vote(votes, participants)
        
        if result is not None:
            session['status'] = 'completed'
            session['result'] = result
            
            # Notify participants of result
            self._notify_consensus_result(session_id, result)
            
            self.logger.info(f"Consensus reached for session {session_id}: {result}")
    
    def _majority_vote(self, votes: Dict[str, Any]) -> Optional[Any]:
        """Calculate majority vote result."""
        vote_counts = defaultdict(int)
        
        for vote in votes.values():
            vote_counts[str(vote)] += 1
        
        if not vote_counts:
            return None
        
        max_votes = max(vote_counts.values())
        majority_threshold = len(votes) / 2
        
        if max_votes > majority_threshold:
            return max(vote_counts.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _unanimous_vote(self, votes: Dict[str, Any]) -> Optional[Any]:
        """Calculate unanimous vote result."""
        if not votes:
            return None
        
        first_vote = next(iter(votes.values()))
        
        if all(vote == first_vote for vote in votes.values()):
            return first_vote
        
        return None
    
    def _weighted_vote(self, votes: Dict[str, Any], participants: Set[str]) -> Optional[Any]:
        """Calculate weighted vote result."""
        # Simple implementation: weight by agent success rate
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        
        for agent_id, vote in votes.items():
            agent = self.registry.get_agent(agent_id)
            weight = agent.success_rate if agent else 1.0
            
            weighted_votes[str(vote)] += weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Find vote with highest weighted score
        best_vote = max(weighted_votes.items(), key=lambda x: x[1])
        
        if best_vote[1] > total_weight / 2:
            return best_vote[0]
        
        return None
    
    def _notify_consensus_result(self, session_id: str, result: Any) -> None:
        """Notify participants of consensus result."""
        session = self.consensus_sessions[session_id]
        
        for participant_id in session['participants']:
            message = Message(
                sender_id="consensus_manager",
                receiver_id=participant_id,
                message_type=MessageType.CONSENSUS,
                content={
                    'session_id': session_id,
                    'proposal_id': session['proposal_id'],
                    'action': 'result',
                    'result': result
                }
            )
            
            self.message_bus.send_message(message)
    
    def get_consensus_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of consensus session."""
        session = self.consensus_sessions.get(session_id)
        
        if session:
            return {
                'id': session['id'],
                'proposal_id': session['proposal_id'],
                'algorithm': session['algorithm'].value,
                'status': session['status'],
                'participants': len(session['participants']),
                'votes_received': len(session['votes']),
                'result': session['result'],
                'created_at': session['created_at'].isoformat()
            }
        
        return None
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired consensus sessions."""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.consensus_sessions.items():
            if current_time - session['created_at'] > session['timeout']:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.consensus_sessions[session_id]
        
        return len(expired_sessions)


class MultiAgentAgent(Agent):
    """Enhanced agent with multi-agent capabilities."""
    
    def __init__(self, config: AgentConfig, registry: AgentRegistry, 
                 message_bus: MessageBus, role: AgentRole = AgentRole.WORKER):
        super().__init__(config)
        
        self.registry = registry
        self.message_bus = message_bus
        self.role = role
        
        # Multi-agent state
        self.peers: Set[str] = set()
        self.conversations: Dict[str, List[Message]] = defaultdict(list)
        
        # Register with the system
        self.agent_info = AgentInfo(
            id=self.id,
            name=self.name,
            role=self.role,
            capabilities=self.capabilities.copy()
        )
        
        self.registry.register_agent(self.agent_info)
        
        # Subscribe to relevant messages
        self.message_bus.subscribe(MessageType.TASK_REQUEST, self._handle_task_request)
        self.message_bus.subscribe(MessageType.TASK_ASSIGNMENT, self._handle_task_assignment)
        self.message_bus.subscribe(MessageType.COORDINATION, self._handle_coordination)
        self.message_bus.subscribe(MessageType.HEARTBEAT, self._handle_heartbeat)
        
        self.logger.info(f"Multi-agent {self.name} initialized with role {self.role.value}")
    
    def send_message(self, receiver_id: str, message_type: MessageType, 
                    content: Dict[str, Any], protocol: CommunicationProtocol = CommunicationProtocol.DIRECT) -> bool:
        """Send message to another agent."""
        message = Message(
            sender_id=self.id,
            receiver_id=receiver_id,
            message_type=message_type,
            protocol=protocol,
            content=content
        )
        
        return self.message_bus.send_message(message)
    
    def broadcast_message(self, message_type: MessageType, content: Dict[str, Any]) -> bool:
        """Broadcast message to all agents."""
        message = Message(
            sender_id=self.id,
            receiver_id="",  # Empty for broadcast
            message_type=message_type,
            protocol=CommunicationProtocol.BROADCAST,
            content=content
        )
        
        return self.message_bus.send_message(message)
    
    def request_task_help(self, task: Task, required_capabilities: Set[str]) -> List[str]:
        """Request help from other agents for a task."""
        suitable_agents = []
        
        for capability in required_capabilities:
            agents = self.registry.find_agents_by_capability(capability)
            suitable_agents.extend([a.id for a in agents if a.is_available() and a.id != self.id])
        
        # Remove duplicates
        suitable_agents = list(set(suitable_agents))
        
        # Send task requests
        for agent_id in suitable_agents:
            self.send_message(
                receiver_id=agent_id,
                message_type=MessageType.TASK_REQUEST,
                content={
                    'task_id': task.id,
                    'task_description': task.description,
                    'required_capabilities': list(required_capabilities),
                    'priority': task.priority.value
                }
            )
        
        return suitable_agents
    
    def _handle_task_request(self, message: Message) -> None:
        """Handle task request from another agent."""
        if message.receiver_id != self.id and message.receiver_id != "":
            return
        
        content = message.content
        task_id = content.get('task_id')
        required_capabilities = set(content.get('required_capabilities', []))
        
        # Check if we can help
        if required_capabilities.issubset(self.capabilities) and len(self.active_tasks) < self.config.max_concurrent_tasks:
            # Send positive response
            self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task_id': task_id,
                    'response': 'accept',
                    'estimated_time': 60,  # seconds
                    'confidence': 0.8
                }
            )
        else:
            # Send negative response
            self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task_id': task_id,
                    'response': 'decline',
                    'reason': 'insufficient_capabilities' if not required_capabilities.issubset(self.capabilities) else 'at_capacity'
                }
            )
    
    def _handle_task_assignment(self, message: Message) -> None:
        """Handle task assignment."""
        if message.receiver_id != self.id:
            return
        
        content = message.content
        task_data = content.get('task')
        
        if task_data:
            # Create task from data
            task = Task(
                id=task_data.get('id', str(uuid4())),
                name=task_data.get('name', ''),
                description=task_data.get('description', ''),
                priority=TaskPriority(task_data.get('priority', 'medium')),
                parameters=task_data.get('parameters', {})
            )
            
            # Assign task
            if self.assign_task(task):
                self.send_message(
                    receiver_id=message.sender_id,
                    message_type=MessageType.STATUS_UPDATE,
                    content={
                        'task_id': task.id,
                        'status': 'accepted'
                    }
                )
            else:
                self.send_message(
                    receiver_id=message.sender_id,
                    message_type=MessageType.STATUS_UPDATE,
                    content={
                        'task_id': task.id,
                        'status': 'rejected',
                        'reason': 'cannot_accept'
                    }
                )
    
    def _handle_coordination(self, message: Message) -> None:
        """Handle coordination message."""
        if message.receiver_id != self.id and message.receiver_id != "":
            return
        
        content = message.content
        coordination_type = content.get('type')
        
        if coordination_type == 'sync_request':
            # Respond with current status
            self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.COORDINATION,
                content={
                    'type': 'sync_response',
                    'status': self.get_status()
                }
            )
    
    def _handle_heartbeat(self, message: Message) -> None:
        """Handle heartbeat message."""
        # Update registry with current status
        self.registry.update_agent_status(
            self.id,
            "active",
            load=len(self.active_tasks) / self.config.max_concurrent_tasks,
            success_rate=self.performance_metrics['success_rate']
        )
    
    def start_heartbeat(self, interval: float = 30.0) -> None:
        """Start sending periodic heartbeats."""
        def heartbeat_loop():
            while True:
                self.broadcast_message(
                    MessageType.HEARTBEAT,
                    {'agent_id': self.id, 'timestamp': datetime.now().isoformat()}
                )
                time.sleep(interval)
        
        import threading
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
    
    def shutdown(self) -> None:
        """Shutdown agent and cleanup."""
        # Notify other agents
        self.broadcast_message(
            MessageType.SHUTDOWN,
            {'agent_id': self.id, 'timestamp': datetime.now().isoformat()}
        )
        
        # Unregister from registry
        self.registry.unregister_agent(self.id)
        
        self.logger.info(f"Agent {self.name} shutdown complete")


class MultiAgentSystem:
    """Main multi-agent system coordinator."""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.registry = AgentRegistry()
        self.task_distributor = TaskDistributor(self.registry)
        self.consensus_manager = ConsensusManager(self.registry, self.message_bus)
        
        self.agents: Dict[str, MultiAgentAgent] = {}
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
        
        # System statistics
        self.stats = {
            'agents_created': 0,
            'tasks_distributed': 0,
            'consensus_sessions': 0,
            'messages_processed': 0
        }
    
    def create_agent(self, config: AgentConfig, role: AgentRole = AgentRole.WORKER) -> MultiAgentAgent:
        """Create and register a new agent."""
        agent = MultiAgentAgent(config, self.registry, self.message_bus, role)
        self.agents[agent.id] = agent
        
        self.stats['agents_created'] += 1
        self.logger.info(f"Created agent {agent.name} with role {role.value}")
        
        return agent
    
    def distribute_task(self, task: Task, strategy: Optional[str] = None) -> Optional[str]:
        """Distribute task to an appropriate agent."""
        agent_id = self.task_distributor.distribute_task(task, strategy)
        
        if agent_id and agent_id in self.agents:
            # Send task assignment
            agent = self.agents[agent_id]
            success = agent.assign_task(task)
            
            if success:
                self.stats['tasks_distributed'] += 1
                return agent_id
        
        return None
    
    def initiate_consensus(self, proposal_id: str, proposal: Dict[str, Any], 
                          algorithm: ConsensusAlgorithm = ConsensusAlgorithm.MAJORITY_VOTE) -> str:
        """Initiate consensus among agents."""
        session_id = self.consensus_manager.initiate_consensus(proposal_id, proposal, algorithm)
        self.stats['consensus_sessions'] += 1
        return session_id
    
    def start(self) -> None:
        """Start the multi-agent system."""
        self.is_running = True
        
        # Start message processing loop
        def message_loop():
            while self.is_running:
                self.message_bus.process_queue()
                self.stats['messages_processed'] = self.message_bus.get_stats()['messages_received']
                time.sleep(0.1)
        
        import threading
        message_thread = threading.Thread(target=message_loop, daemon=True)
        message_thread.start()
        
        # Start cleanup loop
        def cleanup_loop():
            while self.is_running:
                self.registry.cleanup_stale_agents()
                self.consensus_manager.cleanup_expired_sessions()
                time.sleep(60)  # Cleanup every minute
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        self.logger.info("Multi-agent system started")
    
    def stop(self) -> None:
        """Stop the multi-agent system."""
        self.is_running = False
        
        # Shutdown all agents
        for agent in self.agents.values():
            agent.shutdown()
        
        self.logger.info("Multi-agent system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            'is_running': self.is_running,
            'total_agents': len(self.agents),
            'active_agents': len(self.registry.get_available_agents()),
            'message_bus_stats': self.message_bus.get_stats(),
            'allocation_stats': self.task_distributor.get_allocation_stats(),
            'system_stats': self.stats
        }


# Global multi-agent system instance
_multi_agent_system: Optional[MultiAgentSystem] = None


def get_multi_agent_system() -> MultiAgentSystem:
    """Get or create global multi-agent system instance."""
    global _multi_agent_system
    
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
    
    return _multi_agent_system


def initialize_multi_agent_system() -> MultiAgentSystem:
    """Initialize global multi-agent system."""
    global _multi_agent_system
    _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system


def shutdown_multi_agent_system() -> None:
    """Shutdown global multi-agent system."""
    global _multi_agent_system
    
    if _multi_agent_system:
        _multi_agent_system.stop()
        _multi_agent_system = None