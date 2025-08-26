"""Agent autonomy module for AI Bull Ford.

This module provides comprehensive autonomy capabilities including:
- Self-monitoring and health management
- Adaptive behavior and learning
- Goal-driven autonomous operation
- Resource management and optimization
- Self-healing and recovery mechanisms
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from uuid import uuid4

import numpy as np

from .planning import Agent, Task, Plan, AgentConfig, ExecutionStatus, TaskPriority
from .multi_agent import MultiAgentAgent, AgentRole, MessageType


class AutonomyLevel(Enum):
    """Levels of agent autonomy."""
    MANUAL = "manual"  # Human-controlled
    ASSISTED = "assisted"  # Human-guided with AI assistance
    SUPERVISED = "supervised"  # AI-driven with human oversight
    AUTONOMOUS = "autonomous"  # Fully autonomous with minimal human intervention
    SELF_GOVERNING = "self_governing"  # Complete self-governance


class HealthStatus(Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class LearningMode(Enum):
    """Learning modes for adaptive behavior."""
    PASSIVE = "passive"  # Learn from observations
    ACTIVE = "active"  # Actively seek learning opportunities
    REINFORCEMENT = "reinforcement"  # Learn from rewards/penalties
    IMITATION = "imitation"  # Learn by imitating others
    EXPLORATION = "exploration"  # Learn through exploration


class ResourceType(Enum):
    """Types of resources managed by agents."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    ENERGY = "energy"
    TIME = "time"
    TOKENS = "tokens"
    CUSTOM = "custom"


@dataclass
class HealthMetric:
    """Health metric for monitoring agent status."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_status(self) -> HealthStatus:
        """Get health status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def is_healthy(self) -> bool:
        """Check if metric is healthy."""
        return self.get_status() == HealthStatus.HEALTHY


@dataclass
class ResourceUsage:
    """Resource usage information."""
    resource_type: ResourceType
    current_usage: float
    max_capacity: float
    reserved: float = 0.0
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_utilization(self) -> float:
        """Get resource utilization percentage."""
        if self.max_capacity == 0:
            return 0.0
        return (self.current_usage / self.max_capacity) * 100
    
    def get_available(self) -> float:
        """Get available resource amount."""
        return max(0.0, self.max_capacity - self.current_usage - self.reserved)
    
    def is_overloaded(self, threshold: float = 90.0) -> bool:
        """Check if resource is overloaded."""
        return self.get_utilization() > threshold


@dataclass
class Goal:
    """Autonomous goal for agents."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Goal parameters
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    # Status
    status: ExecutionStatus = ExecutionStatus.PENDING
    progress: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_achieved(self) -> bool:
        """Check if goal is achieved."""
        if self.target_value is not None and self.current_value is not None:
            return self.current_value >= self.target_value
        
        # Check success criteria
        for criterion, target in self.success_criteria.items():
            current = self.metadata.get(criterion)
            if current is None or current < target:
                return False
        
        return True
    
    def update_progress(self) -> None:
        """Update goal progress."""
        if self.target_value is not None and self.current_value is not None:
            if self.target_value > 0:
                self.progress = min(1.0, self.current_value / self.target_value)
            else:
                self.progress = 1.0 if self.current_value == self.target_value else 0.0
        elif self.success_criteria:
            achieved_criteria = 0
            for criterion, target in self.success_criteria.items():
                current = self.metadata.get(criterion, 0)
                if current >= target:
                    achieved_criteria += 1
            
            self.progress = achieved_criteria / len(self.success_criteria)
        
        if self.is_achieved():
            self.status = ExecutionStatus.COMPLETED


@dataclass
class LearningExperience:
    """Learning experience for adaptive behavior."""
    id: str = field(default_factory=lambda: str(uuid4()))
    context: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    outcome: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Learning metadata
    success: bool = True
    confidence: float = 1.0
    tags: Set[str] = field(default_factory=set)


class HealthMonitor:
    """Monitors agent health and performance."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.metrics: Dict[str, HealthMetric] = {}
        self.history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.max_history = 1000
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default health metrics."""
        default_metrics = {
            'cpu_usage': HealthMetric('cpu_usage', 0.0, 70.0, 90.0, '%', 'CPU utilization'),
            'memory_usage': HealthMetric('memory_usage', 0.0, 80.0, 95.0, '%', 'Memory utilization'),
            'task_success_rate': HealthMetric('task_success_rate', 100.0, 80.0, 60.0, '%', 'Task success rate'),
            'response_time': HealthMetric('response_time', 0.0, 5.0, 10.0, 's', 'Average response time'),
            'error_rate': HealthMetric('error_rate', 0.0, 5.0, 10.0, '%', 'Error rate')
        }
        
        for name, metric in default_metrics.items():
            self.metrics[name] = metric
    
    def update_metric(self, name: str, value: float) -> None:
        """Update a health metric."""
        if name in self.metrics:
            self.metrics[name].value = value
            self.metrics[name].timestamp = datetime.now()
            
            # Add to history
            self.history[name].append((datetime.now(), value))
            
            # Trim history if needed
            if len(self.history[name]) > self.max_history:
                self.history[name].pop(0)
            
            # Log warnings/critical status
            status = self.metrics[name].get_status()
            if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self.logger.warning(f"Health metric {name} is {status.value}: {value}")
    
    def add_custom_metric(self, metric: HealthMetric) -> None:
        """Add custom health metric."""
        self.metrics[metric.name] = metric
        self.logger.info(f"Added custom health metric: {metric.name}")
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall health status."""
        if not self.metrics:
            return HealthStatus.HEALTHY
        
        statuses = [metric.get_status() for metric in self.metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            'agent_id': self.agent_id,
            'overall_status': self.get_overall_health().value,
            'metrics': {
                name: {
                    'value': metric.value,
                    'status': metric.get_status().value,
                    'threshold_warning': metric.threshold_warning,
                    'threshold_critical': metric.threshold_critical,
                    'unit': metric.unit,
                    'description': metric.description,
                    'timestamp': metric.timestamp.isoformat()
                }
                for name, metric in self.metrics.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metric_trend(self, metric_name: str, duration: timedelta = timedelta(hours=1)) -> Optional[str]:
        """Get trend for a specific metric."""
        if metric_name not in self.history:
            return None
        
        cutoff_time = datetime.now() - duration
        recent_values = [
            value for timestamp, value in self.history[metric_name]
            if timestamp >= cutoff_time
        ]
        
        if len(recent_values) < 2:
            return "insufficient_data"
        
        # Simple trend analysis
        first_half = recent_values[:len(recent_values)//2]
        second_half = recent_values[len(recent_values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        change_percent = ((avg_second - avg_first) / avg_first) * 100 if avg_first != 0 else 0
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"


class ResourceManager:
    """Manages agent resources and optimization."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.resources: Dict[ResourceType, ResourceUsage] = {}
        self.reservations: Dict[str, Tuple[ResourceType, float, datetime]] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default resources
        self._initialize_default_resources()
    
    def _initialize_default_resources(self) -> None:
        """Initialize default resource tracking."""
        default_resources = {
            ResourceType.CPU: ResourceUsage(ResourceType.CPU, 0.0, 100.0, unit='%'),
            ResourceType.MEMORY: ResourceUsage(ResourceType.MEMORY, 0.0, 1024.0, unit='MB'),
            ResourceType.STORAGE: ResourceUsage(ResourceType.STORAGE, 0.0, 10240.0, unit='MB'),
            ResourceType.NETWORK: ResourceUsage(ResourceType.NETWORK, 0.0, 1000.0, unit='Mbps'),
            ResourceType.TOKENS: ResourceUsage(ResourceType.TOKENS, 0.0, 10000.0, unit='tokens')
        }
        
        for resource_type, usage in default_resources.items():
            self.resources[resource_type] = usage
    
    def update_resource_usage(self, resource_type: ResourceType, current_usage: float) -> None:
        """Update current resource usage."""
        if resource_type in self.resources:
            self.resources[resource_type].current_usage = current_usage
            self.resources[resource_type].timestamp = datetime.now()
            
            if self.resources[resource_type].is_overloaded():
                self.logger.warning(f"Resource {resource_type.value} is overloaded: {current_usage}")
    
    def reserve_resource(self, resource_type: ResourceType, amount: float, 
                        duration: timedelta = timedelta(hours=1)) -> Optional[str]:
        """Reserve resources for future use."""
        if resource_type not in self.resources:
            return None
        
        resource = self.resources[resource_type]
        
        if resource.get_available() >= amount:
            reservation_id = str(uuid4())
            expiry = datetime.now() + duration
            
            self.reservations[reservation_id] = (resource_type, amount, expiry)
            resource.reserved += amount
            
            self.logger.info(f"Reserved {amount} {resource.unit} of {resource_type.value}")
            return reservation_id
        
        return None
    
    def release_reservation(self, reservation_id: str) -> bool:
        """Release a resource reservation."""
        if reservation_id not in self.reservations:
            return False
        
        resource_type, amount, _ = self.reservations[reservation_id]
        
        if resource_type in self.resources:
            self.resources[resource_type].reserved -= amount
            self.resources[resource_type].reserved = max(0.0, self.resources[resource_type].reserved)
        
        del self.reservations[reservation_id]
        self.logger.info(f"Released reservation {reservation_id}")
        return True
    
    def cleanup_expired_reservations(self) -> int:
        """Clean up expired reservations."""
        current_time = datetime.now()
        expired_reservations = []
        
        for reservation_id, (_, _, expiry) in self.reservations.items():
            if current_time > expiry:
                expired_reservations.append(reservation_id)
        
        for reservation_id in expired_reservations:
            self.release_reservation(reservation_id)
        
        return len(expired_reservations)
    
    def optimize_resources(self) -> Dict[str, Any]:
        """Optimize resource allocation."""
        optimizations = []
        
        for resource_type, resource in self.resources.items():
            utilization = resource.get_utilization()
            
            if utilization > 90:
                optimizations.append({
                    'resource': resource_type.value,
                    'action': 'scale_up',
                    'reason': f'High utilization: {utilization:.1f}%'
                })
            elif utilization < 20:
                optimizations.append({
                    'resource': resource_type.value,
                    'action': 'scale_down',
                    'reason': f'Low utilization: {utilization:.1f}%'
                })
        
        return {
            'optimizations': optimizations,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive resource report."""
        return {
            'agent_id': self.agent_id,
            'resources': {
                resource_type.value: {
                    'current_usage': resource.current_usage,
                    'max_capacity': resource.max_capacity,
                    'reserved': resource.reserved,
                    'available': resource.get_available(),
                    'utilization': resource.get_utilization(),
                    'unit': resource.unit,
                    'timestamp': resource.timestamp.isoformat()
                }
                for resource_type, resource in self.resources.items()
            },
            'active_reservations': len(self.reservations),
            'timestamp': datetime.now().isoformat()
        }


class AdaptiveLearner:
    """Implements adaptive learning for autonomous behavior."""
    
    def __init__(self, agent_id: str, learning_mode: LearningMode = LearningMode.PASSIVE):
        self.agent_id = agent_id
        self.learning_mode = learning_mode
        
        # Learning data
        self.experiences: List[LearningExperience] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.behavior_patterns: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.memory_size = 10000
        
        self.logger = logging.getLogger(__name__)
    
    def add_experience(self, experience: LearningExperience) -> None:
        """Add learning experience."""
        self.experiences.append(experience)
        
        # Trim experiences if needed
        if len(self.experiences) > self.memory_size:
            self.experiences.pop(0)
        
        # Update behavior patterns
        self._update_behavior_patterns(experience)
        
        self.logger.debug(f"Added learning experience: {experience.action} -> {experience.reward}")
    
    def _update_behavior_patterns(self, experience: LearningExperience) -> None:
        """Update behavior patterns based on experience."""
        context_key = self._serialize_context(experience.context)
        action = experience.action
        
        # Update action value using simple Q-learning approach
        current_value = self.behavior_patterns[context_key][action]
        new_value = current_value + self.learning_rate * (experience.reward - current_value)
        self.behavior_patterns[context_key][action] = new_value
    
    def _serialize_context(self, context: Dict[str, Any]) -> str:
        """Serialize context for pattern matching."""
        # Simple serialization - in practice, this would be more sophisticated
        return json.dumps(context, sort_keys=True)
    
    def suggest_action(self, context: Dict[str, Any], available_actions: List[str]) -> str:
        """Suggest best action based on learned patterns."""
        context_key = self._serialize_context(context)
        
        if context_key not in self.behavior_patterns:
            # No experience with this context, choose randomly or use default
            return np.random.choice(available_actions) if available_actions else ""
        
        patterns = self.behavior_patterns[context_key]
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            # Explore: choose random action
            return np.random.choice(available_actions) if available_actions else ""
        else:
            # Exploit: choose best known action
            best_action = max(
                (action for action in available_actions if action in patterns),
                key=lambda a: patterns[a],
                default=available_actions[0] if available_actions else ""
            )
            return best_action
    
    def get_action_confidence(self, context: Dict[str, Any], action: str) -> float:
        """Get confidence in an action for given context."""
        context_key = self._serialize_context(context)
        
        if context_key not in self.behavior_patterns:
            return 0.5  # Neutral confidence
        
        patterns = self.behavior_patterns[context_key]
        
        if action not in patterns:
            return 0.5
        
        # Normalize confidence to 0-1 range
        action_value = patterns[action]
        max_value = max(patterns.values()) if patterns else 1.0
        min_value = min(patterns.values()) if patterns else 0.0
        
        if max_value == min_value:
            return 0.5
        
        return (action_value - min_value) / (max_value - min_value)
    
    def analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze learning progress and effectiveness."""
        if not self.experiences:
            return {'status': 'no_data'}
        
        # Calculate learning metrics
        recent_experiences = self.experiences[-100:]  # Last 100 experiences
        
        avg_reward = sum(exp.reward for exp in recent_experiences) / len(recent_experiences)
        success_rate = sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences)
        
        # Learning trend
        if len(recent_experiences) >= 50:
            first_half = recent_experiences[:25]
            second_half = recent_experiences[25:]
            
            first_avg = sum(exp.reward for exp in first_half) / len(first_half)
            second_avg = sum(exp.reward for exp in second_half) / len(second_half)
            
            improvement = second_avg - first_avg
        else:
            improvement = 0.0
        
        return {
            'total_experiences': len(self.experiences),
            'recent_avg_reward': avg_reward,
            'recent_success_rate': success_rate,
            'learning_improvement': improvement,
            'behavior_patterns': len(self.behavior_patterns),
            'learning_mode': self.learning_mode.value,
            'exploration_rate': self.exploration_rate
        }
    
    def save_knowledge(self, filepath: Path) -> bool:
        """Save learned knowledge to file."""
        try:
            knowledge_data = {
                'agent_id': self.agent_id,
                'learning_mode': self.learning_mode.value,
                'behavior_patterns': dict(self.behavior_patterns),
                'knowledge_base': self.knowledge_base,
                'learning_parameters': {
                    'learning_rate': self.learning_rate,
                    'exploration_rate': self.exploration_rate,
                    'memory_size': self.memory_size
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(knowledge_data, f, indent=2)
            
            self.logger.info(f"Saved knowledge to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save knowledge: {e}")
            return False
    
    def load_knowledge(self, filepath: Path) -> bool:
        """Load learned knowledge from file."""
        try:
            with open(filepath, 'r') as f:
                knowledge_data = json.load(f)
            
            self.behavior_patterns = defaultdict(lambda: defaultdict(float))
            for context, actions in knowledge_data.get('behavior_patterns', {}).items():
                for action, value in actions.items():
                    self.behavior_patterns[context][action] = value
            
            self.knowledge_base = knowledge_data.get('knowledge_base', {})
            
            params = knowledge_data.get('learning_parameters', {})
            self.learning_rate = params.get('learning_rate', self.learning_rate)
            self.exploration_rate = params.get('exploration_rate', self.exploration_rate)
            self.memory_size = params.get('memory_size', self.memory_size)
            
            self.logger.info(f"Loaded knowledge from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge: {e}")
            return False


class GoalManager:
    """Manages autonomous goals and objectives."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.goals: Dict[str, Goal] = {}
        self.goal_hierarchy: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        
        self.logger = logging.getLogger(__name__)
    
    def add_goal(self, goal: Goal, parent_id: Optional[str] = None) -> bool:
        """Add a new goal."""
        try:
            self.goals[goal.id] = goal
            
            if parent_id:
                if parent_id not in self.goal_hierarchy:
                    self.goal_hierarchy[parent_id] = []
                self.goal_hierarchy[parent_id].append(goal.id)
            
            self.logger.info(f"Added goal: {goal.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add goal: {e}")
            return False
    
    def update_goal_progress(self, goal_id: str, current_value: Optional[float] = None, 
                           metadata_updates: Optional[Dict[str, Any]] = None) -> bool:
        """Update goal progress."""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        
        if current_value is not None:
            goal.current_value = current_value
        
        if metadata_updates:
            goal.metadata.update(metadata_updates)
        
        goal.update_progress()
        
        if goal.is_achieved() and goal.status != ExecutionStatus.COMPLETED:
            goal.status = ExecutionStatus.COMPLETED
            self.logger.info(f"Goal achieved: {goal.name}")
            
            # Check if parent goals are now achievable
            self._check_parent_goals(goal_id)
        
        return True
    
    def _check_parent_goals(self, child_goal_id: str) -> None:
        """Check if parent goals can be updated based on child completion."""
        for parent_id, children in self.goal_hierarchy.items():
            if child_goal_id in children:
                parent_goal = self.goals.get(parent_id)
                if parent_goal:
                    # Update parent goal based on children progress
                    child_goals = [self.goals[child_id] for child_id in children if child_id in self.goals]
                    completed_children = sum(1 for child in child_goals if child.status == ExecutionStatus.COMPLETED)
                    
                    parent_goal.progress = completed_children / len(child_goals) if child_goals else 0.0
                    
                    if parent_goal.progress >= 1.0:
                        parent_goal.status = ExecutionStatus.COMPLETED
                        self.logger.info(f"Parent goal achieved: {parent_goal.name}")
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        return [
            goal for goal in self.goals.values()
            if goal.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]
        ]
    
    def get_priority_goals(self, limit: int = 5) -> List[Goal]:
        """Get highest priority active goals."""
        active_goals = self.get_active_goals()
        
        # Sort by priority and deadline
        priority_order = {
            TaskPriority.URGENT: 5,
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        
        sorted_goals = sorted(
            active_goals,
            key=lambda g: (
                priority_order.get(g.priority, 0),
                -(g.deadline.timestamp() if g.deadline else float('inf'))
            ),
            reverse=True
        )
        
        return sorted_goals[:limit]
    
    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal."""
        if goal_id not in self.goals:
            return False
        
        # Remove from hierarchy
        for parent_id, children in self.goal_hierarchy.items():
            if goal_id in children:
                children.remove(goal_id)
        
        # Remove children if this is a parent
        if goal_id in self.goal_hierarchy:
            del self.goal_hierarchy[goal_id]
        
        # Remove goal
        del self.goals[goal_id]
        
        self.logger.info(f"Removed goal: {goal_id}")
        return True
    
    def get_goal_status_report(self) -> Dict[str, Any]:
        """Get comprehensive goal status report."""
        total_goals = len(self.goals)
        active_goals = len(self.get_active_goals())
        completed_goals = sum(1 for goal in self.goals.values() if goal.status == ExecutionStatus.COMPLETED)
        
        return {
            'agent_id': self.agent_id,
            'total_goals': total_goals,
            'active_goals': active_goals,
            'completed_goals': completed_goals,
            'completion_rate': completed_goals / total_goals if total_goals > 0 else 0.0,
            'priority_goals': [
                {
                    'id': goal.id,
                    'name': goal.name,
                    'priority': goal.priority.value,
                    'progress': goal.progress,
                    'status': goal.status.value
                }
                for goal in self.get_priority_goals()
            ],
            'timestamp': datetime.now().isoformat()
        }


class AutonomousAgent(MultiAgentAgent):
    """Autonomous agent with self-management capabilities."""
    
    def __init__(self, config: AgentConfig, registry, message_bus, 
                 role: AgentRole = AgentRole.WORKER,
                 autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED):
        super().__init__(config, registry, message_bus, role)
        
        self.autonomy_level = autonomy_level
        
        # Autonomy components
        self.health_monitor = HealthMonitor(self.id)
        self.resource_manager = ResourceManager(self.id)
        self.adaptive_learner = AdaptiveLearner(self.id)
        self.goal_manager = GoalManager(self.id)
        
        # Autonomy state
        self.is_autonomous_mode = autonomy_level in [AutonomyLevel.AUTONOMOUS, AutonomyLevel.SELF_GOVERNING]
        self.last_health_check = datetime.now()
        self.last_optimization = datetime.now()
        
        # Self-healing
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        self.logger.info(f"Autonomous agent initialized with autonomy level: {autonomy_level.value}")
    
    def start_autonomous_operation(self) -> None:
        """Start autonomous operation mode."""
        self.is_autonomous_mode = True
        
        # Start autonomous loops
        import threading
        
        # Health monitoring loop
        def health_loop():
            while self.is_autonomous_mode:
                self._perform_health_check()
                time.sleep(30)  # Check every 30 seconds
        
        # Resource optimization loop
        def optimization_loop():
            while self.is_autonomous_mode:
                self._perform_optimization()
                time.sleep(300)  # Optimize every 5 minutes
        
        # Goal management loop
        def goal_loop():
            while self.is_autonomous_mode:
                self._manage_goals()
                time.sleep(60)  # Check goals every minute
        
        threading.Thread(target=health_loop, daemon=True).start()
        threading.Thread(target=optimization_loop, daemon=True).start()
        threading.Thread(target=goal_loop, daemon=True).start()
        
        self.logger.info("Started autonomous operation")
    
    def stop_autonomous_operation(self) -> None:
        """Stop autonomous operation mode."""
        self.is_autonomous_mode = False
        self.logger.info("Stopped autonomous operation")
    
    def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        try:
            # Update health metrics
            self.health_monitor.update_metric('cpu_usage', self._get_cpu_usage())
            self.health_monitor.update_metric('memory_usage', self._get_memory_usage())
            self.health_monitor.update_metric('task_success_rate', self.performance_metrics['success_rate'] * 100)
            self.health_monitor.update_metric('response_time', self.performance_metrics['average_completion_time'])
            
            # Check overall health
            health_status = self.health_monitor.get_overall_health()
            
            if health_status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                self._attempt_self_healing()
            
            self.last_health_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (simulated)."""
        # Simulate CPU usage based on active tasks
        base_usage = len(self.active_tasks) * 10
        return min(100.0, base_usage + np.random.normal(0, 5))
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simulated)."""
        # Simulate memory usage
        base_usage = len(self.active_tasks) * 15 + len(self.task_history) * 0.1
        return min(100.0, base_usage + np.random.normal(0, 10))
    
    def _attempt_self_healing(self) -> None:
        """Attempt self-healing procedures."""
        if self.recovery_attempts >= self.max_recovery_attempts:
            self.logger.error("Maximum recovery attempts reached")
            return
        
        self.recovery_attempts += 1
        self.logger.info(f"Attempting self-healing (attempt {self.recovery_attempts})")
        
        # Self-healing strategies
        healing_actions = [
            self._clear_completed_tasks,
            self._optimize_resources,
            self._reduce_task_load,
            self._restart_components
        ]
        
        for action in healing_actions:
            try:
                if action():
                    self.logger.info(f"Self-healing action successful: {action.__name__}")
                    break
            except Exception as e:
                self.logger.error(f"Self-healing action failed {action.__name__}: {e}")
    
    def _clear_completed_tasks(self) -> bool:
        """Clear completed tasks from memory."""
        initial_count = len(self.task_history)
        
        # Keep only recent tasks
        self.task_history = self.task_history[-100:]
        
        cleared = initial_count - len(self.task_history)
        if cleared > 0:
            self.logger.info(f"Cleared {cleared} completed tasks")
            return True
        
        return False
    
    def _optimize_resources(self) -> bool:
        """Optimize resource usage."""
        optimization_result = self.resource_manager.optimize_resources()
        
        if optimization_result['optimizations']:
            self.logger.info(f"Applied {len(optimization_result['optimizations'])} resource optimizations")
            return True
        
        return False
    
    def _reduce_task_load(self) -> bool:
        """Reduce current task load."""
        if len(self.active_tasks) > self.config.max_concurrent_tasks // 2:
            # Pause some low-priority tasks
            low_priority_tasks = [
                task for task in self.active_tasks.values()
                if task.priority in [TaskPriority.LOW, TaskPriority.MEDIUM]
            ]
            
            for task in low_priority_tasks[:2]:  # Pause up to 2 tasks
                task.status = ExecutionStatus.PAUSED
                self.logger.info(f"Paused task for recovery: {task.name}")
            
            return len(low_priority_tasks) > 0
        
        return False
    
    def _restart_components(self) -> bool:
        """Restart internal components."""
        try:
            # Reinitialize components
            self.health_monitor = HealthMonitor(self.id)
            self.resource_manager = ResourceManager(self.id)
            
            self.logger.info("Restarted internal components")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart components: {e}")
            return False
    
    def _perform_optimization(self) -> None:
        """Perform periodic optimization."""
        try:
            # Resource optimization
            self.resource_manager.cleanup_expired_reservations()
            optimization_result = self.resource_manager.optimize_resources()
            
            # Learning optimization
            if len(self.adaptive_learner.experiences) > 100:
                progress = self.adaptive_learner.analyze_learning_progress()
                
                # Adjust exploration rate based on learning progress
                if progress['learning_improvement'] < 0:
                    self.adaptive_learner.exploration_rate = min(0.5, self.adaptive_learner.exploration_rate * 1.1)
                else:
                    self.adaptive_learner.exploration_rate = max(0.1, self.adaptive_learner.exploration_rate * 0.95)
            
            self.last_optimization = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
    
    def _manage_goals(self) -> None:
        """Manage autonomous goals."""
        try:
            # Update goal progress based on current state
            for goal in self.goal_manager.get_active_goals():
                # Update goal based on performance metrics
                if 'success_rate' in goal.metadata:
                    self.goal_manager.update_goal_progress(
                        goal.id,
                        metadata_updates={'current_success_rate': self.performance_metrics['success_rate']}
                    )
            
            # Create new goals if needed
            if len(self.goal_manager.get_active_goals()) < 3:
                self._create_autonomous_goals()
            
        except Exception as e:
            self.logger.error(f"Goal management failed: {e}")
    
    def _create_autonomous_goals(self) -> None:
        """Create autonomous goals based on current state."""
        # Example autonomous goals
        if self.performance_metrics['success_rate'] < 0.9:
            goal = Goal(
                name="Improve Success Rate",
                description="Achieve 90% task success rate",
                priority=TaskPriority.HIGH,
                target_value=0.9,
                current_value=self.performance_metrics['success_rate'],
                success_criteria={'success_rate': 0.9}
            )
            self.goal_manager.add_goal(goal)
        
        if self.performance_metrics['average_completion_time'] > 60:
            goal = Goal(
                name="Reduce Response Time",
                description="Achieve average response time under 60 seconds",
                priority=TaskPriority.MEDIUM,
                target_value=60.0,
                current_value=self.performance_metrics['average_completion_time'],
                success_criteria={'response_time': 60.0}
            )
            self.goal_manager.add_goal(goal)
    
    def execute_task(self, task_id: str) -> bool:
        """Execute task with autonomous learning."""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        start_time = time.time()
        
        # Create learning context
        context = {
            'task_type': task.task_type.value,
            'priority': task.priority.value,
            'agent_load': len(self.active_tasks),
            'health_status': self.health_monitor.get_overall_health().value
        }
        
        # Get suggested approach from learner
        available_actions = ['standard', 'optimized', 'conservative']
        suggested_action = self.adaptive_learner.suggest_action(context, available_actions)
        
        # Execute task
        success = super().execute_task(task_id)
        
        # Record learning experience
        execution_time = time.time() - start_time
        reward = 1.0 if success else -1.0
        
        # Adjust reward based on execution time and quality
        if success:
            if execution_time < 30:  # Fast execution
                reward += 0.5
            elif execution_time > 120:  # Slow execution
                reward -= 0.3
        
        experience = LearningExperience(
            context=context,
            action=suggested_action,
            outcome={'success': success, 'execution_time': execution_time},
            reward=reward,
            success=success
        )
        
        self.adaptive_learner.add_experience(experience)
        
        return success
    
    def get_autonomy_status(self) -> Dict[str, Any]:
        """Get comprehensive autonomy status."""
        return {
            'agent_id': self.id,
            'autonomy_level': self.autonomy_level.value,
            'is_autonomous_mode': self.is_autonomous_mode,
            'health_status': self.health_monitor.get_health_report(),
            'resource_status': self.resource_manager.get_resource_report(),
            'learning_progress': self.adaptive_learner.analyze_learning_progress(),
            'goal_status': self.goal_manager.get_goal_status_report(),
            'recovery_attempts': self.recovery_attempts,
            'last_health_check': self.last_health_check.isoformat(),
            'last_optimization': self.last_optimization.isoformat(),
            'timestamp': datetime.now().isoformat()
        }


# Global autonomous agent instances
_autonomous_agents: Dict[str, AutonomousAgent] = {}


def create_autonomous_agent(config: AgentConfig, registry, message_bus,
                           role: AgentRole = AgentRole.WORKER,
                           autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED) -> AutonomousAgent:
    """Create autonomous agent."""
    agent = AutonomousAgent(config, registry, message_bus, role, autonomy_level)
    _autonomous_agents[agent.id] = agent
    return agent


def get_autonomous_agent(agent_id: str) -> Optional[AutonomousAgent]:
    """Get autonomous agent by ID."""
    return _autonomous_agents.get(agent_id)


def shutdown_autonomous_agents() -> None:
    """Shutdown all autonomous agents."""
    for agent in _autonomous_agents.values():
        agent.stop_autonomous_operation()
        agent.shutdown()
    
    _autonomous_agents.clear()