"""Agent system for AI Bull Ford.

This module provides comprehensive agent capabilities including:
- Planning and task management
- Multi-agent coordination
- Autonomous operation
"""

# Planning components
from .planning import (
    PlanningStrategy, ExecutionStatus, TaskPriority, TaskType,
    Task, PlanStep, Plan, AgentConfig, Agent,
    Planner, HierarchicalPlanner, ReactivePlanner,
    TaskScheduler, ExecutionEngine,
    get_planner, initialize_planner
)

# Multi-agent components
from .multi_agent import (
    CommunicationProtocol, MessageType, AgentRole, ConsensusAlgorithm,
    Message, AgentInfo, TaskAllocation,
    MessageBus, AgentRegistry, TaskDistributor, ConsensusManager,
    MultiAgentAgent, MultiAgentSystem,
    get_multi_agent_system, initialize_multi_agent_system, shutdown_multi_agent_system
)

# Autonomy components
from .autonomy import (
    AutonomyLevel, HealthStatus, LearningMode, ResourceType,
    HealthMetric, ResourceUsage, Goal, LearningExperience,
    HealthMonitor, ResourceManager, AdaptiveLearner, GoalManager,
    AutonomousAgent,
    create_autonomous_agent, get_autonomous_agent, shutdown_autonomous_agents
)

__all__ = [
    # Planning
    'PlanningStrategy', 'ExecutionStatus', 'TaskPriority', 'TaskType',
    'Task', 'PlanStep', 'Plan', 'AgentConfig', 'Agent',
    'Planner', 'HierarchicalPlanner', 'ReactivePlanner',
    'TaskScheduler', 'ExecutionEngine',
    'get_planner', 'initialize_planner',
    
    # Multi-agent
    'CommunicationProtocol', 'MessageType', 'AgentRole', 'ConsensusAlgorithm',
    'Message', 'AgentInfo', 'TaskAllocation',
    'MessageBus', 'AgentRegistry', 'TaskDistributor', 'ConsensusManager',
    'MultiAgentAgent', 'MultiAgentSystem',
    'get_multi_agent_system', 'initialize_multi_agent_system', 'shutdown_multi_agent_system',
    
    # Autonomy
    'AutonomyLevel', 'HealthStatus', 'LearningMode', 'ResourceType',
    'HealthMetric', 'ResourceUsage', 'Goal', 'LearningExperience',
    'HealthMonitor', 'ResourceManager', 'AdaptiveLearner', 'GoalManager',
    'AutonomousAgent',
    'create_autonomous_agent', 'get_autonomous_agent', 'shutdown_autonomous_agents'
]