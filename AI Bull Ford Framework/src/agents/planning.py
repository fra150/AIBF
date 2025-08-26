"""Agent planning module for AI Bull Ford.

This module provides comprehensive planning capabilities including:
- Task decomposition and planning
- Hierarchical and reactive planning strategies
- Task scheduling and execution
- Plan monitoring and adaptation
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


class PlanningStrategy(Enum):
    """Planning strategies."""
    HIERARCHICAL = "hierarchical"
    REACTIVE = "reactive"
    HYBRID = "hybrid"
    GOAL_ORIENTED = "goal_oriented"
    UTILITY_BASED = "utility_based"


class ExecutionStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


class TaskType(Enum):
    """Types of tasks."""
    ATOMIC = "atomic"
    COMPOSITE = "composite"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"


@dataclass
class Task:
    """Represents a task to be executed."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    task_type: TaskType = TaskType.ATOMIC
    priority: TaskPriority = TaskPriority.MEDIUM
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    # Task parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    
    # Execution details
    action: Optional[Callable] = None
    subtasks: List['Task'] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    
    # Timing
    estimated_duration: Optional[timedelta] = None
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    progress: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready for execution."""
        return (
            self.status == ExecutionStatus.PENDING and
            all(dep in completed_tasks for dep in self.dependencies) and
            self._check_preconditions()
        )
    
    def _check_preconditions(self) -> bool:
        """Check if preconditions are met."""
        # Simplified precondition checking
        # In practice, this would evaluate actual conditions
        return True
    
    def start_execution(self) -> None:
        """Mark task as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.now()
    
    def complete_execution(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark task as completed or failed."""
        self.completed_at = datetime.now()
        if success:
            self.status = ExecutionStatus.COMPLETED
            self.progress = 1.0
        else:
            self.status = ExecutionStatus.FAILED
            self.error_message = error
    
    def update_progress(self, progress: float) -> None:
        """Update task progress."""
        self.progress = max(0.0, min(1.0, progress))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'parameters': self.parameters,
            'preconditions': self.preconditions,
            'postconditions': self.postconditions,
            'subtasks': [task.to_dict() for task in self.subtasks],
            'dependencies': list(self.dependencies),
            'estimated_duration': self.estimated_duration.total_seconds() if self.estimated_duration else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'tags': list(self.tags),
            'metadata': self.metadata,
            'progress': self.progress,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


@dataclass
class PlanStep:
    """Individual step in a plan."""
    id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    step_number: int = 0
    description: str = ""
    action: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    def execute(self) -> bool:
        """Execute the plan step."""
        try:
            self.status = ExecutionStatus.RUNNING
            if self.action:
                result = self.action(**self.parameters)
                self.status = ExecutionStatus.COMPLETED
                return True
            else:
                # Simulate execution
                time.sleep(0.1)
                self.status = ExecutionStatus.COMPLETED
                return True
        except Exception as e:
            self.status = ExecutionStatus.FAILED
            return False


@dataclass
class Plan:
    """Represents an execution plan."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    goal: str = ""
    strategy: PlanningStrategy = PlanningStrategy.HIERARCHICAL
    
    # Plan structure
    steps: List[PlanStep] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    
    # Execution tracking
    status: ExecutionStatus = ExecutionStatus.PENDING
    current_step: int = 0
    progress: float = 0.0
    
    # Timing
    estimated_duration: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: PlanStep) -> None:
        """Add step to plan."""
        step.step_number = len(self.steps)
        self.steps.append(step)
    
    def add_task(self, task: Task) -> None:
        """Add task to plan."""
        self.tasks.append(task)
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get next step to execute."""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def advance_step(self) -> None:
        """Advance to next step."""
        self.current_step += 1
        self.update_progress()
    
    def update_progress(self) -> None:
        """Update plan progress."""
        if self.steps:
            completed_steps = sum(1 for step in self.steps if step.status == ExecutionStatus.COMPLETED)
            self.progress = completed_steps / len(self.steps)
    
    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(step.status == ExecutionStatus.COMPLETED for step in self.steps)
    
    def has_failed(self) -> bool:
        """Check if plan has failed."""
        return any(step.status == ExecutionStatus.FAILED for step in self.steps)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str = "Agent"
    description: str = ""
    planning_strategy: PlanningStrategy = PlanningStrategy.HIERARCHICAL
    
    # Capabilities
    max_concurrent_tasks: int = 5
    max_plan_depth: int = 10
    planning_timeout: int = 30  # seconds
    
    # Behavior
    risk_tolerance: float = 0.5
    learning_rate: float = 0.1
    exploration_factor: float = 0.2
    
    # Resource limits
    memory_limit: int = 1000  # MB
    cpu_limit: float = 0.8  # percentage
    
    # Logging
    log_level: str = "INFO"
    log_actions: bool = True


class Agent:
    """Base agent class with planning capabilities."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = str(uuid4())
        self.name = config.name
        
        # State
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self.current_plan: Optional[Plan] = None
        
        # Capabilities
        self.capabilities: Set[str] = set()
        self.knowledge_base: Dict[str, Any] = {}
        
        # Performance tracking
        self.task_history: List[Task] = []
        self.performance_metrics: Dict[str, float] = {
            'success_rate': 0.0,
            'average_completion_time': 0.0,
            'efficiency_score': 0.0
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(getattr(logging, config.log_level))
    
    def add_capability(self, capability: str) -> None:
        """Add capability to agent."""
        self.capabilities.add(capability)
        self.logger.info(f"Added capability: {capability}")
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the task."""
        # Check if agent has required capabilities
        required_capabilities = task.metadata.get('required_capabilities', set())
        return required_capabilities.issubset(self.capabilities)
    
    def assign_task(self, task: Task) -> bool:
        """Assign task to agent."""
        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
            self.logger.warning(f"Cannot assign task {task.id}: at capacity")
            return False
        
        if not self.can_handle_task(task):
            self.logger.warning(f"Cannot assign task {task.id}: missing capabilities")
            return False
        
        self.active_tasks[task.id] = task
        self.logger.info(f"Assigned task {task.id}: {task.name}")
        return True
    
    def execute_task(self, task_id: str) -> bool:
        """Execute a specific task."""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        
        try:
            task.start_execution()
            
            # Execute task action if available
            if task.action:
                result = task.action(**task.parameters)
                task.complete_execution(success=True)
            else:
                # Simulate task execution
                self._simulate_task_execution(task)
                task.complete_execution(success=True)
            
            # Move to completed
            self.completed_tasks.add(task_id)
            self.task_history.append(self.active_tasks.pop(task_id))
            
            self.logger.info(f"Completed task {task_id}: {task.name}")
            self._update_performance_metrics()
            
            return True
            
        except Exception as e:
            task.complete_execution(success=False, error=str(e))
            self.logger.error(f"Failed to execute task {task_id}: {e}")
            return False
    
    def _simulate_task_execution(self, task: Task) -> None:
        """Simulate task execution."""
        # Simulate work with progress updates
        steps = 10
        for i in range(steps):
            time.sleep(0.01)  # Simulate work
            task.update_progress((i + 1) / steps)
    
    def _update_performance_metrics(self) -> None:
        """Update agent performance metrics."""
        if not self.task_history:
            return
        
        # Calculate success rate
        successful_tasks = sum(1 for task in self.task_history if task.status == ExecutionStatus.COMPLETED)
        self.performance_metrics['success_rate'] = successful_tasks / len(self.task_history)
        
        # Calculate average completion time
        completion_times = []
        for task in self.task_history:
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                completion_times.append(duration)
        
        if completion_times:
            self.performance_metrics['average_completion_time'] = sum(completion_times) / len(completion_times)
        
        # Calculate efficiency score (simplified)
        self.performance_metrics['efficiency_score'] = (
            self.performance_metrics['success_rate'] * 0.7 +
            (1.0 / max(1.0, self.performance_metrics['average_completion_time'] / 60)) * 0.3
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            'id': self.id,
            'name': self.name,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'capabilities': list(self.capabilities),
            'performance_metrics': self.performance_metrics,
            'current_plan': self.current_plan.id if self.current_plan else None
        }


class Planner(ABC):
    """Abstract base class for planners."""
    
    @abstractmethod
    def create_plan(self, goal: str, constraints: Optional[Dict[str, Any]] = None) -> Plan:
        """Create a plan to achieve the goal."""
        pass
    
    @abstractmethod
    def decompose_task(self, task: Task) -> List[Task]:
        """Decompose a complex task into subtasks."""
        pass
    
    @abstractmethod
    def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize an existing plan."""
        pass


class HierarchicalPlanner(Planner):
    """Hierarchical task network planner."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Planning knowledge
        self.task_templates: Dict[str, Dict[str, Any]] = {}
        self.decomposition_rules: Dict[str, List[str]] = {}
        self.method_library: Dict[str, Callable] = {}
    
    def create_plan(self, goal: str, constraints: Optional[Dict[str, Any]] = None) -> Plan:
        """Create hierarchical plan."""
        plan = Plan(
            name=f"Plan for {goal}",
            description=f"Hierarchical plan to achieve: {goal}",
            goal=goal,
            strategy=PlanningStrategy.HIERARCHICAL
        )
        
        # Create high-level task
        main_task = Task(
            name=goal,
            description=f"Main task: {goal}",
            task_type=TaskType.COMPOSITE,
            priority=TaskPriority.HIGH
        )
        
        # Decompose into subtasks
        subtasks = self.decompose_task(main_task)
        
        # Create plan steps
        for i, subtask in enumerate(subtasks):
            step = PlanStep(
                task_id=subtask.id,
                step_number=i,
                description=subtask.description,
                action=subtask.action,
                parameters=subtask.parameters
            )
            plan.add_step(step)
            plan.add_task(subtask)
        
        self.logger.info(f"Created hierarchical plan with {len(plan.steps)} steps")
        return plan
    
    def decompose_task(self, task: Task) -> List[Task]:
        """Decompose task using hierarchical decomposition."""
        if task.task_type == TaskType.ATOMIC:
            return [task]
        
        subtasks = []
        
        # Check if we have decomposition rules for this task
        task_key = task.name.lower().replace(' ', '_')
        if task_key in self.decomposition_rules:
            subtask_names = self.decomposition_rules[task_key]
            
            for i, subtask_name in enumerate(subtask_names):
                subtask = Task(
                    name=subtask_name,
                    description=f"Subtask {i+1} of {task.name}",
                    task_type=TaskType.ATOMIC,
                    priority=task.priority,
                    parameters=task.parameters.copy()
                )
                
                # Add dependency on previous subtask (sequential execution)
                if i > 0:
                    subtask.dependencies.add(subtasks[i-1].id)
                
                subtasks.append(subtask)
        else:
            # Default decomposition
            subtasks = self._default_decomposition(task)
        
        return subtasks
    
    def _default_decomposition(self, task: Task) -> List[Task]:
        """Default task decomposition strategy."""
        # Simple decomposition into planning, execution, and verification
        subtasks = []
        
        # Planning subtask
        planning_task = Task(
            name=f"Plan {task.name}",
            description=f"Plan how to execute {task.name}",
            task_type=TaskType.ATOMIC,
            priority=task.priority
        )
        subtasks.append(planning_task)
        
        # Execution subtask
        execution_task = Task(
            name=f"Execute {task.name}",
            description=f"Execute the main work for {task.name}",
            task_type=TaskType.ATOMIC,
            priority=task.priority,
            dependencies={planning_task.id}
        )
        subtasks.append(execution_task)
        
        # Verification subtask
        verification_task = Task(
            name=f"Verify {task.name}",
            description=f"Verify completion of {task.name}",
            task_type=TaskType.ATOMIC,
            priority=task.priority,
            dependencies={execution_task.id}
        )
        subtasks.append(verification_task)
        
        return subtasks
    
    def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize plan by reordering and parallelizing tasks."""
        # Simple optimization: identify parallel tasks
        optimized_plan = Plan(
            name=f"Optimized {plan.name}",
            description=f"Optimized version of {plan.description}",
            goal=plan.goal,
            strategy=plan.strategy
        )
        
        # Copy tasks and analyze dependencies
        task_graph = self._build_dependency_graph(plan.tasks)
        optimized_steps = self._optimize_execution_order(plan.steps, task_graph)
        
        for step in optimized_steps:
            optimized_plan.add_step(step)
        
        for task in plan.tasks:
            optimized_plan.add_task(task)
        
        return optimized_plan
    
    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """Build task dependency graph."""
        graph = {}
        for task in tasks:
            graph[task.id] = task.dependencies.copy()
        return graph
    
    def _optimize_execution_order(self, steps: List[PlanStep], 
                                 dependency_graph: Dict[str, Set[str]]) -> List[PlanStep]:
        """Optimize step execution order."""
        # Simple topological sort for now
        # In practice, this would be more sophisticated
        return steps.copy()
    
    def add_decomposition_rule(self, task_name: str, subtask_names: List[str]) -> None:
        """Add task decomposition rule."""
        self.decomposition_rules[task_name] = subtask_names
        self.logger.info(f"Added decomposition rule for {task_name}")


class ReactivePlanner(Planner):
    """Reactive planner that responds to environmental changes."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Reactive rules
        self.condition_action_rules: List[Tuple[Callable, Callable]] = []
        self.priority_rules: Dict[str, float] = {}
    
    def create_plan(self, goal: str, constraints: Optional[Dict[str, Any]] = None) -> Plan:
        """Create reactive plan."""
        plan = Plan(
            name=f"Reactive plan for {goal}",
            description=f"Reactive plan to achieve: {goal}",
            goal=goal,
            strategy=PlanningStrategy.REACTIVE
        )
        
        # Create initial reactive step
        initial_step = PlanStep(
            description=f"Reactive response to achieve {goal}",
            action=self._reactive_action,
            parameters={'goal': goal, 'constraints': constraints or {}}
        )
        
        plan.add_step(initial_step)
        
        self.logger.info(f"Created reactive plan for goal: {goal}")
        return plan
    
    def decompose_task(self, task: Task) -> List[Task]:
        """Reactive task decomposition."""
        # Reactive decomposition based on current conditions
        subtasks = []
        
        # Check reactive rules
        for condition, action in self.condition_action_rules:
            if condition(task):
                subtask = Task(
                    name=f"Reactive response to {task.name}",
                    description=f"Reactive subtask for {task.name}",
                    task_type=TaskType.ATOMIC,
                    priority=self._calculate_reactive_priority(task),
                    action=action
                )
                subtasks.append(subtask)
        
        # If no reactive rules match, create default subtask
        if not subtasks:
            subtasks.append(task)
        
        return subtasks
    
    def _reactive_action(self, goal: str, constraints: Dict[str, Any]) -> bool:
        """Default reactive action."""
        self.logger.info(f"Executing reactive action for goal: {goal}")
        # Simulate reactive behavior
        time.sleep(0.1)
        return True
    
    def _calculate_reactive_priority(self, task: Task) -> TaskPriority:
        """Calculate priority based on reactive rules."""
        base_priority = task.priority
        
        # Apply priority rules
        priority_score = 0.0
        for rule_name, weight in self.priority_rules.items():
            if rule_name in task.tags:
                priority_score += weight
        
        # Convert score to priority
        if priority_score > 0.8:
            return TaskPriority.URGENT
        elif priority_score > 0.6:
            return TaskPriority.HIGH
        elif priority_score > 0.4:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW
    
    def optimize_plan(self, plan: Plan) -> Plan:
        """Optimize reactive plan."""
        # Reactive plans are optimized by adjusting to current conditions
        return plan
    
    def add_reactive_rule(self, condition: Callable, action: Callable) -> None:
        """Add condition-action rule."""
        self.condition_action_rules.append((condition, action))
        self.logger.info("Added reactive rule")


class TaskScheduler:
    """Schedules and prioritizes tasks for execution."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Task queues by priority
        self.task_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        
        # Scheduling state
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        
    def schedule_task(self, task: Task) -> None:
        """Schedule task for execution."""
        self.task_queues[task.priority].append(task)
        self.logger.debug(f"Scheduled task {task.id} with priority {task.priority.value}")
    
    def get_next_task(self) -> Optional[Task]:
        """Get next task to execute based on priority and dependencies."""
        # Check queues in priority order
        for priority in [TaskPriority.URGENT, TaskPriority.CRITICAL, 
                        TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]:
            queue = self.task_queues[priority]
            
            # Find ready task in queue
            for _ in range(len(queue)):
                task = queue.popleft()
                
                if task.is_ready(self.completed_tasks):
                    return task
                else:
                    # Put back at end of queue
                    queue.append(task)
        
        return None
    
    def mark_task_running(self, task: Task) -> None:
        """Mark task as running."""
        self.running_tasks[task.id] = task
        task.start_execution()
    
    def mark_task_completed(self, task_id: str, success: bool = True) -> None:
        """Mark task as completed."""
        if task_id in self.running_tasks:
            task = self.running_tasks.pop(task_id)
            task.complete_execution(success=success)
            
            if success:
                self.completed_tasks.add(task_id)
            
            self.logger.info(f"Task {task_id} completed with success={success}")
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get status of task queues."""
        return {
            priority.value: len(queue) 
            for priority, queue in self.task_queues.items()
        }


class ExecutionEngine:
    """Executes plans and manages task execution."""
    
    def __init__(self, agent: Agent, scheduler: TaskScheduler):
        self.agent = agent
        self.scheduler = scheduler
        self.logger = logging.getLogger(__name__)
        
        # Execution state
        self.current_plan: Optional[Plan] = None
        self.execution_thread = None
        self.is_running = False
    
    def execute_plan(self, plan: Plan) -> bool:
        """Execute a plan."""
        self.current_plan = plan
        plan.status = ExecutionStatus.RUNNING
        plan.started_at = datetime.now()
        
        try:
            # Schedule all tasks from the plan
            for task in plan.tasks:
                self.scheduler.schedule_task(task)
            
            # Execute plan steps
            for step in plan.steps:
                success = self._execute_step(step)
                if not success:
                    plan.status = ExecutionStatus.FAILED
                    return False
                
                plan.advance_step()
            
            plan.status = ExecutionStatus.COMPLETED
            plan.completed_at = datetime.now()
            
            self.logger.info(f"Successfully executed plan: {plan.name}")
            return True
            
        except Exception as e:
            plan.status = ExecutionStatus.FAILED
            plan.completed_at = datetime.now()
            self.logger.error(f"Failed to execute plan {plan.name}: {e}")
            return False
    
    def _execute_step(self, step: PlanStep) -> bool:
        """Execute a single plan step."""
        self.logger.debug(f"Executing step: {step.description}")
        
        try:
            success = step.execute()
            if success:
                self.logger.debug(f"Step completed successfully: {step.description}")
            else:
                self.logger.warning(f"Step failed: {step.description}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing step {step.description}: {e}")
            return False
    
    def start_continuous_execution(self) -> None:
        """Start continuous task execution."""
        self.is_running = True
        
        while self.is_running:
            # Get next task from scheduler
            task = self.scheduler.get_next_task()
            
            if task:
                # Execute task
                self.scheduler.mark_task_running(task)
                success = self.agent.execute_task(task.id)
                self.scheduler.mark_task_completed(task.id, success)
            else:
                # No tasks available, wait
                time.sleep(0.1)
    
    def stop_execution(self) -> None:
        """Stop continuous execution."""
        self.is_running = False
        self.logger.info("Stopped continuous execution")


# Global planner instances
_planner: Optional[Planner] = None


def get_planner(strategy: PlanningStrategy = PlanningStrategy.HIERARCHICAL, 
               config: Optional[AgentConfig] = None) -> Planner:
    """Get or create global planner instance."""
    global _planner
    
    if _planner is None or config is not None:
        agent_config = config or AgentConfig()
        
        if strategy == PlanningStrategy.HIERARCHICAL:
            _planner = HierarchicalPlanner(agent_config)
        elif strategy == PlanningStrategy.REACTIVE:
            _planner = ReactivePlanner(agent_config)
        else:
            _planner = HierarchicalPlanner(agent_config)  # Default
    
    return _planner


def initialize_planner(strategy: PlanningStrategy, config: AgentConfig) -> Planner:
    """Initialize global planner with specific strategy and config."""
    global _planner
    
    if strategy == PlanningStrategy.HIERARCHICAL:
        _planner = HierarchicalPlanner(config)
    elif strategy == PlanningStrategy.REACTIVE:
        _planner = ReactivePlanner(config)
    else:
        _planner = HierarchicalPlanner(config)  # Default
    
    return _planner