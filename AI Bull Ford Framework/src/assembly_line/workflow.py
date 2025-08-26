"""Workflow management system for complex AI pipelines.

Provides workflow definition, execution, and orchestration capabilities
for multi-stage AI processes with dependencies and conditional logic.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union, Set, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

from .pipeline import Pipeline, PipelineStage, PipelineContext, StageStatus


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class NodeType(Enum):
    """Types of workflow nodes."""
    PIPELINE = "pipeline"
    STAGE = "stage"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    LOOP = "loop"
    CUSTOM = "custom"


@dataclass
class WorkflowNode:
    """Node in a workflow graph."""
    id: str
    name: str
    node_type: NodeType
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Execution state
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    
    def reset(self) -> None:
        """Reset node execution state."""
        self.status = WorkflowStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.result = None
        self.retry_count = 0
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get node execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'node_type': self.node_type.value,
            'config': self.config,
            'dependencies': self.dependencies,
            'conditions': self.conditions,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error': self.error,
            'duration': str(self.duration) if self.duration else None
        }


@dataclass
class WorkflowDefinition:
    """Definition of a workflow."""
    name: str
    description: str
    version: str = "1.0.0"
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    global_config: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    max_parallel: int = 4
    
    def add_node(self, node: WorkflowNode) -> 'WorkflowDefinition':
        """Add a node to the workflow.
        
        Args:
            node: Workflow node to add
            
        Returns:
            Self for method chaining
        """
        self.nodes[node.id] = node
        return self
    
    def add_dependency(self, node_id: str, dependency_id: str) -> 'WorkflowDefinition':
        """Add a dependency between nodes.
        
        Args:
            node_id: ID of dependent node
            dependency_id: ID of dependency node
            
        Returns:
            Self for method chaining
        """
        if node_id in self.nodes:
            if dependency_id not in self.nodes[node_id].dependencies:
                self.nodes[node_id].dependencies.append(dependency_id)
        return self
    
    def validate(self) -> List[str]:
        """Validate workflow definition.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        if not self.nodes:
            errors.append("Workflow has no nodes")
            return errors
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Circular dependencies detected")
        
        # Check that all dependencies exist
        for node_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    errors.append(f"Node {node_id} depends on non-existent node {dep_id}")
        
        return errors
    
    def get_execution_order(self) -> List[List[str]]:
        """Get topological execution order of nodes.
        
        Returns:
            List of execution levels (nodes that can run in parallel)
        """
        # Kahn's algorithm for topological sorting
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        # Calculate in-degrees
        for node in self.nodes.values():
            for dep_id in node.dependencies:
                if dep_id in in_degree:
                    in_degree[dep_id] += 1
        
        # Find nodes with no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            # Current level (nodes that can run in parallel)
            current_level = queue.copy()
            execution_order.append(current_level)
            queue = []
            
            # Process current level
            for node_id in current_level:
                # Remove this node and update in-degrees
                for other_node in self.nodes.values():
                    if node_id in other_node.dependencies:
                        in_degree[other_node.id] -= 1
                        if in_degree[other_node.id] == 0:
                            queue.append(other_node.id)
        
        return execution_order
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS.
        
        Returns:
            True if circular dependencies exist
        """
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            if node_id in self.nodes:
                for dep_id in self.nodes[node_id].dependencies:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow definition to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'global_config': self.global_config,
            'timeout': self.timeout,
            'max_parallel': self.max_parallel
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowDefinition':
        """Create workflow definition from dictionary."""
        workflow = cls(
            name=data['name'],
            description=data['description'],
            version=data.get('version', '1.0.0'),
            global_config=data.get('global_config', {}),
            timeout=data.get('timeout'),
            max_parallel=data.get('max_parallel', 4)
        )
        
        # Add nodes
        for node_id, node_data in data.get('nodes', {}).items():
            node = WorkflowNode(
                id=node_data['id'],
                name=node_data['name'],
                node_type=NodeType(node_data['node_type']),
                config=node_data.get('config', {}),
                dependencies=node_data.get('dependencies', []),
                conditions=node_data.get('conditions', []),
                timeout=node_data.get('timeout'),
                max_retries=node_data.get('max_retries', 3)
            )
            workflow.add_node(node)
        
        return workflow


class WorkflowExecutor:
    """Executor for workflow instances."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize workflow executor.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger("workflow.executor")
        self._active_executions: Dict[str, 'WorkflowExecution'] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute(self, 
               workflow_def: WorkflowDefinition,
               context: Optional[PipelineContext] = None,
               execution_id: Optional[str] = None) -> 'WorkflowExecution':
        """Execute a workflow.
        
        Args:
            workflow_def: Workflow definition to execute
            context: Initial pipeline context
            execution_id: Optional execution ID
            
        Returns:
            Workflow execution instance
        """
        if execution_id is None:
            execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_def=workflow_def,
            context=context or PipelineContext({}, {}, {}, {}),
            executor=self
        )
        
        self._active_executions[execution_id] = execution
        
        # Start execution in background
        future = self._executor.submit(execution._run)
        execution._future = future
        
        return execution
    
    def get_execution(self, execution_id: str) -> Optional['WorkflowExecution']:
        """Get active workflow execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Workflow execution if found
        """
        return self._active_executions.get(execution_id)
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            True if cancellation successful
        """
        execution = self.get_execution(execution_id)
        if execution:
            return execution.cancel()
        return False
    
    def list_executions(self) -> List[str]:
        """List active execution IDs.
        
        Returns:
            List of execution IDs
        """
        return list(self._active_executions.keys())
    
    def cleanup_completed(self) -> int:
        """Clean up completed executions.
        
        Returns:
            Number of executions cleaned up
        """
        completed = []
        for execution_id, execution in self._active_executions.items():
            if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                completed.append(execution_id)
        
        for execution_id in completed:
            del self._active_executions[execution_id]
        
        return len(completed)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.
        
        Args:
            wait: Whether to wait for active executions
        """
        self._executor.shutdown(wait=wait)


class WorkflowExecution:
    """Runtime execution of a workflow."""
    
    def __init__(self, 
                 execution_id: str,
                 workflow_def: WorkflowDefinition,
                 context: PipelineContext,
                 executor: WorkflowExecutor):
        """Initialize workflow execution.
        
        Args:
            execution_id: Unique execution ID
            workflow_def: Workflow definition
            context: Pipeline context
            executor: Workflow executor
        """
        self.execution_id = execution_id
        self.workflow_def = workflow_def
        self.context = context
        self.executor = executor
        
        # Execution state
        self.status = WorkflowStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        self._future: Optional[Future] = None
        self._cancelled = False
        
        # Node execution tracking
        self.node_results: Dict[str, Any] = {}
        self.execution_order: List[List[str]] = []
        
        self.logger = logging.getLogger(f"workflow.execution.{execution_id[:8]}")
    
    def _run(self) -> None:
        """Internal method to run the workflow."""
        try:
            self.start_time = datetime.now()
            self.status = WorkflowStatus.RUNNING
            
            self.logger.info(f"Starting workflow execution: {self.workflow_def.name}")
            
            # Validate workflow
            errors = self.workflow_def.validate()
            if errors:
                raise ValueError(f"Workflow validation failed: {errors}")
            
            # Get execution order
            self.execution_order = self.workflow_def.get_execution_order()
            
            # Execute workflow levels
            for level_idx, level_nodes in enumerate(self.execution_order):
                if self._cancelled:
                    self.status = WorkflowStatus.CANCELLED
                    return
                
                self.logger.info(f"Executing level {level_idx + 1}: {level_nodes}")
                
                # Execute nodes in parallel within the level
                if len(level_nodes) == 1:
                    # Single node - execute directly
                    self._execute_node(level_nodes[0])
                else:
                    # Multiple nodes - execute in parallel
                    self._execute_level_parallel(level_nodes)
            
            self.status = WorkflowStatus.COMPLETED
            self.end_time = datetime.now()
            
            self.logger.info(f"Workflow execution completed: {self.workflow_def.name}")
            
        except Exception as e:
            self.error = str(e)
            self.status = WorkflowStatus.FAILED
            self.end_time = datetime.now()
            
            self.logger.error(f"Workflow execution failed: {e}")
    
    def _execute_level_parallel(self, node_ids: List[str]) -> None:
        """Execute a level of nodes in parallel.
        
        Args:
            node_ids: List of node IDs to execute
        """
        with ThreadPoolExecutor(max_workers=min(len(node_ids), self.workflow_def.max_parallel)) as executor:
            # Submit all nodes for execution
            futures = {executor.submit(self._execute_node, node_id): node_id 
                      for node_id in node_ids}
            
            # Wait for completion
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Node {node_id} failed: {e}")
                    # Continue with other nodes unless configured to stop
                    if self.workflow_def.global_config.get('stop_on_failure', True):
                        raise
    
    def _execute_node(self, node_id: str) -> None:
        """Execute a single workflow node.
        
        Args:
            node_id: Node ID to execute
        """
        node = self.workflow_def.nodes[node_id]
        
        try:
            node.start_time = datetime.now()
            node.status = WorkflowStatus.RUNNING
            
            self.logger.info(f"Executing node: {node.name} ({node.node_type.value})")
            
            # Check conditions
            if not self._check_conditions(node):
                node.status = WorkflowStatus.COMPLETED  # Skip but mark as completed
                node.end_time = datetime.now()
                self.logger.info(f"Node {node.name} skipped due to conditions")
                return
            
            # Execute based on node type
            if node.node_type == NodeType.PIPELINE:
                result = self._execute_pipeline_node(node)
            elif node.node_type == NodeType.STAGE:
                result = self._execute_stage_node(node)
            elif node.node_type == NodeType.CONDITION:
                result = self._execute_condition_node(node)
            elif node.node_type == NodeType.CUSTOM:
                result = self._execute_custom_node(node)
            else:
                raise ValueError(f"Unsupported node type: {node.node_type}")
            
            node.result = result
            node.status = WorkflowStatus.COMPLETED
            node.end_time = datetime.now()
            
            # Store result in context
            self.node_results[node_id] = result
            self.context.set(f"node_{node_id}_result", result)
            
            self.logger.info(f"Node {node.name} completed successfully")
            
        except Exception as e:
            node.error = str(e)
            node.status = WorkflowStatus.FAILED
            node.end_time = datetime.now()
            
            self.logger.error(f"Node {node.name} failed: {e}")
            
            # Retry logic
            if node.retry_count < node.max_retries:
                node.retry_count += 1
                self.logger.info(f"Retrying node {node.name} (attempt {node.retry_count}/{node.max_retries})")
                time.sleep(2 ** node.retry_count)  # Exponential backoff
                self._execute_node(node_id)
            else:
                raise
    
    def _check_conditions(self, node: WorkflowNode) -> bool:
        """Check if node conditions are met.
        
        Args:
            node: Node to check
            
        Returns:
            True if conditions are met
        """
        if not node.conditions:
            return True
        
        # Simple condition evaluation (can be extended)
        for condition in node.conditions:
            # Example: "node_data_loader_result.success == True"
            try:
                # This is a simplified implementation
                # In practice, you'd want a proper expression evaluator
                if not eval(condition, {}, {
                    'context': self.context,
                    'node_results': self.node_results
                }):
                    return False
            except Exception as e:
                self.logger.warning(f"Condition evaluation failed: {condition} - {e}")
                return False
        
        return True
    
    def _execute_pipeline_node(self, node: WorkflowNode) -> Any:
        """Execute a pipeline node.
        
        Args:
            node: Pipeline node
            
        Returns:
            Pipeline execution result
        """
        # Create pipeline from config
        pipeline_config = node.config.get('pipeline', {})
        pipeline = Pipeline(node.name, pipeline_config)
        
        # Add stages based on config
        stages_config = node.config.get('stages', [])
        for stage_config in stages_config:
            # This would need to be implemented based on your stage factory
            pass
        
        # Execute pipeline
        result_context = pipeline.run(self.context.data)
        
        # Update main context
        self.context.update(result_context.data)
        
        return result_context.data
    
    def _execute_stage_node(self, node: WorkflowNode) -> Any:
        """Execute a stage node.
        
        Args:
            node: Stage node
            
        Returns:
            Stage execution result
        """
        # This would create and execute a single stage
        # Implementation depends on your stage factory
        return {"stage_result": "completed"}
    
    def _execute_condition_node(self, node: WorkflowNode) -> Any:
        """Execute a condition node.
        
        Args:
            node: Condition node
            
        Returns:
            Condition evaluation result
        """
        condition = node.config.get('condition', 'True')
        try:
            result = eval(condition, {}, {
                'context': self.context,
                'node_results': self.node_results
            })
            return {'condition_result': result}
        except Exception as e:
            raise ValueError(f"Condition evaluation failed: {e}")
    
    def _execute_custom_node(self, node: WorkflowNode) -> Any:
        """Execute a custom node.
        
        Args:
            node: Custom node
            
        Returns:
            Custom execution result
        """
        # Custom node execution would be implemented based on config
        custom_function = node.config.get('function')
        if custom_function:
            # This would need proper function resolution
            pass
        
        return {"custom_result": "completed"}
    
    def cancel(self) -> bool:
        """Cancel workflow execution.
        
        Returns:
            True if cancellation successful
        """
        self._cancelled = True
        if self._future:
            return self._future.cancel()
        return True
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for workflow completion.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if completed within timeout
        """
        if self._future:
            try:
                self._future.result(timeout=timeout)
                return True
            except Exception:
                return False
        return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get workflow execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get execution status summary.
        
        Returns:
            Status summary dictionary
        """
        node_statuses = {}
        for node_id, node in self.workflow_def.nodes.items():
            node_statuses[node_id] = {
                'name': node.name,
                'status': node.status.value,
                'duration': str(node.duration) if node.duration else None,
                'error': node.error
            }
        
        return {
            'execution_id': self.execution_id,
            'workflow_name': self.workflow_def.name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': str(self.duration) if self.duration else None,
            'error': self.error,
            'nodes': node_statuses
        }


class WorkflowManager:
    """High-level workflow management interface."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize workflow manager.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.executor = WorkflowExecutor(max_workers)
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.logger = logging.getLogger("workflow.manager")
    
    def register_workflow(self, workflow_def: WorkflowDefinition) -> None:
        """Register a workflow definition.
        
        Args:
            workflow_def: Workflow definition to register
        """
        self.workflows[workflow_def.name] = workflow_def
        self.logger.info(f"Registered workflow: {workflow_def.name}")
    
    def execute_workflow(self, 
                        workflow_name: str,
                        context: Optional[PipelineContext] = None,
                        execution_id: Optional[str] = None) -> WorkflowExecution:
        """Execute a registered workflow.
        
        Args:
            workflow_name: Name of workflow to execute
            context: Initial pipeline context
            execution_id: Optional execution ID
            
        Returns:
            Workflow execution instance
        """
        if workflow_name not in self.workflows:
            raise KeyError(f"Workflow {workflow_name} not found")
        
        workflow_def = self.workflows[workflow_name]
        return self.executor.execute(workflow_def, context, execution_id)
    
    def load_workflow_from_file(self, file_path: str) -> WorkflowDefinition:
        """Load workflow definition from JSON file.
        
        Args:
            file_path: Path to workflow file
            
        Returns:
            Loaded workflow definition
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        workflow_def = WorkflowDefinition.from_dict(data)
        self.register_workflow(workflow_def)
        
        return workflow_def
    
    def save_workflow_to_file(self, workflow_name: str, file_path: str) -> None:
        """Save workflow definition to JSON file.
        
        Args:
            workflow_name: Name of workflow to save
            file_path: Path to save file
        """
        if workflow_name not in self.workflows:
            raise KeyError(f"Workflow {workflow_name} not found")
        
        workflow_def = self.workflows[workflow_name]
        with open(file_path, 'w') as f:
            json.dump(workflow_def.to_dict(), f, indent=2)
    
    def list_workflows(self) -> List[str]:
        """List registered workflows.
        
        Returns:
            List of workflow names
        """
        return list(self.workflows.keys())
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Status summary if found
        """
        execution = self.executor.get_execution(execution_id)
        if execution:
            return execution.get_status_summary()
        return None
    
    def shutdown(self) -> None:
        """Shutdown the workflow manager."""
        self.executor.shutdown()