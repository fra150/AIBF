"""Main orchestrator for AI Bull Ford assembly line system.

Provides high-level coordination and management of pipelines,
workflows, modules, and resources in the AI system.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union, Set
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
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import gc

from .pipeline import Pipeline, PipelineStage, PipelineContext
from .module_registry import ModuleRegistry, ModuleType, get_registry
from .workflow import WorkflowManager, WorkflowDefinition, WorkflowExecution


class OrchestratorStatus(Enum):
    """Status of the orchestrator."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    ERROR = "error"


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceUsage:
    """System resource usage information."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_gb': self.memory_used_gb,
            'memory_total_gb': self.memory_total_gb,
            'disk_percent': self.disk_percent,
            'disk_used_gb': self.disk_used_gb,
            'disk_total_gb': self.disk_total_gb,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_concurrent_pipelines: int = 5
    max_concurrent_workflows: int = 3
    resource_monitoring_interval: float = 30.0
    auto_cleanup_interval: float = 300.0
    memory_threshold_percent: float = 85.0
    cpu_threshold_percent: float = 90.0
    enable_auto_scaling: bool = True
    enable_resource_monitoring: bool = True
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_concurrent_pipelines': self.max_concurrent_pipelines,
            'max_concurrent_workflows': self.max_concurrent_workflows,
            'resource_monitoring_interval': self.resource_monitoring_interval,
            'auto_cleanup_interval': self.auto_cleanup_interval,
            'memory_threshold_percent': self.memory_threshold_percent,
            'cpu_threshold_percent': self.cpu_threshold_percent,
            'enable_auto_scaling': self.enable_auto_scaling,
            'enable_resource_monitoring': self.enable_resource_monitoring,
            'log_level': self.log_level
        }


class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self, interval: float = 30.0):
        """Initialize resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.logger = logging.getLogger("orchestrator.monitor")
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._usage_history: List[ResourceUsage] = []
        self._max_history = 100
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Started resource monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                usage = self.get_current_usage()
                self._usage_history.append(usage)
                
                # Keep history size manageable
                if len(self._usage_history) > self._max_history:
                    self._usage_history = self._usage_history[-self._max_history:]
                
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.interval)
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage.
        
        Returns:
            Current resource usage
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb
        )
    
    def get_usage_history(self, limit: Optional[int] = None) -> List[ResourceUsage]:
        """Get resource usage history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of resource usage entries
        """
        if limit:
            return self._usage_history[-limit:]
        return self._usage_history.copy()
    
    def get_average_usage(self, minutes: int = 10) -> Optional[ResourceUsage]:
        """Get average resource usage over time period.
        
        Args:
            minutes: Time period in minutes
            
        Returns:
            Average resource usage
        """
        if not self._usage_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_usage = [u for u in self._usage_history if u.timestamp >= cutoff_time]
        
        if not recent_usage:
            return None
        
        avg_cpu = sum(u.cpu_percent for u in recent_usage) / len(recent_usage)
        avg_memory = sum(u.memory_percent for u in recent_usage) / len(recent_usage)
        avg_disk = sum(u.disk_percent for u in recent_usage) / len(recent_usage)
        
        # Use latest values for absolute numbers
        latest = recent_usage[-1]
        
        return ResourceUsage(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            memory_used_gb=latest.memory_used_gb,
            memory_total_gb=latest.memory_total_gb,
            disk_percent=avg_disk,
            disk_used_gb=latest.disk_used_gb,
            disk_total_gb=latest.disk_total_gb
        )


class Orchestrator:
    """Main orchestrator for AI Bull Ford system."""
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        self.status = OrchestratorStatus.INITIALIZING
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger("orchestrator")
        
        # Core components
        self.module_registry = get_registry()
        self.workflow_manager = WorkflowManager(max_workers=self.config.max_concurrent_workflows)
        self.resource_monitor = ResourceMonitor(self.config.resource_monitoring_interval)
        
        # Execution tracking
        self._active_pipelines: Dict[str, Pipeline] = {}
        self._pipeline_futures: Dict[str, Future] = {}
        self._pipeline_executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_pipelines)
        
        # Cleanup and maintenance
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Statistics
        self._stats = {
            'pipelines_executed': 0,
            'workflows_executed': 0,
            'modules_loaded': 0,
            'errors_count': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("Orchestrator initialized")
    
    def start(self) -> None:
        """Start the orchestrator."""
        try:
            self.status = OrchestratorStatus.RUNNING
            
            # Start resource monitoring
            if self.config.enable_resource_monitoring:
                self.resource_monitor.start_monitoring()
            
            # Start cleanup thread
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
            
            # Auto-discover modules
            self._auto_discover_modules()
            
            self.status = OrchestratorStatus.READY
            self.logger.info("Orchestrator started and ready")
            
        except Exception as e:
            self.status = OrchestratorStatus.ERROR
            self.logger.error(f"Failed to start orchestrator: {e}")
            raise
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the orchestrator.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        self.logger.info("Shutting down orchestrator")
        self.status = OrchestratorStatus.SHUTTING_DOWN
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Cancel active pipelines
        for pipeline_id, future in self._pipeline_futures.items():
            if not future.done():
                future.cancel()
                self.logger.info(f"Cancelled pipeline: {pipeline_id}")
        
        # Shutdown executors
        self._pipeline_executor.shutdown(wait=True, timeout=timeout/2)
        self.workflow_manager.shutdown()
        
        # Wait for cleanup thread
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=timeout/2)
        
        self.status = OrchestratorStatus.SHUTDOWN
        self.logger.info("Orchestrator shutdown complete")
    
    def execute_pipeline(self, 
                        pipeline: Pipeline,
                        initial_data: Optional[Dict[str, Any]] = None,
                        pipeline_id: Optional[str] = None) -> str:
        """Execute a pipeline.
        
        Args:
            pipeline: Pipeline to execute
            initial_data: Initial data for pipeline
            pipeline_id: Optional pipeline ID
            
        Returns:
            Pipeline execution ID
        """
        if pipeline_id is None:
            pipeline_id = str(uuid.uuid4())
        
        # Check resource constraints
        if not self._check_resource_constraints():
            raise RuntimeError("Resource constraints exceeded")
        
        # Check concurrent pipeline limit
        if len(self._active_pipelines) >= self.config.max_concurrent_pipelines:
            raise RuntimeError("Maximum concurrent pipelines exceeded")
        
        # Store pipeline
        self._active_pipelines[pipeline_id] = pipeline
        
        # Submit for execution
        future = self._pipeline_executor.submit(self._execute_pipeline_wrapper, 
                                               pipeline, initial_data, pipeline_id)
        self._pipeline_futures[pipeline_id] = future
        
        self._stats['pipelines_executed'] += 1
        self.logger.info(f"Started pipeline execution: {pipeline_id}")
        
        return pipeline_id
    
    def execute_workflow(self, 
                        workflow_name: str,
                        context: Optional[PipelineContext] = None,
                        execution_id: Optional[str] = None) -> str:
        """Execute a workflow.
        
        Args:
            workflow_name: Name of workflow to execute
            context: Initial pipeline context
            execution_id: Optional execution ID
            
        Returns:
            Workflow execution ID
        """
        # Check resource constraints
        if not self._check_resource_constraints():
            raise RuntimeError("Resource constraints exceeded")
        
        execution = self.workflow_manager.execute_workflow(workflow_name, context, execution_id)
        
        self._stats['workflows_executed'] += 1
        self.logger.info(f"Started workflow execution: {execution.execution_id}")
        
        return execution.execution_id
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline execution status.
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            Pipeline status if found
        """
        if pipeline_id not in self._pipeline_futures:
            return None
        
        future = self._pipeline_futures[pipeline_id]
        pipeline = self._active_pipelines.get(pipeline_id)
        
        status = {
            'pipeline_id': pipeline_id,
            'pipeline_name': pipeline.name if pipeline else 'unknown',
            'is_running': not future.done(),
            'is_completed': future.done() and not future.cancelled(),
            'is_cancelled': future.cancelled(),
            'has_error': False,
            'error': None
        }
        
        if future.done():
            try:
                result = future.result()
                status['result'] = result
            except Exception as e:
                status['has_error'] = True
                status['error'] = str(e)
        
        return status
    
    def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Workflow status if found
        """
        return self.workflow_manager.get_execution_status(execution_id)
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a pipeline execution.
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            True if cancellation successful
        """
        if pipeline_id in self._pipeline_futures:
            future = self._pipeline_futures[pipeline_id]
            success = future.cancel()
            if success:
                self.logger.info(f"Cancelled pipeline: {pipeline_id}")
            return success
        return False
    
    def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a workflow execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            True if cancellation successful
        """
        return self.workflow_manager.executor.cancel_execution(execution_id)
    
    def list_active_pipelines(self) -> List[str]:
        """List active pipeline IDs.
        
        Returns:
            List of active pipeline IDs
        """
        return [pid for pid, future in self._pipeline_futures.items() if not future.done()]
    
    def list_active_workflows(self) -> List[str]:
        """List active workflow execution IDs.
        
        Returns:
            List of active execution IDs
        """
        return self.workflow_manager.executor.list_executions()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns:
            System status dictionary
        """
        current_usage = self.resource_monitor.get_current_usage()
        avg_usage = self.resource_monitor.get_average_usage(10)
        
        return {
            'orchestrator_status': self.status.value,
            'uptime': str(datetime.now() - self._stats['start_time']),
            'statistics': self._stats.copy(),
            'active_pipelines': len(self.list_active_pipelines()),
            'active_workflows': len(self.list_active_workflows()),
            'registered_modules': len(self.module_registry),
            'registered_workflows': len(self.workflow_manager.list_workflows()),
            'current_resource_usage': current_usage.to_dict() if current_usage else None,
            'average_resource_usage': avg_usage.to_dict() if avg_usage else None,
            'config': self.config.to_dict()
        }
    
    def get_module_registry(self) -> ModuleRegistry:
        """Get the module registry.
        
        Returns:
            Module registry instance
        """
        return self.module_registry
    
    def get_workflow_manager(self) -> WorkflowManager:
        """Get the workflow manager.
        
        Returns:
            Workflow manager instance
        """
        return self.workflow_manager
    
    def _execute_pipeline_wrapper(self, 
                                 pipeline: Pipeline,
                                 initial_data: Optional[Dict[str, Any]],
                                 pipeline_id: str) -> Any:
        """Wrapper for pipeline execution with error handling.
        
        Args:
            pipeline: Pipeline to execute
            initial_data: Initial data
            pipeline_id: Pipeline ID
            
        Returns:
            Pipeline execution result
        """
        try:
            self.logger.info(f"Executing pipeline {pipeline_id}: {pipeline.name}")
            result = pipeline.run(initial_data)
            self.logger.info(f"Pipeline {pipeline_id} completed successfully")
            return result
            
        except Exception as e:
            self._stats['errors_count'] += 1
            self.logger.error(f"Pipeline {pipeline_id} failed: {e}")
            raise
        
        finally:
            # Cleanup
            if pipeline_id in self._active_pipelines:
                del self._active_pipelines[pipeline_id]
    
    def _check_resource_constraints(self) -> bool:
        """Check if resource constraints allow new execution.
        
        Returns:
            True if resources are available
        """
        if not self.config.enable_resource_monitoring:
            return True
        
        try:
            usage = self.resource_monitor.get_current_usage()
            
            if usage.memory_percent > self.config.memory_threshold_percent:
                self.logger.warning(f"Memory usage too high: {usage.memory_percent}%")
                return False
            
            if usage.cpu_percent > self.config.cpu_threshold_percent:
                self.logger.warning(f"CPU usage too high: {usage.cpu_percent}%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking resource constraints: {e}")
            return True  # Allow execution if monitoring fails
    
    def _auto_discover_modules(self) -> None:
        """Auto-discover and register modules."""
        try:
            # Discover from core modules
            discovered = self.module_registry.auto_discover('src.core')
            self._stats['modules_loaded'] += discovered
            
            self.logger.info(f"Auto-discovered {discovered} modules")
            
        except Exception as e:
            self.logger.warning(f"Auto-discovery failed: {e}")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown_event.wait(self.config.auto_cleanup_interval):
            try:
                self._cleanup_completed_executions()
                self._cleanup_memory()
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_completed_executions(self) -> None:
        """Clean up completed pipeline and workflow executions."""
        # Clean up completed pipelines
        completed_pipelines = []
        for pipeline_id, future in self._pipeline_futures.items():
            if future.done():
                completed_pipelines.append(pipeline_id)
        
        for pipeline_id in completed_pipelines:
            del self._pipeline_futures[pipeline_id]
            if pipeline_id in self._active_pipelines:
                del self._active_pipelines[pipeline_id]
        
        if completed_pipelines:
            self.logger.debug(f"Cleaned up {len(completed_pipelines)} completed pipelines")
        
        # Clean up completed workflows
        cleaned_workflows = self.workflow_manager.executor.cleanup_completed()
        if cleaned_workflows > 0:
            self.logger.debug(f"Cleaned up {cleaned_workflows} completed workflows")
    
    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            self.logger.debug(f"Garbage collected {collected} objects")
    
    def pause(self) -> None:
        """Pause the orchestrator."""
        if self.status == OrchestratorStatus.RUNNING:
            self.status = OrchestratorStatus.PAUSED
            self.logger.info("Orchestrator paused")
    
    def resume(self) -> None:
        """Resume the orchestrator."""
        if self.status == OrchestratorStatus.PAUSED:
            self.status = OrchestratorStatus.RUNNING
            self.logger.info("Orchestrator resumed")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check results
        """
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        try:
            # Check orchestrator status
            health['checks']['orchestrator'] = {
                'status': self.status.value,
                'healthy': self.status in [OrchestratorStatus.READY, OrchestratorStatus.RUNNING]
            }
            
            # Check resource usage
            usage = self.resource_monitor.get_current_usage()
            health['checks']['resources'] = {
                'memory_percent': usage.memory_percent,
                'cpu_percent': usage.cpu_percent,
                'healthy': (usage.memory_percent < self.config.memory_threshold_percent and 
                           usage.cpu_percent < self.config.cpu_threshold_percent)
            }
            
            # Check module registry
            health['checks']['modules'] = {
                'registered_count': len(self.module_registry),
                'healthy': len(self.module_registry) > 0
            }
            
            # Overall health
            all_healthy = all(check.get('healthy', False) for check in health['checks'].values())
            health['status'] = 'healthy' if all_healthy else 'unhealthy'
            
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
        
        return health


# Global orchestrator instance
_global_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Optional[Orchestrator]:
    """Get the global orchestrator instance.
    
    Returns:
        Global orchestrator if initialized
    """
    return _global_orchestrator


def initialize_orchestrator(config: Optional[OrchestratorConfig] = None) -> Orchestrator:
    """Initialize the global orchestrator.
    
    Args:
        config: Orchestrator configuration
        
    Returns:
        Initialized orchestrator
    """
    global _global_orchestrator
    
    if _global_orchestrator is not None:
        raise RuntimeError("Orchestrator already initialized")
    
    _global_orchestrator = Orchestrator(config)
    return _global_orchestrator


def shutdown_orchestrator() -> None:
    """Shutdown the global orchestrator."""
    global _global_orchestrator
    
    if _global_orchestrator is not None:
        _global_orchestrator.shutdown()
        _global_orchestrator = None