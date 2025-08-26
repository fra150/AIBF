"""Health monitoring module for AI Bull Ford.

This module provides comprehensive health monitoring capabilities including:
- System health checks and status monitoring
- Service dependency tracking
- Health metrics collection and analysis
- Automated health recovery procedures
- Health dashboard and reporting
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class CheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"
    DEPENDENCY = "dependency"
    CUSTOM = "custom"


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RESTART = "restart"
    RESET = "reset"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    FAILOVER = "failover"
    ALERT_ONLY = "alert_only"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceHealth:
    """Health information for a service."""
    service_name: str
    status: HealthStatus
    last_check: datetime
    uptime: timedelta
    check_results: List[HealthCheckResult] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    last_recovery: Optional[datetime] = None


@dataclass
class HealthConfig:
    """Configuration for health monitoring."""
    service_name: str
    check_interval: float = 30.0  # seconds
    timeout: float = 10.0  # seconds
    max_failures: int = 3
    recovery_enabled: bool = True
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    tags: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    critical: bool = False


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    services: Dict[str, ServiceHealth]
    last_updated: datetime = field(default_factory=datetime.now)
    total_services: int = 0
    healthy_services: int = 0
    warning_services: int = 0
    critical_services: int = 0
    unknown_services: int = 0
    uptime: timedelta = field(default_factory=lambda: timedelta(0))


class HealthCheck(ABC):
    """Abstract base class for health checks."""
    
    def __init__(self, name: str, check_type: CheckType = CheckType.CUSTOM):
        self.name = name
        self.check_type = check_type
        self.last_run: Optional[datetime] = None
        self.failure_count = 0
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        pass
    
    def reset_failures(self) -> None:
        """Reset failure count."""
        self.failure_count = 0


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, name: str, connection_string: str):
        super().__init__(name, CheckType.DEPENDENCY)
        self.connection_string = connection_string
    
    async def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        start_time = time.time()
        
        try:
            # Simulate database check
            await asyncio.sleep(0.1)  # Simulate connection time
            
            # In real implementation, would test actual database connection
            # For now, simulate success
            status = HealthStatus.HEALTHY
            message = "Database connection successful"
            details = {
                "connection_string": self.connection_string.split('@')[-1] if '@' in self.connection_string else "localhost",
                "response_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Database connection failed: {str(e)}"
            details = {"error": str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            check_name=self.name,
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms
        )


class APIHealthCheck(HealthCheck):
    """Health check for API endpoints."""
    
    def __init__(self, name: str, endpoint_url: str, expected_status: int = 200):
        super().__init__(name, CheckType.READINESS)
        self.endpoint_url = endpoint_url
        self.expected_status = expected_status
    
    async def check(self) -> HealthCheckResult:
        """Check API endpoint health."""
        start_time = time.time()
        
        try:
            # Simulate HTTP request
            await asyncio.sleep(0.05)  # Simulate request time
            
            # In real implementation, would make actual HTTP request
            # For now, simulate success
            status_code = self.expected_status
            
            if status_code == self.expected_status:
                status = HealthStatus.HEALTHY
                message = f"API endpoint responding correctly (status: {status_code})"
            else:
                status = HealthStatus.WARNING
                message = f"API endpoint returned unexpected status: {status_code}"
            
            details = {
                "endpoint": self.endpoint_url,
                "status_code": status_code,
                "response_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"API endpoint check failed: {str(e)}"
            details = {"error": str(e), "endpoint": self.endpoint_url}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            check_name=self.name,
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms
        )


class ResourceHealthCheck(HealthCheck):
    """Health check for system resources."""
    
    def __init__(self, name: str, resource_type: str, threshold: float = 80.0):
        super().__init__(name, CheckType.LIVENESS)
        self.resource_type = resource_type
        self.threshold = threshold
    
    async def check(self) -> HealthCheckResult:
        """Check system resource usage."""
        start_time = time.time()
        
        try:
            import psutil
            
            if self.resource_type == "cpu":
                usage = psutil.cpu_percent(interval=0.1)
                unit = "%"
            elif self.resource_type == "memory":
                usage = psutil.virtual_memory().percent
                unit = "%"
            elif self.resource_type == "disk":
                usage = psutil.disk_usage('/').percent
                unit = "%"
            else:
                raise ValueError(f"Unknown resource type: {self.resource_type}")
            
            if usage < self.threshold * 0.8:
                status = HealthStatus.HEALTHY
                message = f"{self.resource_type.title()} usage is normal: {usage:.1f}{unit}"
            elif usage < self.threshold:
                status = HealthStatus.WARNING
                message = f"{self.resource_type.title()} usage is elevated: {usage:.1f}{unit}"
            else:
                status = HealthStatus.CRITICAL
                message = f"{self.resource_type.title()} usage is critical: {usage:.1f}{unit}"
            
            details = {
                "resource_type": self.resource_type,
                "usage": usage,
                "threshold": self.threshold,
                "unit": unit
            }
            
        except ImportError:
            status = HealthStatus.UNKNOWN
            message = "psutil not available for resource monitoring"
            details = {"error": "psutil not installed"}
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Resource check failed: {str(e)}"
            details = {"error": str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            check_name=self.name,
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms
        )


class CustomHealthCheck(HealthCheck):
    """Custom health check with user-defined function."""
    
    def __init__(self, name: str, check_function: Callable, check_type: CheckType = CheckType.CUSTOM):
        super().__init__(name, check_type)
        self.check_function = check_function
    
    async def check(self) -> HealthCheckResult:
        """Execute custom health check function."""
        start_time = time.time()
        
        try:
            result = await self.check_function()
            
            if isinstance(result, HealthCheckResult):
                return result
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                message = "Custom check passed" if result else "Custom check failed"
                details = {}
            elif isinstance(result, dict):
                status = HealthStatus(result.get('status', 'unknown'))
                message = result.get('message', 'Custom check completed')
                details = result.get('details', {})
            else:
                status = HealthStatus.UNKNOWN
                message = f"Custom check returned unexpected result: {result}"
                details = {"result": str(result)}
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Custom check failed: {str(e)}"
            details = {"error": str(e)}
        
        duration_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            check_name=self.name,
            status=status,
            message=message,
            details=details,
            duration_ms=duration_ms
        )


class RecoveryManager:
    """Manages automated recovery actions."""
    
    def __init__(self):
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.max_recovery_attempts = 3
        self.recovery_cooldown = timedelta(minutes=5)
    
    def register_recovery_handler(self, action: RecoveryAction, handler: Callable) -> None:
        """Register recovery action handler."""
        self.recovery_handlers[action] = handler
    
    async def execute_recovery(
        self,
        service_name: str,
        actions: List[RecoveryAction],
        service_health: ServiceHealth
    ) -> bool:
        """Execute recovery actions for a service."""
        # Check recovery cooldown
        if service_health.last_recovery:
            time_since_recovery = datetime.now() - service_health.last_recovery
            if time_since_recovery < self.recovery_cooldown:
                logging.info(f"Recovery for {service_name} in cooldown period")
                return False
        
        # Check max attempts
        if service_health.recovery_attempts >= self.max_recovery_attempts:
            logging.warning(f"Max recovery attempts reached for {service_name}")
            return False
        
        success = False
        
        for action in actions:
            if action in self.recovery_handlers:
                try:
                    logging.info(f"Executing recovery action {action.value} for {service_name}")
                    
                    result = await self.recovery_handlers[action](service_name, service_health)
                    
                    self.recovery_history.append({
                        "service": service_name,
                        "action": action.value,
                        "timestamp": datetime.now(),
                        "success": result,
                        "attempt": service_health.recovery_attempts + 1
                    })
                    
                    if result:
                        success = True
                        break
                        
                except Exception as e:
                    logging.error(f"Recovery action {action.value} failed for {service_name}: {e}")
                    
                    self.recovery_history.append({
                        "service": service_name,
                        "action": action.value,
                        "timestamp": datetime.now(),
                        "success": False,
                        "error": str(e),
                        "attempt": service_health.recovery_attempts + 1
                    })
        
        # Update recovery tracking
        service_health.recovery_attempts += 1
        service_health.last_recovery = datetime.now()
        
        return success
    
    def get_recovery_history(self, service_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recovery history."""
        if service_name:
            return [h for h in self.recovery_history if h["service"] == service_name]
        return self.recovery_history.copy()


class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.health_checks: Dict[str, List[HealthCheck]] = defaultdict(list)
        self.configs: Dict[str, HealthConfig] = {}
        self.recovery_manager = RecoveryManager()
        self.health_listeners: List[Callable] = []
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._start_time = datetime.now()
    
    def register_service(self, config: HealthConfig) -> None:
        """Register a service for health monitoring."""
        self.configs[config.service_name] = config
        
        self.services[config.service_name] = ServiceHealth(
            service_name=config.service_name,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.now(),
            uptime=timedelta(0),
            dependencies=config.dependencies
        )
    
    def add_health_check(self, service_name: str, health_check: HealthCheck) -> None:
        """Add health check for a service."""
        self.health_checks[service_name].append(health_check)
    
    def add_health_listener(self, listener: Callable) -> None:
        """Add health status change listener."""
        self.health_listeners.append(listener)
    
    def register_recovery_handler(self, action: RecoveryAction, handler: Callable) -> None:
        """Register recovery action handler."""
        self.recovery_manager.register_recovery_handler(action, handler)
    
    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._start_time = datetime.now()
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logging.info("Health monitoring started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Check all services
                for service_name in self.services.keys():
                    await self._check_service_health(service_name)
                
                # Update overall system health
                self._update_system_health()
                
                # Wait before next check cycle
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_service_health(self, service_name: str) -> None:
        """Check health of a specific service."""
        config = self.configs.get(service_name)
        if not config or not config.enabled:
            return
        
        service_health = self.services[service_name]
        
        # Check if it's time for health check
        time_since_check = (datetime.now() - service_health.last_check).total_seconds()
        if time_since_check < config.check_interval:
            return
        
        # Run all health checks for this service
        check_results = []
        overall_status = HealthStatus.HEALTHY
        
        for health_check in self.health_checks[service_name]:
            try:
                result = await asyncio.wait_for(
                    health_check.check(),
                    timeout=config.timeout
                )
                check_results.append(result)
                
                # Update overall status based on worst result
                if result.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif result.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.WARNING
                
                health_check.reset_failures()
                
            except asyncio.TimeoutError:
                result = HealthCheckResult(
                    check_name=health_check.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check timed out after {config.timeout}s",
                    error="timeout"
                )
                check_results.append(result)
                overall_status = HealthStatus.CRITICAL
                health_check.failure_count += 1
                
            except Exception as e:
                result = HealthCheckResult(
                    check_name=health_check.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    error=str(e)
                )
                check_results.append(result)
                overall_status = HealthStatus.CRITICAL
                health_check.failure_count += 1
        
        # Update service health
        previous_status = service_health.status
        service_health.status = overall_status
        service_health.last_check = datetime.now()
        service_health.check_results = check_results[-10:]  # Keep last 10 results
        service_health.uptime = datetime.now() - self._start_time
        
        # Trigger recovery if needed
        if overall_status == HealthStatus.CRITICAL and config.recovery_enabled:
            await self.recovery_manager.execute_recovery(
                service_name,
                config.recovery_actions,
                service_health
            )
        
        # Notify listeners of status change
        if previous_status != overall_status:
            await self._notify_health_listeners(service_name, previous_status, overall_status)
    
    async def _notify_health_listeners(self, service_name: str, old_status: HealthStatus, new_status: HealthStatus) -> None:
        """Notify health status change listeners."""
        for listener in self.health_listeners:
            try:
                await listener(service_name, old_status, new_status)
            except Exception as e:
                logging.error(f"Health listener failed: {e}")
    
    def _update_system_health(self) -> None:
        """Update overall system health status."""
        if not self.services:
            return
        
        status_counts = defaultdict(int)
        for service_health in self.services.values():
            status_counts[service_health.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Update system health
        self._system_health = SystemHealth(
            overall_status=overall_status,
            services=self.services.copy(),
            total_services=len(self.services),
            healthy_services=status_counts[HealthStatus.HEALTHY],
            warning_services=status_counts[HealthStatus.WARNING],
            critical_services=status_counts[HealthStatus.CRITICAL],
            unknown_services=status_counts[HealthStatus.UNKNOWN],
            uptime=datetime.now() - self._start_time
        )
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health."""
        if not hasattr(self, '_system_health'):
            self._update_system_health()
        return self._system_health
    
    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status for a specific service."""
        return self.services.get(service_name)
    
    def get_all_services_health(self) -> Dict[str, ServiceHealth]:
        """Get health status for all services."""
        return self.services.copy()
    
    async def manual_health_check(self, service_name: str) -> Optional[ServiceHealth]:
        """Manually trigger health check for a service."""
        if service_name in self.services:
            await self._check_service_health(service_name)
            return self.services[service_name]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get health monitoring statistics."""
        system_health = self.get_system_health()
        
        return {
            "running": self._running,
            "uptime_seconds": system_health.uptime.total_seconds(),
            "total_services": system_health.total_services,
            "healthy_services": system_health.healthy_services,
            "warning_services": system_health.warning_services,
            "critical_services": system_health.critical_services,
            "unknown_services": system_health.unknown_services,
            "overall_status": system_health.overall_status.value,
            "total_health_checks": sum(len(checks) for checks in self.health_checks.values()),
            "recovery_attempts": sum(s.recovery_attempts for s in self.services.values())
        }


# Global instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def initialize_health_monitoring(
    service_configs: Optional[List[HealthConfig]] = None
) -> None:
    """Initialize health monitoring with default services."""
    global _health_monitor
    _health_monitor = HealthMonitor()
    
    # Add default service configs if none provided
    if service_configs is None:
        service_configs = [
            HealthConfig(
                service_name="system",
                check_interval=30.0,
                recovery_enabled=True,
                recovery_actions=[RecoveryAction.ALERT_ONLY],
                critical=True
            )
        ]
    
    for config in service_configs:
        _health_monitor.register_service(config)
        
        # Add default health checks
        if config.service_name == "system":
            _health_monitor.add_health_check(
                "system",
                ResourceHealthCheck("cpu_check", "cpu", 80.0)
            )
            _health_monitor.add_health_check(
                "system",
                ResourceHealthCheck("memory_check", "memory", 85.0)
            )
            _health_monitor.add_health_check(
                "system",
                ResourceHealthCheck("disk_check", "disk", 90.0)
            )


async def shutdown_health_monitoring() -> None:
    """Shutdown health monitoring."""
    global _health_monitor
    if _health_monitor:
        await _health_monitor.stop()
        _health_monitor = None