"""Resource monitoring module for AI Bull Ford.

This module provides comprehensive resource monitoring capabilities including:
- System resource tracking (CPU, memory, disk, network)
- Application resource usage monitoring
- Resource optimization and recommendations
- Resource allocation and management
- Resource usage forecasting and planning
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    PROCESS = "process"
    THREAD = "thread"
    FILE_HANDLE = "file_handle"
    SOCKET = "socket"


class ResourceStatus(Enum):
    """Resource status levels."""
    OPTIMAL = "optimal"
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"
    UNKNOWN = "unknown"


class OptimizationAction(Enum):
    """Resource optimization actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLEANUP = "cleanup"
    REDISTRIBUTE = "redistribute"
    CACHE_CLEAR = "cache_clear"
    PROCESS_RESTART = "process_restart"
    MEMORY_COMPACT = "memory_compact"
    DISK_CLEANUP = "disk_cleanup"
    NETWORK_THROTTLE = "network_throttle"


@dataclass
class ResourceMetric:
    """Resource usage metric."""
    resource_type: ResourceType
    name: str
    value: Union[int, float]
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Resource usage information."""
    resource_type: ResourceType
    total: float
    used: float
    available: float
    percentage: float
    status: ResourceStatus
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    trends: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceLimit:
    """Resource usage limits."""
    resource_type: ResourceType
    soft_limit: float
    hard_limit: float
    warning_threshold: float = 80.0
    critical_threshold: float = 95.0
    unit: str = ""
    enforcement_enabled: bool = True


@dataclass
class ResourceAlert:
    """Resource usage alert."""
    id: str
    resource_type: ResourceType
    resource_name: str
    status: ResourceStatus
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Resource optimization recommendation."""
    resource_type: ResourceType
    action: OptimizationAction
    description: str
    priority: str  # low, medium, high, critical
    estimated_impact: float  # percentage improvement
    implementation_effort: str  # low, medium, high
    automated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceConfig:
    """Configuration for resource monitoring."""
    resource_type: ResourceType
    collection_interval: float = 5.0  # seconds
    retention_period: timedelta = field(default_factory=lambda: timedelta(hours=24))
    limits: Optional[ResourceLimit] = None
    optimization_enabled: bool = True
    alerting_enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    custom_collector: Optional[Callable] = None


class ResourceCollector(ABC):
    """Abstract base class for resource collectors."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.last_collection: Optional[datetime] = None
    
    @abstractmethod
    async def collect(self) -> List[ResourceMetric]:
        """Collect resource metrics."""
        pass
    
    def should_collect(self) -> bool:
        """Check if it's time to collect metrics."""
        if not self.config.enabled:
            return False
        
        if self.last_collection is None:
            return True
        
        elapsed = (datetime.now() - self.last_collection).total_seconds()
        return elapsed >= self.config.collection_interval


class SystemResourceCollector(ResourceCollector):
    """Collector for system resources."""
    
    async def collect(self) -> List[ResourceMetric]:
        """Collect system resource metrics."""
        metrics = []
        
        try:
            import psutil
            
            if self.config.resource_type == ResourceType.CPU:
                metrics.extend(await self._collect_cpu_metrics(psutil))
            elif self.config.resource_type == ResourceType.MEMORY:
                metrics.extend(await self._collect_memory_metrics(psutil))
            elif self.config.resource_type == ResourceType.DISK:
                metrics.extend(await self._collect_disk_metrics(psutil))
            elif self.config.resource_type == ResourceType.NETWORK:
                metrics.extend(await self._collect_network_metrics(psutil))
            elif self.config.resource_type == ResourceType.PROCESS:
                metrics.extend(await self._collect_process_metrics(psutil))
            
            self.last_collection = datetime.now()
            
        except ImportError:
            logging.warning("psutil not available for system resource collection")
        except Exception as e:
            logging.error(f"Failed to collect {self.config.resource_type.value} metrics: {e}")
        
        return metrics
    
    async def _collect_cpu_metrics(self, psutil) -> List[ResourceMetric]:
        """Collect CPU metrics."""
        metrics = []
        
        # Overall CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(ResourceMetric(
            resource_type=ResourceType.CPU,
            name="cpu_usage_percent",
            value=cpu_percent,
            unit="%",
            tags=self.config.tags
        ))
        
        # Per-core CPU usage
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        for i, usage in enumerate(cpu_per_core):
            metrics.append(ResourceMetric(
                resource_type=ResourceType.CPU,
                name=f"cpu_core_{i}_usage_percent",
                value=usage,
                unit="%",
                tags={**self.config.tags, "core": str(i)}
            ))
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics.append(ResourceMetric(
                    resource_type=ResourceType.CPU,
                    name="cpu_frequency_mhz",
                    value=cpu_freq.current,
                    unit="MHz",
                    tags=self.config.tags,
                    metadata={"min": cpu_freq.min, "max": cpu_freq.max}
                ))
        except Exception:
            pass  # CPU frequency not available on all systems
        
        # Load average (Unix-like systems)
        try:
            load_avg = psutil.getloadavg()
            for i, load in enumerate(load_avg):
                period = ["1min", "5min", "15min"][i]
                metrics.append(ResourceMetric(
                    resource_type=ResourceType.CPU,
                    name=f"load_average_{period}",
                    value=load,
                    unit="",
                    tags={**self.config.tags, "period": period}
                ))
        except Exception:
            pass  # Load average not available on Windows
        
        return metrics
    
    async def _collect_memory_metrics(self, psutil) -> List[ResourceMetric]:
        """Collect memory metrics."""
        metrics = []
        
        # Virtual memory
        vmem = psutil.virtual_memory()
        metrics.extend([
            ResourceMetric(
                resource_type=ResourceType.MEMORY,
                name="memory_total_bytes",
                value=vmem.total,
                unit="bytes",
                tags=self.config.tags
            ),
            ResourceMetric(
                resource_type=ResourceType.MEMORY,
                name="memory_used_bytes",
                value=vmem.used,
                unit="bytes",
                tags=self.config.tags
            ),
            ResourceMetric(
                resource_type=ResourceType.MEMORY,
                name="memory_available_bytes",
                value=vmem.available,
                unit="bytes",
                tags=self.config.tags
            ),
            ResourceMetric(
                resource_type=ResourceType.MEMORY,
                name="memory_usage_percent",
                value=vmem.percent,
                unit="%",
                tags=self.config.tags
            )
        ])
        
        # Swap memory
        swap = psutil.swap_memory()
        metrics.extend([
            ResourceMetric(
                resource_type=ResourceType.MEMORY,
                name="swap_total_bytes",
                value=swap.total,
                unit="bytes",
                tags={**self.config.tags, "type": "swap"}
            ),
            ResourceMetric(
                resource_type=ResourceType.MEMORY,
                name="swap_used_bytes",
                value=swap.used,
                unit="bytes",
                tags={**self.config.tags, "type": "swap"}
            ),
            ResourceMetric(
                resource_type=ResourceType.MEMORY,
                name="swap_usage_percent",
                value=swap.percent,
                unit="%",
                tags={**self.config.tags, "type": "swap"}
            )
        ])
        
        return metrics
    
    async def _collect_disk_metrics(self, psutil) -> List[ResourceMetric]:
        """Collect disk metrics."""
        metrics = []
        
        # Disk usage for all mounted filesystems
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                device_tags = {**self.config.tags, "device": partition.device, "mountpoint": partition.mountpoint}
                
                metrics.extend([
                    ResourceMetric(
                        resource_type=ResourceType.DISK,
                        name="disk_total_bytes",
                        value=usage.total,
                        unit="bytes",
                        tags=device_tags
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.DISK,
                        name="disk_used_bytes",
                        value=usage.used,
                        unit="bytes",
                        tags=device_tags
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.DISK,
                        name="disk_free_bytes",
                        value=usage.free,
                        unit="bytes",
                        tags=device_tags
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.DISK,
                        name="disk_usage_percent",
                        value=(usage.used / usage.total) * 100,
                        unit="%",
                        tags=device_tags
                    )
                ])
            except Exception as e:
                logging.warning(f"Failed to collect disk metrics for {partition.device}: {e}")
        
        # Disk I/O statistics
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.extend([
                    ResourceMetric(
                        resource_type=ResourceType.DISK,
                        name="disk_read_bytes",
                        value=disk_io.read_bytes,
                        unit="bytes",
                        tags={**self.config.tags, "operation": "read"}
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.DISK,
                        name="disk_write_bytes",
                        value=disk_io.write_bytes,
                        unit="bytes",
                        tags={**self.config.tags, "operation": "write"}
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.DISK,
                        name="disk_read_count",
                        value=disk_io.read_count,
                        unit="operations",
                        tags={**self.config.tags, "operation": "read"}
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.DISK,
                        name="disk_write_count",
                        value=disk_io.write_count,
                        unit="operations",
                        tags={**self.config.tags, "operation": "write"}
                    )
                ])
        except Exception as e:
            logging.warning(f"Failed to collect disk I/O metrics: {e}")
        
        return metrics
    
    async def _collect_network_metrics(self, psutil) -> List[ResourceMetric]:
        """Collect network metrics."""
        metrics = []
        
        try:
            # Network I/O statistics
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.extend([
                    ResourceMetric(
                        resource_type=ResourceType.NETWORK,
                        name="network_bytes_sent",
                        value=net_io.bytes_sent,
                        unit="bytes",
                        tags={**self.config.tags, "direction": "sent"}
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.NETWORK,
                        name="network_bytes_recv",
                        value=net_io.bytes_recv,
                        unit="bytes",
                        tags={**self.config.tags, "direction": "received"}
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.NETWORK,
                        name="network_packets_sent",
                        value=net_io.packets_sent,
                        unit="packets",
                        tags={**self.config.tags, "direction": "sent"}
                    ),
                    ResourceMetric(
                        resource_type=ResourceType.NETWORK,
                        name="network_packets_recv",
                        value=net_io.packets_recv,
                        unit="packets",
                        tags={**self.config.tags, "direction": "received"}
                    )
                ])
            
            # Network connections
            connections = psutil.net_connections()
            connection_counts = defaultdict(int)
            for conn in connections:
                connection_counts[conn.status] += 1
            
            for status, count in connection_counts.items():
                metrics.append(ResourceMetric(
                    resource_type=ResourceType.NETWORK,
                    name="network_connections",
                    value=count,
                    unit="connections",
                    tags={**self.config.tags, "status": status}
                ))
            
        except Exception as e:
            logging.warning(f"Failed to collect network metrics: {e}")
        
        return metrics
    
    async def _collect_process_metrics(self, psutil) -> List[ResourceMetric]:
        """Collect process metrics."""
        metrics = []
        
        try:
            # Process count
            process_count = len(psutil.pids())
            metrics.append(ResourceMetric(
                resource_type=ResourceType.PROCESS,
                name="process_count",
                value=process_count,
                unit="processes",
                tags=self.config.tags
            ))
            
            # Current process metrics
            current_process = psutil.Process()
            
            # Memory usage
            memory_info = current_process.memory_info()
            metrics.extend([
                ResourceMetric(
                    resource_type=ResourceType.PROCESS,
                    name="process_memory_rss",
                    value=memory_info.rss,
                    unit="bytes",
                    tags={**self.config.tags, "type": "rss"}
                ),
                ResourceMetric(
                    resource_type=ResourceType.PROCESS,
                    name="process_memory_vms",
                    value=memory_info.vms,
                    unit="bytes",
                    tags={**self.config.tags, "type": "vms"}
                )
            ])
            
            # CPU usage
            cpu_percent = current_process.cpu_percent()
            metrics.append(ResourceMetric(
                resource_type=ResourceType.PROCESS,
                name="process_cpu_percent",
                value=cpu_percent,
                unit="%",
                tags=self.config.tags
            ))
            
            # Thread count
            thread_count = current_process.num_threads()
            metrics.append(ResourceMetric(
                resource_type=ResourceType.PROCESS,
                name="process_thread_count",
                value=thread_count,
                unit="threads",
                tags=self.config.tags
            ))
            
            # File descriptors (Unix-like systems)
            try:
                fd_count = current_process.num_fds()
                metrics.append(ResourceMetric(
                    resource_type=ResourceType.PROCESS,
                    name="process_file_descriptors",
                    value=fd_count,
                    unit="descriptors",
                    tags=self.config.tags
                ))
            except Exception:
                pass  # Not available on Windows
            
        except Exception as e:
            logging.warning(f"Failed to collect process metrics: {e}")
        
        return metrics


class ResourceAnalyzer:
    """Analyzes resource usage and provides insights."""
    
    def __init__(self):
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def add_metrics(self, metrics: List[ResourceMetric]) -> None:
        """Add metrics for analysis."""
        for metric in metrics:
            key = f"{metric.resource_type.value}_{metric.name}"
            self.metric_history[key].append(metric)
    
    def analyze_usage(self, resource_type: ResourceType, time_window: Optional[timedelta] = None) -> Dict[str, ResourceUsage]:
        """Analyze resource usage."""
        usage_info = {}
        
        # Filter metrics by resource type and time window
        cutoff_time = datetime.now() - time_window if time_window else None
        
        for key, metrics in self.metric_history.items():
            if not key.startswith(resource_type.value):
                continue
            
            if not metrics:
                continue
            
            # Filter by time window
            if cutoff_time:
                filtered_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            else:
                filtered_metrics = list(metrics)
            
            if not filtered_metrics:
                continue
            
            # Calculate usage statistics
            values = [m.value for m in filtered_metrics]
            latest_metric = filtered_metrics[-1]
            
            # Determine status based on value
            status = self._determine_status(latest_metric.value, resource_type)
            
            # Calculate trends
            trends = self._calculate_trends(values)
            
            usage_info[latest_metric.name] = ResourceUsage(
                resource_type=resource_type,
                total=100.0 if "percent" in latest_metric.name else max(values),
                used=latest_metric.value,
                available=100.0 - latest_metric.value if "percent" in latest_metric.name else 0,
                percentage=latest_metric.value if "percent" in latest_metric.name else 0,
                status=status,
                timestamp=latest_metric.timestamp,
                details={
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values)
                },
                trends=trends
            )
        
        return usage_info
    
    def _determine_status(self, value: float, resource_type: ResourceType) -> ResourceStatus:
        """Determine resource status based on value."""
        # Default thresholds (can be customized)
        if resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.DISK]:
            if value >= 95:
                return ResourceStatus.CRITICAL
            elif value >= 80:
                return ResourceStatus.WARNING
            elif value >= 60:
                return ResourceStatus.NORMAL
            else:
                return ResourceStatus.OPTIMAL
        else:
            return ResourceStatus.NORMAL
    
    def _calculate_trends(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend information."""
        if len(values) < 2:
            return {}
        
        # Calculate simple trend (slope)
        x_values = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        # Calculate volatility (standard deviation)
        mean_value = sum_y / n
        volatility = (sum((v - mean_value) ** 2 for v in values) / n) ** 0.5
        
        return {
            "slope": slope,
            "volatility": volatility,
            "change_rate": (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
        }
    
    def generate_recommendations(self, usage_info: Dict[str, ResourceUsage]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        for name, usage in usage_info.items():
            if usage.status == ResourceStatus.CRITICAL:
                if usage.resource_type == ResourceType.CPU:
                    recommendations.append(OptimizationRecommendation(
                        resource_type=usage.resource_type,
                        action=OptimizationAction.SCALE_UP,
                        description=f"CPU usage is critical ({usage.percentage:.1f}%). Consider scaling up or optimizing CPU-intensive operations.",
                        priority="critical",
                        estimated_impact=30.0,
                        implementation_effort="medium"
                    ))
                elif usage.resource_type == ResourceType.MEMORY:
                    recommendations.append(OptimizationRecommendation(
                        resource_type=usage.resource_type,
                        action=OptimizationAction.MEMORY_COMPACT,
                        description=f"Memory usage is critical ({usage.percentage:.1f}%). Consider memory cleanup or scaling up.",
                        priority="critical",
                        estimated_impact=25.0,
                        implementation_effort="low",
                        automated=True
                    ))
                elif usage.resource_type == ResourceType.DISK:
                    recommendations.append(OptimizationRecommendation(
                        resource_type=usage.resource_type,
                        action=OptimizationAction.DISK_CLEANUP,
                        description=f"Disk usage is critical ({usage.percentage:.1f}%). Consider disk cleanup or expansion.",
                        priority="critical",
                        estimated_impact=20.0,
                        implementation_effort="low",
                        automated=True
                    ))
            
            elif usage.status == ResourceStatus.WARNING:
                if usage.trends.get("slope", 0) > 0.1:  # Increasing trend
                    recommendations.append(OptimizationRecommendation(
                        resource_type=usage.resource_type,
                        action=OptimizationAction.SCALE_UP,
                        description=f"{usage.resource_type.value.title()} usage is increasing and approaching limits. Consider proactive scaling.",
                        priority="medium",
                        estimated_impact=15.0,
                        implementation_effort="medium"
                    ))
        
        return recommendations


class ResourceMonitor:
    """Main resource monitoring system."""
    
    def __init__(self):
        self.collectors: Dict[ResourceType, ResourceCollector] = {}
        self.configs: Dict[ResourceType, ResourceConfig] = {}
        self.analyzer = ResourceAnalyzer()
        self.alerts: List[ResourceAlert] = []
        self.alert_handlers: List[Callable] = []
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._next_alert_id = 1
    
    def add_resource_config(self, config: ResourceConfig) -> None:
        """Add resource monitoring configuration."""
        self.configs[config.resource_type] = config
        
        # Create appropriate collector
        if config.custom_collector:
            collector = config.custom_collector(config)
        else:
            collector = SystemResourceCollector(config)
        
        self.collectors[config.resource_type] = collector
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add resource alert handler."""
        self.alert_handlers.append(handler)
    
    async def start(self) -> None:
        """Start resource monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logging.info("Resource monitoring started")
    
    async def stop(self) -> None:
        """Stop resource monitoring."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Resource monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics from all collectors
                all_metrics = []
                for collector in self.collectors.values():
                    if collector.should_collect():
                        metrics = await collector.collect()
                        all_metrics.extend(metrics)
                
                # Add metrics to analyzer
                if all_metrics:
                    self.analyzer.add_metrics(all_metrics)
                    
                    # Check for alerts
                    await self._check_alerts(all_metrics)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_alerts(self, metrics: List[ResourceMetric]) -> None:
        """Check for resource alerts."""
        for metric in metrics:
            config = self.configs.get(metric.resource_type)
            if not config or not config.alerting_enabled or not config.limits:
                continue
            
            limits = config.limits
            status = ResourceStatus.NORMAL
            
            # Check thresholds
            if metric.value >= limits.critical_threshold:
                status = ResourceStatus.CRITICAL
            elif metric.value >= limits.warning_threshold:
                status = ResourceStatus.WARNING
            
            if status in [ResourceStatus.WARNING, ResourceStatus.CRITICAL]:
                # Create alert
                alert = ResourceAlert(
                    id=f"resource_alert_{self._next_alert_id}",
                    resource_type=metric.resource_type,
                    resource_name=metric.name,
                    status=status,
                    message=f"{metric.resource_type.value.title()} {metric.name} is {status.value}: {metric.value}{metric.unit}",
                    current_value=metric.value,
                    threshold=limits.critical_threshold if status == ResourceStatus.CRITICAL else limits.warning_threshold,
                    metadata={"metric": metric, "limits": limits}
                )
                
                self._next_alert_id += 1
                self.alerts.append(alert)
                
                # Trigger alert handlers
                for handler in self.alert_handlers:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logging.error(f"Resource alert handler failed: {e}")
    
    def get_resource_usage(
        self,
        resource_type: ResourceType,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, ResourceUsage]:
        """Get resource usage information."""
        return self.analyzer.analyze_usage(resource_type, time_window)
    
    def get_all_resource_usage(self, time_window: Optional[timedelta] = None) -> Dict[ResourceType, Dict[str, ResourceUsage]]:
        """Get usage information for all resources."""
        all_usage = {}
        for resource_type in self.configs.keys():
            all_usage[resource_type] = self.get_resource_usage(resource_type, time_window)
        return all_usage
    
    def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Get optimization recommendations."""
        all_recommendations = []
        
        for resource_type in self.configs.keys():
            usage_info = self.get_resource_usage(resource_type, timedelta(hours=1))
            recommendations = self.analyzer.generate_recommendations(usage_info)
            all_recommendations.extend(recommendations)
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))
        
        return all_recommendations
    
    def get_active_alerts(self) -> List[ResourceAlert]:
        """Get active resource alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a resource alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource monitoring statistics."""
        return {
            "running": self._running,
            "monitored_resources": len(self.configs),
            "active_collectors": len(self.collectors),
            "active_alerts": len(self.get_active_alerts()),
            "total_alerts": len(self.alerts),
            "resource_types": [rt.value for rt in self.configs.keys()]
        }


# Global instance
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


def initialize_resource_monitoring(
    resource_configs: Optional[List[ResourceConfig]] = None
) -> None:
    """Initialize resource monitoring with default configurations."""
    global _resource_monitor
    _resource_monitor = ResourceMonitor()
    
    # Add default resource configs if none provided
    if resource_configs is None:
        resource_configs = [
            ResourceConfig(
                resource_type=ResourceType.CPU,
                collection_interval=5.0,
                limits=ResourceLimit(
                    resource_type=ResourceType.CPU,
                    soft_limit=80.0,
                    hard_limit=95.0,
                    warning_threshold=70.0,
                    critical_threshold=90.0,
                    unit="%"
                )
            ),
            ResourceConfig(
                resource_type=ResourceType.MEMORY,
                collection_interval=5.0,
                limits=ResourceLimit(
                    resource_type=ResourceType.MEMORY,
                    soft_limit=85.0,
                    hard_limit=95.0,
                    warning_threshold=75.0,
                    critical_threshold=90.0,
                    unit="%"
                )
            ),
            ResourceConfig(
                resource_type=ResourceType.DISK,
                collection_interval=30.0,
                limits=ResourceLimit(
                    resource_type=ResourceType.DISK,
                    soft_limit=90.0,
                    hard_limit=95.0,
                    warning_threshold=80.0,
                    critical_threshold=90.0,
                    unit="%"
                )
            ),
            ResourceConfig(
                resource_type=ResourceType.NETWORK,
                collection_interval=10.0
            ),
            ResourceConfig(
                resource_type=ResourceType.PROCESS,
                collection_interval=15.0
            )
        ]
    
    for config in resource_configs:
        _resource_monitor.add_resource_config(config)


async def shutdown_resource_monitoring() -> None:
    """Shutdown resource monitoring."""
    global _resource_monitor
    if _resource_monitor:
        await _resource_monitor.stop()
        _resource_monitor = None