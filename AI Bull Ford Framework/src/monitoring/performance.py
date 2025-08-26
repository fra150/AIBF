"""Performance monitoring module for AI Bull Ford.

This module provides comprehensive performance monitoring capabilities including:
- Real-time metrics collection and aggregation
- Performance alerts and notifications
- Custom metric definitions and tracking
- Historical performance analysis
- Automated performance reporting
"""

import asyncio
import logging
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Deque


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    PERCENTAGE = "percentage"


class AggregationType(Enum):
    """Types of metric aggregation."""
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    COUNT = "count"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Performance metric data."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""
    source: str = ""


@dataclass
class MetricConfig:
    """Configuration for metric collection."""
    name: str
    metric_type: MetricType
    collection_interval: float = 1.0  # seconds
    retention_period: timedelta = field(default_factory=lambda: timedelta(hours=24))
    aggregation_types: List[AggregationType] = field(default_factory=lambda: [AggregationType.AVERAGE])
    alert_thresholds: Dict[AlertSeverity, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    custom_collector: Optional[Callable] = None


@dataclass
class Alert:
    """Performance alert."""
    id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    value: Union[int, float]
    threshold: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Performance analysis report."""
    start_time: datetime
    end_time: datetime
    metrics_summary: Dict[str, Dict[str, Any]]
    alerts_summary: Dict[AlertSeverity, int]
    recommendations: List[str] = field(default_factory=list)
    trends: Dict[str, str] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.last_collection: Optional[datetime] = None
    
    @abstractmethod
    async def collect(self) -> Optional[Metric]:
        """Collect metric value."""
        pass
    
    def should_collect(self) -> bool:
        """Check if it's time to collect metric."""
        if not self.config.enabled:
            return False
        
        if self.last_collection is None:
            return True
        
        elapsed = (datetime.now() - self.last_collection).total_seconds()
        return elapsed >= self.config.collection_interval


class SystemMetricCollector(MetricCollector):
    """Collector for system metrics."""
    
    async def collect(self) -> Optional[Metric]:
        """Collect system metric."""
        try:
            import psutil
            
            if self.config.name == "cpu_usage":
                value = psutil.cpu_percent(interval=0.1)
            elif self.config.name == "memory_usage":
                value = psutil.virtual_memory().percent
            elif self.config.name == "disk_usage":
                value = psutil.disk_usage('/').percent
            elif self.config.name == "network_io":
                stats = psutil.net_io_counters()
                value = stats.bytes_sent + stats.bytes_recv
            else:
                return None
            
            self.last_collection = datetime.now()
            
            return Metric(
                name=self.config.name,
                value=value,
                metric_type=self.config.metric_type,
                tags=self.config.tags,
                source="system"
            )
            
        except ImportError:
            logging.warning("psutil not available for system metrics")
            return None
        except Exception as e:
            logging.error(f"Failed to collect system metric {self.config.name}: {e}")
            return None


class CustomMetricCollector(MetricCollector):
    """Collector for custom metrics."""
    
    async def collect(self) -> Optional[Metric]:
        """Collect custom metric using provided function."""
        if not self.config.custom_collector:
            return None
        
        try:
            value = await self.config.custom_collector()
            if value is None:
                return None
            
            self.last_collection = datetime.now()
            
            return Metric(
                name=self.config.name,
                value=value,
                metric_type=self.config.metric_type,
                tags=self.config.tags,
                source="custom"
            )
            
        except Exception as e:
            logging.error(f"Failed to collect custom metric {self.config.name}: {e}")
            return None


class MetricAggregator:
    """Aggregates metrics over time windows."""
    
    def __init__(self):
        self.metric_data: Dict[str, Deque[Metric]] = defaultdict(lambda: deque(maxlen=10000))
    
    def add_metric(self, metric: Metric) -> None:
        """Add metric to aggregation."""
        self.metric_data[metric.name].append(metric)
    
    def aggregate(
        self,
        metric_name: str,
        aggregation_type: AggregationType,
        time_window: Optional[timedelta] = None
    ) -> Optional[float]:
        """Aggregate metric values."""
        if metric_name not in self.metric_data:
            return None
        
        metrics = self.metric_data[metric_name]
        if not metrics:
            return None
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        
        try:
            if aggregation_type == AggregationType.SUM:
                return sum(values)
            elif aggregation_type == AggregationType.AVERAGE:
                return statistics.mean(values)
            elif aggregation_type == AggregationType.MIN:
                return min(values)
            elif aggregation_type == AggregationType.MAX:
                return max(values)
            elif aggregation_type == AggregationType.MEDIAN:
                return statistics.median(values)
            elif aggregation_type == AggregationType.PERCENTILE_95:
                return self._percentile(values, 95)
            elif aggregation_type == AggregationType.PERCENTILE_99:
                return self._percentile(values, 99)
            elif aggregation_type == AggregationType.COUNT:
                return len(values)
            else:
                return None
                
        except Exception as e:
            logging.error(f"Failed to aggregate metric {metric_name}: {e}")
            return None
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_metric_history(
        self,
        metric_name: str,
        time_window: Optional[timedelta] = None
    ) -> List[Metric]:
        """Get metric history."""
        if metric_name not in self.metric_data:
            return []
        
        metrics = list(self.metric_data[metric_name])
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return sorted(metrics, key=lambda m: m.timestamp)
    
    def cleanup_old_metrics(self, retention_period: timedelta) -> None:
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.now() - retention_period
        
        for metric_name in self.metric_data:
            metrics = self.metric_data[metric_name]
            # Remove old metrics
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()


class AlertManager:
    """Manages performance alerts."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self._next_alert_id = 1
    
    def add_alert_handler(self, severity: AlertSeverity, handler: Callable) -> None:
        """Add alert handler for specific severity."""
        self.alert_handlers[severity].append(handler)
    
    def check_thresholds(
        self,
        metric: Metric,
        thresholds: Dict[AlertSeverity, float]
    ) -> Optional[Alert]:
        """Check if metric exceeds alert thresholds."""
        for severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM, AlertSeverity.LOW]:
            if severity in thresholds and metric.value >= thresholds[severity]:
                return self._create_alert(metric, severity, thresholds[severity])
        
        return None
    
    def _create_alert(self, metric: Metric, severity: AlertSeverity, threshold: float) -> Alert:
        """Create new alert."""
        alert_id = f"alert_{self._next_alert_id}"
        self._next_alert_id += 1
        
        alert = Alert(
            id=alert_id,
            metric_name=metric.name,
            severity=severity,
            message=f"Metric {metric.name} exceeded {severity.value} threshold: {metric.value} >= {threshold}",
            value=metric.value,
            threshold=threshold,
            tags=metric.tags
        )
        
        self.alerts[alert_id] = alert
        return alert
    
    async def trigger_alert(self, alert: Alert) -> None:
        """Trigger alert handlers."""
        handlers = self.alert_handlers.get(alert.severity, [])
        
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = datetime.now()
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [alert for alert in self.alerts.values() if alert.severity == severity]
    
    def cleanup_old_alerts(self, retention_period: timedelta) -> None:
        """Remove old resolved alerts."""
        cutoff_time = datetime.now() - retention_period
        
        alerts_to_remove = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
        ]
        
        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self):
        self.collectors: Dict[str, MetricCollector] = {}
        self.aggregator = MetricAggregator()
        self.alert_manager = AlertManager()
        self.configs: Dict[str, MetricConfig] = {}
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
    
    def add_metric_config(self, config: MetricConfig) -> None:
        """Add metric configuration."""
        self.configs[config.name] = config
        
        # Create appropriate collector
        if config.custom_collector:
            collector = CustomMetricCollector(config)
        else:
            collector = SystemMetricCollector(config)
        
        self.collectors[config.name] = collector
    
    def add_alert_handler(self, severity: AlertSeverity, handler: Callable) -> None:
        """Add alert handler."""
        self.alert_manager.add_alert_handler(severity, handler)
    
    async def start(self) -> None:
        """Start performance monitoring."""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logging.info("Performance monitoring started")
    
    async def stop(self) -> None:
        """Stop performance monitoring."""
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Performance monitoring stopped")
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                # Collect metrics
                for collector in self.collectors.values():
                    if collector.should_collect():
                        metric = await collector.collect()
                        if metric:
                            await self._process_metric(metric)
                
                # Cleanup old data
                self._cleanup_old_data()
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logging.error(f"Error in collection loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_metric(self, metric: Metric) -> None:
        """Process collected metric."""
        # Add to aggregator
        self.aggregator.add_metric(metric)
        
        # Check for alerts
        config = self.configs.get(metric.name)
        if config and config.alert_thresholds:
            alert = self.alert_manager.check_thresholds(metric, config.alert_thresholds)
            if alert:
                await self.alert_manager.trigger_alert(alert)
    
    def _cleanup_old_data(self) -> None:
        """Cleanup old metrics and alerts."""
        # Cleanup metrics
        for config in self.configs.values():
            self.aggregator.cleanup_old_metrics(config.retention_period)
        
        # Cleanup alerts (keep for 7 days)
        self.alert_manager.cleanup_old_alerts(timedelta(days=7))
    
    def get_metric_value(
        self,
        metric_name: str,
        aggregation_type: AggregationType = AggregationType.AVERAGE,
        time_window: Optional[timedelta] = None
    ) -> Optional[float]:
        """Get aggregated metric value."""
        return self.aggregator.aggregate(metric_name, aggregation_type, time_window)
    
    def get_metric_history(
        self,
        metric_name: str,
        time_window: Optional[timedelta] = None
    ) -> List[Metric]:
        """Get metric history."""
        return self.aggregator.get_metric_history(metric_name, time_window)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self.alert_manager.get_active_alerts()
    
    def generate_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> PerformanceReport:
        """Generate performance report."""
        time_window = end_time - start_time
        
        # Collect metrics summary
        metrics_summary = {}
        for metric_name in self.configs.keys():
            summary = {}
            for agg_type in AggregationType:
                value = self.aggregator.aggregate(metric_name, agg_type, time_window)
                if value is not None:
                    summary[agg_type.value] = value
            
            if summary:
                metrics_summary[metric_name] = summary
        
        # Collect alerts summary
        alerts_summary = {severity: 0 for severity in AlertSeverity}
        for alert in self.alert_manager.alerts.values():
            if start_time <= alert.timestamp <= end_time:
                alerts_summary[alert.severity] += 1
        
        # Generate recommendations (simplified)
        recommendations = self._generate_recommendations(metrics_summary)
        
        return PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            metrics_summary=metrics_summary,
            alerts_summary=alerts_summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, metrics_summary: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check CPU usage
        if "cpu_usage" in metrics_summary:
            avg_cpu = metrics_summary["cpu_usage"].get("average", 0)
            if avg_cpu > 80:
                recommendations.append("High CPU usage detected. Consider optimizing CPU-intensive operations.")
        
        # Check memory usage
        if "memory_usage" in metrics_summary:
            avg_memory = metrics_summary["memory_usage"].get("average", 0)
            if avg_memory > 85:
                recommendations.append("High memory usage detected. Consider optimizing memory allocation.")
        
        # Check disk usage
        if "disk_usage" in metrics_summary:
            avg_disk = metrics_summary["disk_usage"].get("average", 0)
            if avg_disk > 90:
                recommendations.append("High disk usage detected. Consider cleaning up disk space.")
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "running": self._running,
            "metrics_configured": len(self.configs),
            "active_collectors": len(self.collectors),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "total_alerts": len(self.alert_manager.alerts),
            "metric_names": list(self.configs.keys())
        }


# Global instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def initialize_performance_monitoring(
    metric_configs: Optional[List[MetricConfig]] = None
) -> None:
    """Initialize performance monitoring with default metrics."""
    global _performance_monitor
    _performance_monitor = PerformanceMonitor()
    
    # Add default system metrics if no configs provided
    if metric_configs is None:
        metric_configs = [
            MetricConfig(
                name="cpu_usage",
                metric_type=MetricType.GAUGE,
                collection_interval=5.0,
                alert_thresholds={
                    AlertSeverity.HIGH: 80.0,
                    AlertSeverity.CRITICAL: 95.0
                }
            ),
            MetricConfig(
                name="memory_usage",
                metric_type=MetricType.GAUGE,
                collection_interval=5.0,
                alert_thresholds={
                    AlertSeverity.HIGH: 85.0,
                    AlertSeverity.CRITICAL: 95.0
                }
            ),
            MetricConfig(
                name="disk_usage",
                metric_type=MetricType.GAUGE,
                collection_interval=30.0,
                alert_thresholds={
                    AlertSeverity.HIGH: 90.0,
                    AlertSeverity.CRITICAL: 95.0
                }
            )
        ]
    
    for config in metric_configs:
        _performance_monitor.add_metric_config(config)


async def shutdown_performance_monitoring() -> None:
    """Shutdown performance monitoring."""
    global _performance_monitor
    if _performance_monitor:
        await _performance_monitor.stop()
        _performance_monitor = None