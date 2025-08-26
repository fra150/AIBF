"""Metrics collection and management for AI Bull Ford framework.

Provides comprehensive metrics collection, aggregation, and reporting capabilities.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Represents a single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


class MetricsCollector:
    """Collects and manages system and application metrics."""
    
    def __init__(self, collection_interval: int = 60):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Interval in seconds between metric collections
        """
        self.collection_interval = collection_interval
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.custom_collectors: List[Callable[[], List[Metric]]] = []
        self.running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
    async def start(self) -> None:
        """Start metrics collection."""
        if self.running:
            return
            
        self.running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collector started")
        
    async def stop(self) -> None:
        """Stop metrics collection."""
        if not self.running:
            return
            
        self.running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Metrics collector stopped")
        
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
                
    async def _collect_metrics(self) -> None:
        """Collect all metrics."""
        timestamp = datetime.now()
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics(timestamp)
        
        # Collect custom metrics
        custom_metrics = []
        for collector in self.custom_collectors:
            try:
                custom_metrics.extend(collector())
            except Exception as e:
                logger.error(f"Error in custom metric collector: {e}")
        
        # Store all metrics
        all_metrics = system_metrics + custom_metrics
        with self._lock:
            for metric in all_metrics:
                self.metrics[metric.name].append(metric)
                
    def _collect_system_metrics(self, timestamp: datetime) -> List[Metric]:
        """Collect system performance metrics."""
        metrics = []
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            metrics.append(Metric(
                name="system.cpu.usage_percent",
                value=cpu_percent,
                timestamp=timestamp,
                unit="percent"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(Metric(
                name="system.memory.usage_percent",
                value=memory.percent,
                timestamp=timestamp,
                unit="percent"
            ))
            metrics.append(Metric(
                name="system.memory.available_bytes",
                value=memory.available,
                timestamp=timestamp,
                unit="bytes"
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(Metric(
                name="system.disk.usage_percent",
                value=(disk.used / disk.total) * 100,
                timestamp=timestamp,
                unit="percent"
            ))
            
            # Network metrics
            network = psutil.net_io_counters()
            if network:
                metrics.append(Metric(
                    name="system.network.bytes_sent",
                    value=network.bytes_sent,
                    timestamp=timestamp,
                    unit="bytes"
                ))
                metrics.append(Metric(
                    name="system.network.bytes_recv",
                    value=network.bytes_recv,
                    timestamp=timestamp,
                    unit="bytes"
                ))
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
        return metrics
        
    def add_custom_collector(self, collector: Callable[[], List[Metric]]) -> None:
        """Add a custom metrics collector.
        
        Args:
            collector: Function that returns a list of metrics
        """
        self.custom_collectors.append(collector)
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: Optional[str] = None) -> None:
        """Record a single metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            unit: Optional unit
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[Metric]:
        """Get metrics by name.
        
        Args:
            name: Metric name
            since: Optional timestamp to filter from
            
        Returns:
            List of metrics
        """
        with self._lock:
            metrics = list(self.metrics.get(name, []))
            
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
            
        return metrics
        
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the latest metric value.
        
        Args:
            name: Metric name
            
        Returns:
            Latest metric or None
        """
        with self._lock:
            metric_queue = self.metrics.get(name)
            if metric_queue:
                return metric_queue[-1]
        return None
        
    def get_metric_summary(self, name: str, window: timedelta = timedelta(hours=1)) -> Dict[str, float]:
        """Get summary statistics for a metric.
        
        Args:
            name: Metric name
            window: Time window for summary
            
        Returns:
            Dictionary with summary statistics
        """
        since = datetime.now() - window
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {}
            
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1]
        }
        
    def get_all_metric_names(self) -> List[str]:
        """Get all metric names.
        
        Returns:
            List of metric names
        """
        with self._lock:
            return list(self.metrics.keys())
            
    def clear_metrics(self, name: Optional[str] = None) -> None:
        """Clear metrics.
        
        Args:
            name: Specific metric name to clear, or None to clear all
        """
        with self._lock:
            if name:
                self.metrics[name].clear()
            else:
                self.metrics.clear()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def initialize_metrics(config: Dict[str, Any]) -> MetricsCollector:
    """Initialize metrics collection with configuration.
    
    Args:
        config: Metrics configuration
        
    Returns:
        Metrics collector instance
    """
    global _metrics_collector
    
    metrics_config = config.get("monitoring", {}).get("metrics", {})
    collection_interval = metrics_config.get("collection_interval", 60)
    
    _metrics_collector = MetricsCollector(collection_interval)
    
    return _metrics_collector
