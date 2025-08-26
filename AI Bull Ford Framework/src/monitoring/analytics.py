"""Analytics module for AI Bull Ford.

This module provides comprehensive analytics capabilities including:
- Data collection and aggregation
- Statistical analysis and trend detection
- Predictive analytics and forecasting
- Custom analytics pipelines
- Real-time analytics dashboards
"""

import asyncio
import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple


class AnalyticsType(Enum):
    """Types of analytics."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"
    REAL_TIME = "real_time"


class AggregationWindow(Enum):
    """Time windows for data aggregation."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class TrendDirection(Enum):
    """Trend direction indicators."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class AlertCondition(Enum):
    """Analytics alert conditions."""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    ANOMALY_DETECTED = "anomaly_detected"
    TREND_CHANGED = "trend_changed"
    PATTERN_MATCHED = "pattern_matched"
    CORRELATION_FOUND = "correlation_found"


@dataclass
class DataPoint:
    """Single data point for analytics."""
    timestamp: datetime
    value: Union[int, float, str, bool]
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeries:
    """Time series data collection."""
    name: str
    data_points: List[DataPoint] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalSummary:
    """Statistical summary of data."""
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    variance: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    direction: TrendDirection
    slope: float
    correlation: float
    confidence: float
    start_value: float
    end_value: float
    change_percent: float
    duration: timedelta
    volatility: float


@dataclass
class Anomaly:
    """Detected anomaly."""
    timestamp: datetime
    value: Union[int, float]
    expected_value: Union[int, float]
    deviation: float
    severity: float
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class Forecast:
    """Forecast prediction."""
    timestamp: datetime
    predicted_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsAlert:
    """Analytics-based alert."""
    id: str
    condition: AlertCondition
    message: str
    severity: str
    timestamp: datetime = field(default_factory=datetime.now)
    data_source: str = ""
    value: Optional[Union[int, float]] = None
    threshold: Optional[Union[int, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class AnalyticsConfig:
    """Configuration for analytics processing."""
    name: str
    analytics_type: AnalyticsType
    data_sources: List[str] = field(default_factory=list)
    aggregation_window: AggregationWindow = AggregationWindow.HOUR
    retention_period: timedelta = field(default_factory=lambda: timedelta(days=30))
    update_interval: float = 60.0  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    custom_processor: Optional[Callable] = None


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.last_processed: Optional[datetime] = None
    
    @abstractmethod
    async def process(self, data: List[DataPoint]) -> Dict[str, Any]:
        """Process data points."""
        pass
    
    def should_process(self) -> bool:
        """Check if processing should occur."""
        if not self.config.enabled:
            return False
        
        if self.last_processed is None:
            return True
        
        elapsed = (datetime.now() - self.last_processed).total_seconds()
        return elapsed >= self.config.update_interval


class StatisticalProcessor(DataProcessor):
    """Processor for statistical analysis."""
    
    async def process(self, data: List[DataPoint]) -> Dict[str, Any]:
        """Process statistical analysis."""
        if not data:
            return {}
        
        # Extract numeric values
        numeric_values = []
        for point in data:
            if isinstance(point.value, (int, float)):
                numeric_values.append(float(point.value))
        
        if not numeric_values:
            return {}
        
        try:
            summary = StatisticalSummary(
                count=len(numeric_values),
                mean=statistics.mean(numeric_values),
                median=statistics.median(numeric_values),
                std_dev=statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
                min_value=min(numeric_values),
                max_value=max(numeric_values),
                percentile_25=self._percentile(numeric_values, 25),
                percentile_75=self._percentile(numeric_values, 75),
                percentile_95=self._percentile(numeric_values, 95),
                percentile_99=self._percentile(numeric_values, 99),
                variance=statistics.variance(numeric_values) if len(numeric_values) > 1 else 0.0
            )
            
            self.last_processed = datetime.now()
            
            return {
                "type": "statistical_summary",
                "summary": summary,
                "data_points": len(data),
                "time_range": {
                    "start": min(point.timestamp for point in data),
                    "end": max(point.timestamp for point in data)
                }
            }
            
        except Exception as e:
            logging.error(f"Statistical processing failed: {e}")
            return {"error": str(e)}
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]


class TrendProcessor(DataProcessor):
    """Processor for trend analysis."""
    
    async def process(self, data: List[DataPoint]) -> Dict[str, Any]:
        """Process trend analysis."""
        if len(data) < 2:
            return {}
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        # Extract numeric values with timestamps
        time_value_pairs = []
        for point in sorted_data:
            if isinstance(point.value, (int, float)):
                timestamp_seconds = point.timestamp.timestamp()
                time_value_pairs.append((timestamp_seconds, float(point.value)))
        
        if len(time_value_pairs) < 2:
            return {}
        
        try:
            # Calculate trend
            trend_analysis = self._calculate_trend(time_value_pairs)
            
            self.last_processed = datetime.now()
            
            return {
                "type": "trend_analysis",
                "trend": trend_analysis,
                "data_points": len(time_value_pairs),
                "time_range": {
                    "start": sorted_data[0].timestamp,
                    "end": sorted_data[-1].timestamp
                }
            }
            
        except Exception as e:
            logging.error(f"Trend processing failed: {e}")
            return {"error": str(e)}
    
    def _calculate_trend(self, time_value_pairs: List[Tuple[float, float]]) -> TrendAnalysis:
        """Calculate trend analysis."""
        times = [pair[0] for pair in time_value_pairs]
        values = [pair[1] for pair in time_value_pairs]
        
        # Linear regression for slope
        n = len(time_value_pairs)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in time_value_pairs)
        sum_x2 = sum(x * x for x in times)
        
        # Calculate slope and correlation
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate correlation coefficient
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in time_value_pairs)
        denominator_x = sum((x - mean_x) ** 2 for x in times)
        denominator_y = sum((y - mean_y) ** 2 for y in values)
        
        correlation = numerator / (denominator_x * denominator_y) ** 0.5 if denominator_x * denominator_y > 0 else 0
        
        # Determine trend direction
        if abs(slope) < 0.001:  # Very small slope
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING
        
        # Calculate volatility (standard deviation of values)
        volatility = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Calculate change percentage
        start_value = values[0]
        end_value = values[-1]
        change_percent = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0
        
        # Calculate duration
        duration = timedelta(seconds=times[-1] - times[0])
        
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            correlation=correlation,
            confidence=abs(correlation),
            start_value=start_value,
            end_value=end_value,
            change_percent=change_percent,
            duration=duration,
            volatility=volatility
        )


class AnomalyDetector(DataProcessor):
    """Processor for anomaly detection."""
    
    def __init__(self, config: AnalyticsConfig, sensitivity: float = 2.0):
        super().__init__(config)
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baseline_window = 100  # Number of points for baseline calculation
    
    async def process(self, data: List[DataPoint]) -> Dict[str, Any]:
        """Process anomaly detection."""
        if len(data) < self.baseline_window:
            return {}
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        # Extract numeric values
        numeric_values = []
        for point in sorted_data:
            if isinstance(point.value, (int, float)):
                numeric_values.append((point.timestamp, float(point.value)))
        
        if len(numeric_values) < self.baseline_window:
            return {}
        
        try:
            anomalies = self._detect_anomalies(numeric_values)
            
            self.last_processed = datetime.now()
            
            return {
                "type": "anomaly_detection",
                "anomalies": anomalies,
                "total_anomalies": len(anomalies),
                "data_points": len(numeric_values),
                "sensitivity": self.sensitivity
            }
            
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            return {"error": str(e)}
    
    def _detect_anomalies(self, time_value_pairs: List[Tuple[datetime, float]]) -> List[Anomaly]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        
        # Use sliding window for baseline calculation
        for i in range(self.baseline_window, len(time_value_pairs)):
            # Calculate baseline statistics from previous points
            baseline_values = [pair[1] for pair in time_value_pairs[i-self.baseline_window:i]]
            
            if len(baseline_values) < 2:
                continue
            
            mean_baseline = statistics.mean(baseline_values)
            std_baseline = statistics.stdev(baseline_values)
            
            # Current point
            current_timestamp, current_value = time_value_pairs[i]
            
            # Check if current value is anomalous
            if std_baseline > 0:
                deviation = abs(current_value - mean_baseline) / std_baseline
                
                if deviation > self.sensitivity:
                    severity = min(deviation / self.sensitivity, 5.0)  # Cap at 5.0
                    confidence = min(deviation / (self.sensitivity * 2), 1.0)  # Cap at 1.0
                    
                    anomaly = Anomaly(
                        timestamp=current_timestamp,
                        value=current_value,
                        expected_value=mean_baseline,
                        deviation=deviation,
                        severity=severity,
                        confidence=confidence,
                        description=f"Value {current_value:.2f} deviates {deviation:.2f} standard deviations from baseline {mean_baseline:.2f}"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies


class ForecastProcessor(DataProcessor):
    """Processor for forecasting."""
    
    def __init__(self, config: AnalyticsConfig, forecast_horizon: int = 10):
        super().__init__(config)
        self.forecast_horizon = forecast_horizon  # Number of future points to predict
    
    async def process(self, data: List[DataPoint]) -> Dict[str, Any]:
        """Process forecasting."""
        if len(data) < 10:  # Need minimum data for forecasting
            return {}
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        # Extract numeric values with timestamps
        time_value_pairs = []
        for point in sorted_data:
            if isinstance(point.value, (int, float)):
                time_value_pairs.append((point.timestamp, float(point.value)))
        
        if len(time_value_pairs) < 10:
            return {}
        
        try:
            forecasts = self._generate_forecasts(time_value_pairs)
            
            self.last_processed = datetime.now()
            
            return {
                "type": "forecast",
                "forecasts": forecasts,
                "forecast_horizon": self.forecast_horizon,
                "data_points": len(time_value_pairs),
                "model": "linear_trend"
            }
            
        except Exception as e:
            logging.error(f"Forecasting failed: {e}")
            return {"error": str(e)}
    
    def _generate_forecasts(self, time_value_pairs: List[Tuple[datetime, float]]) -> List[Forecast]:
        """Generate forecasts using simple linear trend."""
        # Convert timestamps to seconds for calculation
        time_seconds = [(ts.timestamp(), val) for ts, val in time_value_pairs]
        
        # Calculate trend
        times = [pair[0] for pair in time_seconds]
        values = [pair[1] for pair in time_seconds]
        
        # Linear regression
        n = len(time_seconds)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in time_seconds)
        sum_x2 = sum(x * x for x in times)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate prediction error (RMSE)
        predicted_values = [slope * x + intercept for x in times]
        errors = [actual - predicted for actual, predicted in zip(values, predicted_values)]
        rmse = (sum(e * e for e in errors) / len(errors)) ** 0.5
        
        # Generate forecasts
        forecasts = []
        last_timestamp = time_value_pairs[-1][0]
        
        # Estimate time interval between data points
        if len(time_value_pairs) > 1:
            time_interval = (time_value_pairs[-1][0] - time_value_pairs[-2][0]).total_seconds()
        else:
            time_interval = 3600  # Default to 1 hour
        
        for i in range(1, self.forecast_horizon + 1):
            future_timestamp = last_timestamp + timedelta(seconds=time_interval * i)
            future_time_seconds = future_timestamp.timestamp()
            
            predicted_value = slope * future_time_seconds + intercept
            
            # Simple confidence interval based on RMSE
            confidence_level = 0.95
            margin = rmse * 1.96  # 95% confidence interval
            
            forecast = Forecast(
                timestamp=future_timestamp,
                predicted_value=predicted_value,
                confidence_interval_lower=predicted_value - margin,
                confidence_interval_upper=predicted_value + margin,
                confidence_level=confidence_level,
                model_used="linear_trend",
                metadata={
                    "slope": slope,
                    "intercept": intercept,
                    "rmse": rmse
                }
            )
            
            forecasts.append(forecast)
        
        return forecasts


class AnalyticsEngine:
    """Main analytics processing engine."""
    
    def __init__(self):
        self.data_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.processors: Dict[str, DataProcessor] = {}
        self.configs: Dict[str, AnalyticsConfig] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[AnalyticsAlert] = []
        self.alert_handlers: List[Callable] = []
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        self._next_alert_id = 1
    
    def add_config(self, config: AnalyticsConfig) -> None:
        """Add analytics configuration."""
        self.configs[config.name] = config
        
        # Create appropriate processor
        if config.analytics_type == AnalyticsType.DESCRIPTIVE:
            processor = StatisticalProcessor(config)
        elif config.analytics_type == AnalyticsType.DIAGNOSTIC:
            processor = AnomalyDetector(config)
        elif config.analytics_type == AnalyticsType.PREDICTIVE:
            processor = ForecastProcessor(config)
        elif config.custom_processor:
            processor = config.custom_processor(config)
        else:
            processor = StatisticalProcessor(config)  # Default
        
        self.processors[config.name] = processor
    
    def add_data_point(self, source: str, data_point: DataPoint) -> None:
        """Add data point to analytics."""
        self.data_store[source].append(data_point)
    
    def add_data_points(self, source: str, data_points: List[DataPoint]) -> None:
        """Add multiple data points to analytics."""
        for point in data_points:
            self.data_store[source].append(point)
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add analytics alert handler."""
        self.alert_handlers.append(handler)
    
    async def start(self) -> None:
        """Start analytics processing."""
        if self._running:
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._processing_loop())
        logging.info("Analytics engine started")
    
    async def stop(self) -> None:
        """Stop analytics processing."""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Analytics engine stopped")
    
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Process all configured analytics
                for config_name, processor in self.processors.items():
                    if processor.should_process():
                        await self._process_analytics(config_name, processor)
                
                # Cleanup old data
                self._cleanup_old_data()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"Error in analytics processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_analytics(self, config_name: str, processor: DataProcessor) -> None:
        """Process analytics for a specific configuration."""
        config = self.configs[config_name]
        
        # Collect data from configured sources
        all_data = []
        for source in config.data_sources:
            if source in self.data_store:
                # Get data within retention period
                cutoff_time = datetime.now() - config.retention_period
                source_data = [
                    point for point in self.data_store[source]
                    if point.timestamp >= cutoff_time
                ]
                all_data.extend(source_data)
        
        if not all_data:
            return
        
        # Process analytics
        result = await processor.process(all_data)
        
        if result:
            self.results[config_name] = result
            
            # Check for alert conditions
            await self._check_alert_conditions(config_name, result)
    
    async def _check_alert_conditions(self, config_name: str, result: Dict[str, Any]) -> None:
        """Check for analytics alert conditions."""
        config = self.configs[config_name]
        
        # Check threshold alerts
        for threshold_name, threshold_value in config.alert_thresholds.items():
            if self._check_threshold_condition(result, threshold_name, threshold_value):
                alert = AnalyticsAlert(
                    id=f"analytics_alert_{self._next_alert_id}",
                    condition=AlertCondition.THRESHOLD_EXCEEDED,
                    message=f"Analytics threshold '{threshold_name}' exceeded: {threshold_value}",
                    severity="warning",
                    data_source=config_name,
                    threshold=threshold_value,
                    metadata=result
                )
                
                self._next_alert_id += 1
                self.alerts.append(alert)
                
                # Trigger alert handlers
                for handler in self.alert_handlers:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logging.error(f"Analytics alert handler failed: {e}")
        
        # Check for anomalies
        if result.get("type") == "anomaly_detection":
            anomalies = result.get("anomalies", [])
            for anomaly in anomalies:
                if anomaly.severity > 2.0:  # High severity anomalies
                    alert = AnalyticsAlert(
                        id=f"analytics_alert_{self._next_alert_id}",
                        condition=AlertCondition.ANOMALY_DETECTED,
                        message=f"Anomaly detected: {anomaly.description}",
                        severity="high" if anomaly.severity > 3.0 else "medium",
                        data_source=config_name,
                        value=anomaly.value,
                        metadata={"anomaly": anomaly}
                    )
                    
                    self._next_alert_id += 1
                    self.alerts.append(alert)
                    
                    # Trigger alert handlers
                    for handler in self.alert_handlers:
                        try:
                            await handler(alert)
                        except Exception as e:
                            logging.error(f"Analytics alert handler failed: {e}")
    
    def _check_threshold_condition(self, result: Dict[str, Any], threshold_name: str, threshold_value: float) -> bool:
        """Check if threshold condition is met."""
        # Check statistical summary thresholds
        if result.get("type") == "statistical_summary":
            summary = result.get("summary")
            if summary and hasattr(summary, threshold_name):
                value = getattr(summary, threshold_name)
                return value > threshold_value
        
        # Check trend analysis thresholds
        elif result.get("type") == "trend_analysis":
            trend = result.get("trend")
            if trend and hasattr(trend, threshold_name):
                value = getattr(trend, threshold_name)
                return abs(value) > threshold_value
        
        return False
    
    def _cleanup_old_data(self) -> None:
        """Cleanup old data points."""
        for config in self.configs.values():
            cutoff_time = datetime.now() - config.retention_period
            
            for source in config.data_sources:
                if source in self.data_store:
                    # Remove old data points
                    data_queue = self.data_store[source]
                    while data_queue and data_queue[0].timestamp < cutoff_time:
                        data_queue.popleft()
        
        # Cleanup old alerts (keep for 7 days)
        alert_cutoff = datetime.now() - timedelta(days=7)
        self.alerts = [alert for alert in self.alerts if alert.timestamp >= alert_cutoff]
    
    def get_analytics_result(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Get analytics result for a configuration."""
        return self.results.get(config_name)
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all analytics results."""
        return self.results.copy()
    
    def get_active_alerts(self) -> List[AnalyticsAlert]:
        """Get active analytics alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an analytics alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of stored data."""
        summary = {}
        
        for source, data_queue in self.data_store.items():
            if data_queue:
                summary[source] = {
                    "count": len(data_queue),
                    "oldest": data_queue[0].timestamp,
                    "newest": data_queue[-1].timestamp,
                    "sources": list(set(point.source for point in data_queue))
                }
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analytics engine statistics."""
        return {
            "running": self._running,
            "configurations": len(self.configs),
            "processors": len(self.processors),
            "data_sources": len(self.data_store),
            "total_data_points": sum(len(queue) for queue in self.data_store.values()),
            "active_alerts": len(self.get_active_alerts()),
            "total_alerts": len(self.alerts),
            "results_available": len(self.results)
        }


# Global instance
_analytics_engine: Optional[AnalyticsEngine] = None


def get_analytics_engine() -> AnalyticsEngine:
    """Get global analytics engine instance."""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine()
    return _analytics_engine


def initialize_analytics(
    analytics_configs: Optional[List[AnalyticsConfig]] = None
) -> None:
    """Initialize analytics with default configurations."""
    global _analytics_engine
    _analytics_engine = AnalyticsEngine()
    
    # Add default analytics configs if none provided
    if analytics_configs is None:
        analytics_configs = [
            AnalyticsConfig(
                name="system_performance",
                analytics_type=AnalyticsType.DESCRIPTIVE,
                data_sources=["system_metrics"],
                aggregation_window=AggregationWindow.HOUR,
                update_interval=300.0,  # 5 minutes
                alert_thresholds={"mean": 80.0, "max": 95.0}
            ),
            AnalyticsConfig(
                name="anomaly_detection",
                analytics_type=AnalyticsType.DIAGNOSTIC,
                data_sources=["system_metrics", "application_metrics"],
                update_interval=60.0,  # 1 minute
            ),
            AnalyticsConfig(
                name="performance_forecast",
                analytics_type=AnalyticsType.PREDICTIVE,
                data_sources=["system_metrics"],
                update_interval=3600.0,  # 1 hour
            )
        ]
    
    for config in analytics_configs:
        _analytics_engine.add_config(config)


async def shutdown_analytics() -> None:
    """Shutdown analytics engine."""
    global _analytics_engine
    if _analytics_engine:
        await _analytics_engine.stop()
        _analytics_engine = None