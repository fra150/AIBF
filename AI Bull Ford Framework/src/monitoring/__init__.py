"""Monitoring and analytics module for AI Bull Ford.

This module provides comprehensive monitoring capabilities including:
- Performance monitoring and metrics collection
- Health checks and system status monitoring
- Analytics and data processing
- Resource monitoring and optimization
"""

# Performance monitoring
from .performance import (
    MetricType,
    AggregationType,
    AlertSeverity,
    Metric,
    MetricConfig,
    Alert,
    PerformanceReport,
    MetricCollector,
    SystemMetricCollector,
    CustomMetricCollector,
    MetricAggregator,
    AlertManager,
    PerformanceMonitor,
    get_performance_monitor,
    initialize_performance_monitoring,
    shutdown_performance_monitoring
)

# Health monitoring
from .health import (
    HealthStatus,
    CheckType,
    RecoveryAction,
    HealthCheckResult,
    ServiceHealth,
    HealthConfig,
    SystemHealth,
    HealthCheck,
    DatabaseHealthCheck,
    APIHealthCheck,
    ResourceHealthCheck,
    CustomHealthCheck,
    RecoveryManager,
    HealthMonitor,
    get_health_monitor,
    initialize_health_monitoring,
    shutdown_health_monitoring
)

# Analytics
from .analytics import (
    AnalyticsType,
    AggregationWindow,
    TrendDirection,
    AlertCondition,
    DataPoint,
    TimeSeries,
    StatisticalSummary,
    TrendAnalysis,
    Anomaly,
    Forecast,
    AnalyticsAlert,
    AnalyticsConfig,
    DataProcessor,
    StatisticalProcessor,
    TrendProcessor,
    AnomalyDetector,
    ForecastProcessor,
    AnalyticsEngine,
    get_analytics_engine,
    initialize_analytics,
    shutdown_analytics
)

# Resource monitoring
from .resource import (
    ResourceType,
    ResourceStatus,
    OptimizationAction,
    ResourceMetric,
    ResourceUsage,
    ResourceLimit,
    ResourceAlert,
    OptimizationRecommendation,
    ResourceConfig,
    ResourceCollector,
    SystemResourceCollector,
    ResourceAnalyzer,
    ResourceMonitor,
    get_resource_monitor,
    initialize_resource_monitoring,
    shutdown_resource_monitoring
)

__all__ = [
    # Performance monitoring
    "MetricType",
    "AggregationType",
    "AlertSeverity",
    "Metric",
    "MetricConfig",
    "Alert",
    "PerformanceReport",
    "MetricCollector",
    "SystemMetricCollector",
    "CustomMetricCollector",
    "MetricAggregator",
    "AlertManager",
    "PerformanceMonitor",
    "get_performance_monitor",
    "initialize_performance_monitoring",
    "shutdown_performance_monitoring",
    
    # Health monitoring
    "HealthStatus",
    "CheckType",
    "RecoveryAction",
    "HealthCheckResult",
    "ServiceHealth",
    "HealthConfig",
    "SystemHealth",
    "HealthCheck",
    "DatabaseHealthCheck",
    "APIHealthCheck",
    "ResourceHealthCheck",
    "CustomHealthCheck",
    "RecoveryManager",
    "HealthMonitor",
    "get_health_monitor",
    "initialize_health_monitoring",
    "shutdown_health_monitoring",
    
    # Analytics
    "AnalyticsType",
    "AggregationWindow",
    "TrendDirection",
    "AlertCondition",
    "DataPoint",
    "TimeSeries",
    "StatisticalSummary",
    "TrendAnalysis",
    "Anomaly",
    "Forecast",
    "AnalyticsAlert",
    "AnalyticsConfig",
    "DataProcessor",
    "StatisticalProcessor",
    "TrendProcessor",
    "AnomalyDetector",
    "ForecastProcessor",
    "AnalyticsEngine",
    "get_analytics_engine",
    "initialize_analytics",
    "shutdown_analytics",
    
    # Resource monitoring
    "ResourceType",
    "ResourceStatus",
    "OptimizationAction",
    "ResourceMetric",
    "ResourceUsage",
    "ResourceLimit",
    "ResourceAlert",
    "OptimizationRecommendation",
    "ResourceConfig",
    "ResourceCollector",
    "SystemResourceCollector",
    "ResourceAnalyzer",
    "ResourceMonitor",
    "get_resource_monitor",
    "initialize_resource_monitoring",
    "shutdown_resource_monitoring"
]