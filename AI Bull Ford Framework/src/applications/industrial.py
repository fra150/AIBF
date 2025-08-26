"""Industrial automation module for AI Bull Ford.

This module provides comprehensive industrial automation capabilities including:
- Manufacturing process control
- Quality control and inspection
- Predictive maintenance
- Supply chain optimization
- Production planning and scheduling
- Industrial IoT integration
"""

import asyncio
import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class ProcessType(Enum):
    """Types of industrial processes."""
    ASSEMBLY = "assembly"
    MACHINING = "machining"
    WELDING = "welding"
    PAINTING = "painting"
    PACKAGING = "packaging"
    QUALITY_CONTROL = "quality_control"
    MATERIAL_HANDLING = "material_handling"
    CHEMICAL = "chemical"
    PHARMACEUTICAL = "pharmaceutical"
    FOOD_PROCESSING = "food_processing"


class EquipmentStatus(Enum):
    """Equipment operational status."""
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    IDLE = "idle"
    SETUP = "setup"
    EMERGENCY_STOP = "emergency_stop"
    OFFLINE = "offline"


class QualityStatus(Enum):
    """Quality control status."""
    PASS = "pass"
    FAIL = "fail"
    REWORK = "rework"
    PENDING = "pending"
    QUARANTINE = "quarantine"


class MaintenanceType(Enum):
    """Types of maintenance."""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    SCHEDULED = "scheduled"


@dataclass
class IndustrialConfig:
    """Configuration for industrial automation systems."""
    plant_id: str = "default_plant"
    max_production_rate: float = 100.0  # units per hour
    quality_threshold: float = 0.95  # 95% quality target
    maintenance_interval: float = 168.0  # hours (1 week)
    alert_threshold: float = 0.8  # 80% threshold for alerts
    data_retention_days: int = 30
    real_time_monitoring: bool = True
    predictive_maintenance: bool = True
    auto_quality_control: bool = True
    logging_enabled: bool = True


@dataclass
class ProductionData:
    """Production data point."""
    timestamp: datetime
    process_type: ProcessType
    equipment_id: str
    production_count: int
    quality_score: float
    cycle_time: float  # seconds
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    vibration: Optional[float] = None
    power_consumption: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityInspection:
    """Quality inspection result."""
    inspection_id: str
    product_id: str
    timestamp: datetime
    inspector: str  # human or AI system
    status: QualityStatus
    defects: List[str] = field(default_factory=list)
    measurements: Dict[str, float] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)  # image paths
    confidence: float = 1.0
    notes: str = ""


@dataclass
class MaintenanceRecord:
    """Maintenance record."""
    record_id: str
    equipment_id: str
    maintenance_type: MaintenanceType
    scheduled_date: datetime
    completed_date: Optional[datetime] = None
    technician: str = ""
    description: str = ""
    parts_replaced: List[str] = field(default_factory=list)
    cost: float = 0.0
    downtime_hours: float = 0.0
    notes: str = ""


@dataclass
class EquipmentHealth:
    """Equipment health status."""
    equipment_id: str
    timestamp: datetime
    overall_health: float  # 0-1 score
    status: EquipmentStatus
    temperature: Optional[float] = None
    vibration_level: Optional[float] = None
    power_consumption: Optional[float] = None
    cycle_count: int = 0
    last_maintenance: Optional[datetime] = None
    predicted_failure_date: Optional[datetime] = None
    alerts: List[str] = field(default_factory=list)


@dataclass
class ProductionOrder:
    """Production order."""
    order_id: str
    product_type: str
    quantity: int
    priority: int
    due_date: datetime
    created_date: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    assigned_equipment: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0-1
    quality_requirements: Dict[str, Any] = field(default_factory=dict)


class ProcessController:
    """Controls industrial processes."""
    
    def __init__(self, config: IndustrialConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processes = {}
        self.production_data = []
        self.active_orders = {}
    
    def register_process(self, process_id: str, process_type: ProcessType, equipment_ids: List[str]) -> None:
        """Register a new process."""
        try:
            self.processes[process_id] = {
                'type': process_type,
                'equipment_ids': equipment_ids,
                'status': 'idle',
                'current_order': None,
                'start_time': None,
                'parameters': {}
            }
            self.logger.info(f"Registered process {process_id} of type {process_type}")
        except Exception as e:
            self.logger.error(f"Failed to register process {process_id}: {e}")
            raise
    
    def start_production(self, process_id: str, order: ProductionOrder) -> None:
        """Start production for an order."""
        try:
            if process_id not in self.processes:
                raise ValueError(f"Unknown process: {process_id}")
            
            process = self.processes[process_id]
            if process['status'] != 'idle':
                raise ValueError(f"Process {process_id} is not idle")
            
            process['status'] = 'running'
            process['current_order'] = order.order_id
            process['start_time'] = datetime.now()
            
            self.active_orders[order.order_id] = order
            order.status = 'in_progress'
            
            self.logger.info(f"Started production for order {order.order_id} on process {process_id}")
        except Exception as e:
            self.logger.error(f"Failed to start production: {e}")
            raise
    
    def record_production_data(self, equipment_id: str, process_type: ProcessType, 
                             production_count: int, quality_score: float, cycle_time: float,
                             **kwargs) -> None:
        """Record production data."""
        try:
            data_point = ProductionData(
                timestamp=datetime.now(),
                process_type=process_type,
                equipment_id=equipment_id,
                production_count=production_count,
                quality_score=quality_score,
                cycle_time=cycle_time,
                temperature=kwargs.get('temperature'),
                pressure=kwargs.get('pressure'),
                vibration=kwargs.get('vibration'),
                power_consumption=kwargs.get('power_consumption'),
                metadata=kwargs.get('metadata', {})
            )
            
            self.production_data.append(data_point)
            
            # Keep only recent data
            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
            self.production_data = [d for d in self.production_data if d.timestamp > cutoff_date]
            
            # Check for alerts
            if quality_score < self.config.quality_threshold:
                self.logger.warning(f"Quality alert: {equipment_id} quality score {quality_score} below threshold")
            
        except Exception as e:
            self.logger.error(f"Failed to record production data: {e}")
            raise
    
    def get_production_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get production metrics for the last N hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = [d for d in self.production_data if d.timestamp > cutoff_time]
            
            if not recent_data:
                return {'total_production': 0, 'average_quality': 0, 'average_cycle_time': 0}
            
            total_production = sum(d.production_count for d in recent_data)
            quality_scores = [d.quality_score for d in recent_data]
            cycle_times = [d.cycle_time for d in recent_data]
            
            metrics = {
                'total_production': total_production,
                'average_quality': statistics.mean(quality_scores),
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores),
                'average_cycle_time': statistics.mean(cycle_times),
                'min_cycle_time': min(cycle_times),
                'max_cycle_time': max(cycle_times),
                'production_rate': total_production / hours,
                'data_points': len(recent_data)
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get production metrics: {e}")
            return {}


class QualityController:
    """Controls quality inspection and management."""
    
    def __init__(self, config: IndustrialConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.inspections = []
        self.quality_standards = {}
        self.defect_patterns = {}
    
    def set_quality_standard(self, product_type: str, standards: Dict[str, Any]) -> None:
        """Set quality standards for a product type."""
        self.quality_standards[product_type] = standards
        self.logger.info(f"Set quality standards for {product_type}")
    
    def perform_inspection(self, product_id: str, product_type: str, 
                         measurements: Dict[str, float], images: List[str] = None) -> QualityInspection:
        """Perform quality inspection."""
        try:
            inspection_id = f"QI_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{product_id}"
            
            # Get quality standards
            standards = self.quality_standards.get(product_type, {})
            
            # Check measurements against standards
            defects = []
            for measurement, value in measurements.items():
                if measurement in standards:
                    min_val = standards[measurement].get('min')
                    max_val = standards[measurement].get('max')
                    
                    if min_val is not None and value < min_val:
                        defects.append(f"{measurement} below minimum ({value} < {min_val})")
                    if max_val is not None and value > max_val:
                        defects.append(f"{measurement} above maximum ({value} > {max_val})")
            
            # Determine status
            if defects:
                status = QualityStatus.FAIL if len(defects) > 2 else QualityStatus.REWORK
            else:
                status = QualityStatus.PASS
            
            # Calculate confidence (simplified)
            confidence = 1.0 - (len(defects) * 0.2)
            confidence = max(0.0, min(1.0, confidence))
            
            inspection = QualityInspection(
                inspection_id=inspection_id,
                product_id=product_id,
                timestamp=datetime.now(),
                inspector="AI_QC_System",
                status=status,
                defects=defects,
                measurements=measurements,
                images=images or [],
                confidence=confidence
            )
            
            self.inspections.append(inspection)
            
            # Keep only recent inspections
            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
            self.inspections = [i for i in self.inspections if i.timestamp > cutoff_date]
            
            self.logger.info(f"Completed inspection {inspection_id}: {status}")
            return inspection
            
        except Exception as e:
            self.logger.error(f"Failed to perform inspection: {e}")
            raise
    
    def get_quality_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get quality metrics for the last N hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_inspections = [i for i in self.inspections if i.timestamp > cutoff_time]
            
            if not recent_inspections:
                return {'total_inspections': 0, 'pass_rate': 0, 'defect_rate': 0}
            
            total_inspections = len(recent_inspections)
            passed = sum(1 for i in recent_inspections if i.status == QualityStatus.PASS)
            failed = sum(1 for i in recent_inspections if i.status == QualityStatus.FAIL)
            rework = sum(1 for i in recent_inspections if i.status == QualityStatus.REWORK)
            
            # Count defect types
            defect_counts = {}
            for inspection in recent_inspections:
                for defect in inspection.defects:
                    defect_counts[defect] = defect_counts.get(defect, 0) + 1
            
            metrics = {
                'total_inspections': total_inspections,
                'passed': passed,
                'failed': failed,
                'rework': rework,
                'pass_rate': passed / total_inspections,
                'fail_rate': failed / total_inspections,
                'rework_rate': rework / total_inspections,
                'defect_rate': (failed + rework) / total_inspections,
                'top_defects': sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to get quality metrics: {e}")
            return {}


class MaintenanceManager:
    """Manages equipment maintenance."""
    
    def __init__(self, config: IndustrialConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.equipment_health = {}
        self.maintenance_records = []
        self.maintenance_schedule = {}
    
    def register_equipment(self, equipment_id: str, equipment_type: str) -> None:
        """Register equipment for monitoring."""
        try:
            self.equipment_health[equipment_id] = EquipmentHealth(
                equipment_id=equipment_id,
                timestamp=datetime.now(),
                overall_health=1.0,
                status=EquipmentStatus.OPERATIONAL
            )
            
            # Schedule initial maintenance
            next_maintenance = datetime.now() + timedelta(hours=self.config.maintenance_interval)
            self.maintenance_schedule[equipment_id] = next_maintenance
            
            self.logger.info(f"Registered equipment {equipment_id} of type {equipment_type}")
        except Exception as e:
            self.logger.error(f"Failed to register equipment {equipment_id}: {e}")
            raise
    
    def update_equipment_health(self, equipment_id: str, **health_data) -> None:
        """Update equipment health data."""
        try:
            if equipment_id not in self.equipment_health:
                raise ValueError(f"Unknown equipment: {equipment_id}")
            
            health = self.equipment_health[equipment_id]
            health.timestamp = datetime.now()
            
            # Update health parameters
            if 'temperature' in health_data:
                health.temperature = health_data['temperature']
            if 'vibration_level' in health_data:
                health.vibration_level = health_data['vibration_level']
            if 'power_consumption' in health_data:
                health.power_consumption = health_data['power_consumption']
            if 'cycle_count' in health_data:
                health.cycle_count = health_data['cycle_count']
            
            # Calculate overall health score (simplified)
            health_score = 1.0
            alerts = []
            
            # Temperature check
            if health.temperature and health.temperature > 80:  # Example threshold
                health_score -= 0.2
                alerts.append("High temperature detected")
            
            # Vibration check
            if health.vibration_level and health.vibration_level > 5.0:  # Example threshold
                health_score -= 0.3
                alerts.append("High vibration detected")
            
            # Power consumption check
            if health.power_consumption and health.power_consumption > 1000:  # Example threshold
                health_score -= 0.1
                alerts.append("High power consumption")
            
            health.overall_health = max(0.0, health_score)
            health.alerts = alerts
            
            # Update status based on health
            if health.overall_health < 0.3:
                health.status = EquipmentStatus.FAULT
            elif health.overall_health < 0.7:
                health.status = EquipmentStatus.MAINTENANCE
            else:
                health.status = EquipmentStatus.OPERATIONAL
            
            # Predict failure date (simplified)
            if health.overall_health < 0.8:
                days_to_failure = int(health.overall_health * 30)  # Simplified prediction
                health.predicted_failure_date = datetime.now() + timedelta(days=days_to_failure)
            
            # Check if maintenance is needed
            if health.overall_health < self.config.alert_threshold:
                self.logger.warning(f"Equipment {equipment_id} health below threshold: {health.overall_health}")
            
        except Exception as e:
            self.logger.error(f"Failed to update equipment health: {e}")
            raise
    
    def schedule_maintenance(self, equipment_id: str, maintenance_type: MaintenanceType, 
                           scheduled_date: datetime, description: str = "") -> str:
        """Schedule maintenance for equipment."""
        try:
            record_id = f"MR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{equipment_id}"
            
            record = MaintenanceRecord(
                record_id=record_id,
                equipment_id=equipment_id,
                maintenance_type=maintenance_type,
                scheduled_date=scheduled_date,
                description=description
            )
            
            self.maintenance_records.append(record)
            self.maintenance_schedule[equipment_id] = scheduled_date
            
            self.logger.info(f"Scheduled {maintenance_type} maintenance for {equipment_id} on {scheduled_date}")
            return record_id
        except Exception as e:
            self.logger.error(f"Failed to schedule maintenance: {e}")
            raise
    
    def complete_maintenance(self, record_id: str, technician: str, 
                           parts_replaced: List[str] = None, cost: float = 0.0, 
                           downtime_hours: float = 0.0, notes: str = "") -> None:
        """Mark maintenance as completed."""
        try:
            record = next((r for r in self.maintenance_records if r.record_id == record_id), None)
            if not record:
                raise ValueError(f"Maintenance record not found: {record_id}")
            
            record.completed_date = datetime.now()
            record.technician = technician
            record.parts_replaced = parts_replaced or []
            record.cost = cost
            record.downtime_hours = downtime_hours
            record.notes = notes
            
            # Update equipment health
            if record.equipment_id in self.equipment_health:
                health = self.equipment_health[record.equipment_id]
                health.last_maintenance = record.completed_date
                health.overall_health = min(1.0, health.overall_health + 0.3)  # Improve health
                health.status = EquipmentStatus.OPERATIONAL
                health.alerts = []
            
            # Schedule next maintenance
            next_maintenance = record.completed_date + timedelta(hours=self.config.maintenance_interval)
            self.maintenance_schedule[record.equipment_id] = next_maintenance
            
            self.logger.info(f"Completed maintenance {record_id} for {record.equipment_id}")
        except Exception as e:
            self.logger.error(f"Failed to complete maintenance: {e}")
            raise
    
    def get_maintenance_due(self, days_ahead: int = 7) -> List[str]:
        """Get equipment due for maintenance in the next N days."""
        try:
            cutoff_date = datetime.now() + timedelta(days=days_ahead)
            due_equipment = []
            
            for equipment_id, scheduled_date in self.maintenance_schedule.items():
                if scheduled_date <= cutoff_date:
                    due_equipment.append(equipment_id)
            
            return due_equipment
        except Exception as e:
            self.logger.error(f"Failed to get maintenance due: {e}")
            return []


class ProductionPlanner:
    """Plans and schedules production."""
    
    def __init__(self, config: IndustrialConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.production_orders = {}
        self.equipment_capacity = {}
        self.production_schedule = {}
    
    def add_production_order(self, order: ProductionOrder) -> None:
        """Add a production order."""
        try:
            self.production_orders[order.order_id] = order
            self.logger.info(f"Added production order {order.order_id} for {order.quantity} units")
        except Exception as e:
            self.logger.error(f"Failed to add production order: {e}")
            raise
    
    def set_equipment_capacity(self, equipment_id: str, capacity: float) -> None:
        """Set equipment production capacity (units per hour)."""
        self.equipment_capacity[equipment_id] = capacity
        self.logger.info(f"Set capacity for {equipment_id}: {capacity} units/hour")
    
    def schedule_production(self) -> Dict[str, List[str]]:
        """Schedule production orders on available equipment."""
        try:
            # Sort orders by priority and due date
            sorted_orders = sorted(
                self.production_orders.values(),
                key=lambda x: (x.priority, x.due_date),
                reverse=True
            )
            
            schedule = {equipment_id: [] for equipment_id in self.equipment_capacity.keys()}
            equipment_availability = {equipment_id: datetime.now() for equipment_id in self.equipment_capacity.keys()}
            
            for order in sorted_orders:
                if order.status != 'pending':
                    continue
                
                # Find best equipment for this order
                best_equipment = None
                earliest_completion = None
                
                for equipment_id, capacity in self.equipment_capacity.items():
                    if capacity <= 0:
                        continue
                    
                    # Calculate completion time
                    production_time = order.quantity / capacity  # hours
                    start_time = equipment_availability[equipment_id]
                    completion_time = start_time + timedelta(hours=production_time)
                    
                    if earliest_completion is None or completion_time < earliest_completion:
                        best_equipment = equipment_id
                        earliest_completion = completion_time
                
                if best_equipment:
                    # Schedule order on best equipment
                    schedule[best_equipment].append(order.order_id)
                    order.assigned_equipment = [best_equipment]
                    order.status = 'scheduled'
                    
                    # Update equipment availability
                    production_time = order.quantity / self.equipment_capacity[best_equipment]
                    equipment_availability[best_equipment] += timedelta(hours=production_time)
                    
                    self.logger.info(f"Scheduled order {order.order_id} on {best_equipment}")
            
            self.production_schedule = schedule
            return schedule
        except Exception as e:
            self.logger.error(f"Failed to schedule production: {e}")
            return {}
    
    def get_production_forecast(self, days: int = 7) -> Dict[str, Any]:
        """Get production forecast for the next N days."""
        try:
            forecast_end = datetime.now() + timedelta(days=days)
            
            # Calculate total planned production
            total_planned = 0
            orders_in_period = 0
            
            for order in self.production_orders.values():
                if order.due_date <= forecast_end and order.status in ['pending', 'scheduled', 'in_progress']:
                    total_planned += order.quantity
                    orders_in_period += 1
            
            # Calculate capacity utilization
            total_capacity = sum(self.equipment_capacity.values()) * 24 * days  # total capacity in period
            capacity_utilization = (total_planned / total_capacity) if total_capacity > 0 else 0
            
            forecast = {
                'period_days': days,
                'total_planned_production': total_planned,
                'orders_in_period': orders_in_period,
                'total_capacity': total_capacity,
                'capacity_utilization': capacity_utilization,
                'equipment_schedule': self.production_schedule
            }
            
            return forecast
        except Exception as e:
            self.logger.error(f"Failed to get production forecast: {e}")
            return {}


class IndustrialAutomationSystem:
    """Main industrial automation system integrating all components."""
    
    def __init__(self, config: Optional[IndustrialConfig] = None):
        self.config = config or IndustrialConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize subsystems
        self.process_controller = ProcessController(self.config)
        self.quality_controller = QualityController(self.config)
        self.maintenance_manager = MaintenanceManager(self.config)
        self.production_planner = ProductionPlanner(self.config)
        
        self.running = False
        self.monitoring_task = None
    
    async def start(self) -> None:
        """Start industrial automation system."""
        try:
            self.running = True
            if self.config.real_time_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Industrial automation system started")
        except Exception as e:
            self.logger.error(f"Failed to start industrial automation system: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop industrial automation system."""
        try:
            self.running = False
            if self.monitoring_task:
                await self.monitoring_task
            self.logger.info("Industrial automation system stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop industrial automation system: {e}")
            raise
    
    async def _monitoring_loop(self) -> None:
        """Real-time monitoring loop."""
        while self.running:
            try:
                # Check equipment health
                for equipment_id in self.maintenance_manager.equipment_health.keys():
                    # Simulate health data updates (in real system, would read from sensors)
                    health_data = {
                        'temperature': np.random.normal(70, 10),
                        'vibration_level': np.random.normal(2, 1),
                        'power_consumption': np.random.normal(500, 100)
                    }
                    self.maintenance_manager.update_equipment_health(equipment_id, **health_data)
                
                # Check for maintenance due
                due_equipment = self.maintenance_manager.get_maintenance_due()
                if due_equipment:
                    self.logger.info(f"Equipment due for maintenance: {due_equipment}")
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        try:
            production_metrics = self.process_controller.get_production_metrics()
            quality_metrics = self.quality_controller.get_quality_metrics()
            
            # Equipment health summary
            equipment_health = list(self.maintenance_manager.equipment_health.values())
            avg_health = statistics.mean([h.overall_health for h in equipment_health]) if equipment_health else 0
            
            # Production forecast
            forecast = self.production_planner.get_production_forecast()
            
            status = {
                'timestamp': datetime.now(),
                'system_running': self.running,
                'production_metrics': production_metrics,
                'quality_metrics': quality_metrics,
                'average_equipment_health': avg_health,
                'equipment_count': len(equipment_health),
                'production_forecast': forecast,
                'alerts': self._get_system_alerts()
            }
            
            return status
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {}
    
    def _get_system_alerts(self) -> List[str]:
        """Get system-wide alerts."""
        alerts = []
        
        # Equipment health alerts
        for health in self.maintenance_manager.equipment_health.values():
            if health.overall_health < self.config.alert_threshold:
                alerts.append(f"Equipment {health.equipment_id} health low: {health.overall_health:.2f}")
            alerts.extend(health.alerts)
        
        # Quality alerts
        quality_metrics = self.quality_controller.get_quality_metrics()
        if quality_metrics.get('pass_rate', 1.0) < self.config.quality_threshold:
            alerts.append(f"Quality pass rate below threshold: {quality_metrics.get('pass_rate', 0):.2f}")
        
        # Production alerts
        production_metrics = self.process_controller.get_production_metrics()
        if production_metrics.get('production_rate', 0) < self.config.max_production_rate * 0.8:
            alerts.append(f"Production rate below 80% of capacity")
        
        return alerts


# Global industrial automation system instance
_industrial_system: Optional[IndustrialAutomationSystem] = None


def initialize_industrial(config: Optional[IndustrialConfig] = None) -> None:
    """Initialize industrial automation system."""
    global _industrial_system
    _industrial_system = IndustrialAutomationSystem(config)


async def shutdown_industrial() -> None:
    """Shutdown industrial automation system."""
    global _industrial_system
    if _industrial_system:
        await _industrial_system.stop()
        _industrial_system = None