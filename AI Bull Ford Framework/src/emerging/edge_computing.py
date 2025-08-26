"""Edge Computing Module for AIBF Framework.

This module provides edge computing capabilities including:
- Edge device management
- Distributed inference
- Model compression and optimization
- Edge-cloud coordination
- Resource-aware deployment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
import threading
from collections import defaultdict, deque
import json
import hashlib

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of edge devices."""
    MOBILE = "MOBILE"
    IOT_SENSOR = "IOT_SENSOR"
    EMBEDDED = "EMBEDDED"
    EDGE_SERVER = "EDGE_SERVER"
    GATEWAY = "GATEWAY"
    DRONE = "DRONE"
    VEHICLE = "VEHICLE"
    WEARABLE = "WEARABLE"


class ComputeCapability(Enum):
    """Compute capability levels."""
    ULTRA_LOW = "ULTRA_LOW"  # < 1 GFLOPS
    LOW = "LOW"              # 1-10 GFLOPS
    MEDIUM = "MEDIUM"        # 10-100 GFLOPS
    HIGH = "HIGH"            # 100-1000 GFLOPS
    ULTRA_HIGH = "ULTRA_HIGH"  # > 1000 GFLOPS


class NetworkCondition(Enum):
    """Network connectivity conditions."""
    OFFLINE = "OFFLINE"
    POOR = "POOR"        # High latency, low bandwidth
    FAIR = "FAIR"        # Moderate latency and bandwidth
    GOOD = "GOOD"        # Low latency, high bandwidth
    EXCELLENT = "EXCELLENT"  # Very low latency, very high bandwidth


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    EDGE_ONLY = "EDGE_ONLY"
    CLOUD_ONLY = "CLOUD_ONLY"
    HYBRID = "HYBRID"
    ADAPTIVE = "ADAPTIVE"
    FEDERATED = "FEDERATED"


@dataclass
class DeviceSpecs:
    """Hardware specifications of edge device."""
    cpu_cores: int
    cpu_frequency: float  # GHz
    memory_gb: float
    storage_gb: float
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    power_budget_watts: float = 10.0
    battery_capacity_wh: Optional[float] = None
    
    def compute_score(self) -> float:
        """Calculate overall compute capability score."""
        base_score = self.cpu_cores * self.cpu_frequency * self.memory_gb
        if self.gpu_available:
            base_score *= (1 + self.gpu_memory_gb / 10.0)
        return base_score


@dataclass
class NetworkSpecs:
    """Network specifications."""
    bandwidth_mbps: float
    latency_ms: float
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0
    is_metered: bool = False
    
    def get_condition(self) -> NetworkCondition:
        """Determine network condition based on specs."""
        if self.bandwidth_mbps == 0:
            return NetworkCondition.OFFLINE
        elif self.latency_ms > 500 or self.bandwidth_mbps < 1:
            return NetworkCondition.POOR
        elif self.latency_ms > 100 or self.bandwidth_mbps < 10:
            return NetworkCondition.FAIR
        elif self.latency_ms > 50 or self.bandwidth_mbps < 100:
            return NetworkCondition.GOOD
        else:
            return NetworkCondition.EXCELLENT


@dataclass
class EdgeDevice:
    """Represents an edge computing device."""
    device_id: str
    device_type: DeviceType
    specs: DeviceSpecs
    network: NetworkSpecs
    location: Tuple[float, float] = (0.0, 0.0)  # lat, lon
    is_active: bool = True
    current_load: float = 0.0  # 0-1
    battery_level: Optional[float] = None  # 0-1
    temperature: float = 25.0  # Celsius
    last_heartbeat: float = field(default_factory=time.time)
    deployed_models: List[str] = field(default_factory=list)
    
    def get_capability(self) -> ComputeCapability:
        """Get compute capability level."""
        score = self.specs.compute_score()
        if score < 1:
            return ComputeCapability.ULTRA_LOW
        elif score < 10:
            return ComputeCapability.LOW
        elif score < 100:
            return ComputeCapability.MEDIUM
        elif score < 1000:
            return ComputeCapability.HIGH
        else:
            return ComputeCapability.ULTRA_HIGH
    
    def can_run_model(self, model_requirements: Dict[str, Any]) -> bool:
        """Check if device can run a model."""
        required_memory = model_requirements.get('memory_gb', 0)
        required_compute = model_requirements.get('min_gflops', 0)
        
        available_memory = self.specs.memory_gb * (1 - self.current_load)
        available_compute = self.specs.compute_score() * (1 - self.current_load)
        
        return (available_memory >= required_memory and 
                available_compute >= required_compute and
                self.is_active)
    
    def update_status(self, load: float, battery: Optional[float] = None, 
                     temperature: float = None):
        """Update device status."""
        self.current_load = max(0.0, min(1.0, load))
        if battery is not None:
            self.battery_level = max(0.0, min(1.0, battery))
        if temperature is not None:
            self.temperature = temperature
        self.last_heartbeat = time.time()


@dataclass
class ModelProfile:
    """Profile of a machine learning model for edge deployment."""
    model_id: str
    model_size_mb: float
    memory_requirement_gb: float
    compute_requirement_gflops: float
    inference_latency_ms: float
    accuracy: float
    supported_devices: List[DeviceType] = field(default_factory=list)
    compression_ratio: float = 1.0
    quantization_bits: int = 32
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get model requirements dictionary."""
        return {
            'memory_gb': self.memory_requirement_gb,
            'min_gflops': self.compute_requirement_gflops,
            'model_size_mb': self.model_size_mb,
            'latency_ms': self.inference_latency_ms
        }


class ModelCompressor:
    """Model compression utilities for edge deployment."""
    
    def __init__(self):
        self.compression_techniques = {
            'quantization': self._quantize_model,
            'pruning': self._prune_model,
            'distillation': self._distill_model,
            'low_rank': self._low_rank_approximation
        }
    
    def compress_model(self, model_profile: ModelProfile, 
                      target_device: EdgeDevice,
                      techniques: List[str] = None) -> ModelProfile:
        """Compress model for target device."""
        try:
            if techniques is None:
                techniques = self._select_compression_techniques(model_profile, target_device)
            
            compressed_profile = ModelProfile(
                model_id=f"{model_profile.model_id}_compressed",
                model_size_mb=model_profile.model_size_mb,
                memory_requirement_gb=model_profile.memory_requirement_gb,
                compute_requirement_gflops=model_profile.compute_requirement_gflops,
                inference_latency_ms=model_profile.inference_latency_ms,
                accuracy=model_profile.accuracy,
                supported_devices=model_profile.supported_devices.copy()
            )
            
            for technique in techniques:
                if technique in self.compression_techniques:
                    compressed_profile = self.compression_techniques[technique](compressed_profile, target_device)
            
            logger.info(f"Compressed model {model_profile.model_id} for {target_device.device_id}")
            return compressed_profile
            
        except Exception as e:
            logger.error(f"Model compression failed: {e}")
            raise
    
    def _select_compression_techniques(self, model_profile: ModelProfile, 
                                     target_device: EdgeDevice) -> List[str]:
        """Select appropriate compression techniques."""
        techniques = []
        capability = target_device.get_capability()
        
        if capability in [ComputeCapability.ULTRA_LOW, ComputeCapability.LOW]:
            techniques.extend(['quantization', 'pruning', 'distillation'])
        elif capability == ComputeCapability.MEDIUM:
            techniques.extend(['quantization', 'pruning'])
        else:
            techniques.append('quantization')
        
        return techniques
    
    def _quantize_model(self, profile: ModelProfile, device: EdgeDevice) -> ModelProfile:
        """Apply quantization compression."""
        if device.get_capability() in [ComputeCapability.ULTRA_LOW, ComputeCapability.LOW]:
            profile.quantization_bits = 8
            profile.model_size_mb *= 0.25
            profile.memory_requirement_gb *= 0.3
            profile.compute_requirement_gflops *= 0.5
            profile.inference_latency_ms *= 0.7
            profile.accuracy *= 0.98  # Small accuracy loss
        else:
            profile.quantization_bits = 16
            profile.model_size_mb *= 0.5
            profile.memory_requirement_gb *= 0.6
            profile.compute_requirement_gflops *= 0.7
            profile.inference_latency_ms *= 0.8
            profile.accuracy *= 0.995
        
        return profile
    
    def _prune_model(self, profile: ModelProfile, device: EdgeDevice) -> ModelProfile:
        """Apply pruning compression."""
        capability = device.get_capability()
        
        if capability == ComputeCapability.ULTRA_LOW:
            pruning_ratio = 0.8
        elif capability == ComputeCapability.LOW:
            pruning_ratio = 0.6
        else:
            pruning_ratio = 0.3
        
        profile.model_size_mb *= (1 - pruning_ratio)
        profile.memory_requirement_gb *= (1 - pruning_ratio * 0.8)
        profile.compute_requirement_gflops *= (1 - pruning_ratio)
        profile.inference_latency_ms *= (1 - pruning_ratio * 0.5)
        profile.accuracy *= (1 - pruning_ratio * 0.1)  # Accuracy loss
        
        return profile
    
    def _distill_model(self, profile: ModelProfile, device: EdgeDevice) -> ModelProfile:
        """Apply knowledge distillation."""
        # Create smaller student model
        profile.model_size_mb *= 0.2
        profile.memory_requirement_gb *= 0.3
        profile.compute_requirement_gflops *= 0.25
        profile.inference_latency_ms *= 0.4
        profile.accuracy *= 0.95  # Moderate accuracy loss
        
        return profile
    
    def _low_rank_approximation(self, profile: ModelProfile, device: EdgeDevice) -> ModelProfile:
        """Apply low-rank approximation."""
        profile.model_size_mb *= 0.7
        profile.memory_requirement_gb *= 0.8
        profile.compute_requirement_gflops *= 0.6
        profile.inference_latency_ms *= 0.9
        profile.accuracy *= 0.99
        
        return profile


class EdgeOrchestrator:
    """Orchestrates edge computing resources and deployments."""
    
    def __init__(self):
        self.devices: Dict[str, EdgeDevice] = {}
        self.models: Dict[str, ModelProfile] = {}
        self.deployments: Dict[str, Dict[str, Any]] = {}  # model_id -> deployment info
        self.compressor = ModelCompressor()
        self.load_balancer = EdgeLoadBalancer()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def register_device(self, device: EdgeDevice):
        """Register edge device."""
        self.devices[device.device_id] = device
        logger.info(f"Registered edge device: {device.device_id} ({device.device_type.value})")
    
    def unregister_device(self, device_id: str):
        """Unregister edge device."""
        if device_id in self.devices:
            # Migrate models if any
            device = self.devices[device_id]
            if device.deployed_models:
                self._migrate_models(device_id)
            
            del self.devices[device_id]
            logger.info(f"Unregistered edge device: {device_id}")
    
    def register_model(self, model: ModelProfile):
        """Register model profile."""
        self.models[model.model_id] = model
        logger.info(f"Registered model: {model.model_id}")
    
    def deploy_model(self, model_id: str, strategy: DeploymentStrategy = DeploymentStrategy.ADAPTIVE,
                    target_devices: List[str] = None) -> Dict[str, Any]:
        """Deploy model to edge devices."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            if strategy == DeploymentStrategy.EDGE_ONLY:
                deployment_plan = self._plan_edge_deployment(model, target_devices)
            elif strategy == DeploymentStrategy.ADAPTIVE:
                deployment_plan = self._plan_adaptive_deployment(model, target_devices)
            else:
                raise NotImplementedError(f"Strategy {strategy.value} not implemented")
            
            # Execute deployment
            deployment_result = self._execute_deployment(model, deployment_plan)
            
            self.deployments[model_id] = {
                'strategy': strategy,
                'plan': deployment_plan,
                'result': deployment_result,
                'timestamp': time.time()
            }
            
            logger.info(f"Deployed model {model_id} using {strategy.value} strategy")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    def _plan_edge_deployment(self, model: ModelProfile, 
                            target_devices: List[str] = None) -> Dict[str, Any]:
        """Plan edge-only deployment."""
        suitable_devices = []
        
        devices_to_check = (
            [self.devices[d] for d in target_devices if d in self.devices]
            if target_devices else list(self.devices.values())
        )
        
        for device in devices_to_check:
            if device.can_run_model(model.get_requirements()):
                # Compress model for device
                compressed_model = self.compressor.compress_model(model, device)
                
                if device.can_run_model(compressed_model.get_requirements()):
                    suitable_devices.append({
                        'device_id': device.device_id,
                        'device': device,
                        'compressed_model': compressed_model,
                        'score': self._calculate_deployment_score(device, compressed_model)
                    })
        
        # Sort by deployment score
        suitable_devices.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'type': 'edge_only',
            'suitable_devices': suitable_devices,
            'primary_device': suitable_devices[0] if suitable_devices else None,
            'backup_devices': suitable_devices[1:3] if len(suitable_devices) > 1 else []
        }
    
    def _plan_adaptive_deployment(self, model: ModelProfile,
                                target_devices: List[str] = None) -> Dict[str, Any]:
        """Plan adaptive deployment based on conditions."""
        edge_plan = self._plan_edge_deployment(model, target_devices)
        
        # Analyze network conditions
        network_conditions = [device.network.get_condition() for device in self.devices.values()]
        avg_network_quality = sum(cond.value == 'EXCELLENT' for cond in network_conditions) / len(network_conditions)
        
        if edge_plan['suitable_devices'] and avg_network_quality < 0.5:
            # Prefer edge deployment when network is poor
            return edge_plan
        else:
            # Hybrid deployment
            return {
                'type': 'hybrid',
                'edge_plan': edge_plan,
                'cloud_fallback': True,
                'load_balancing': True
            }
    
    def _calculate_deployment_score(self, device: EdgeDevice, model: ModelProfile) -> float:
        """Calculate deployment score for device-model pair."""
        # Base score from device capability
        capability_score = device.specs.compute_score()
        
        # Network quality score
        network_score = {
            NetworkCondition.OFFLINE: 0,
            NetworkCondition.POOR: 0.2,
            NetworkCondition.FAIR: 0.5,
            NetworkCondition.GOOD: 0.8,
            NetworkCondition.EXCELLENT: 1.0
        }[device.network.get_condition()]
        
        # Load score (prefer less loaded devices)
        load_score = 1.0 - device.current_load
        
        # Battery score (if applicable)
        battery_score = device.battery_level if device.battery_level is not None else 1.0
        
        # Model fit score
        requirements = model.get_requirements()
        memory_fit = min(1.0, device.specs.memory_gb / requirements['memory_gb'])
        compute_fit = min(1.0, device.specs.compute_score() / requirements['min_gflops'])
        
        # Weighted combination
        score = (
            capability_score * 0.3 +
            network_score * 0.2 +
            load_score * 0.2 +
            battery_score * 0.1 +
            memory_fit * 0.1 +
            compute_fit * 0.1
        )
        
        return score
    
    def _execute_deployment(self, model: ModelProfile, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment plan."""
        deployed_devices = []
        failed_devices = []
        
        if plan['type'] == 'edge_only':
            primary_device = plan['primary_device']
            if primary_device:
                success = self._deploy_to_device(
                    primary_device['device'], 
                    primary_device['compressed_model']
                )
                
                if success:
                    deployed_devices.append(primary_device['device_id'])
                else:
                    failed_devices.append(primary_device['device_id'])
                    
                    # Try backup devices
                    for backup in plan['backup_devices']:
                        success = self._deploy_to_device(
                            backup['device'],
                            backup['compressed_model']
                        )
                        if success:
                            deployed_devices.append(backup['device_id'])
                            break
                        else:
                            failed_devices.append(backup['device_id'])
        
        return {
            'deployed_devices': deployed_devices,
            'failed_devices': failed_devices,
            'deployment_time': time.time(),
            'success': len(deployed_devices) > 0
        }
    
    def _deploy_to_device(self, device: EdgeDevice, model: ModelProfile) -> bool:
        """Deploy model to specific device."""
        try:
            # Simulate deployment process
            if device.can_run_model(model.get_requirements()):
                device.deployed_models.append(model.model_id)
                device.current_load += model.compute_requirement_gflops / device.specs.compute_score()
                device.current_load = min(1.0, device.current_load)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Deployment to device {device.device_id} failed: {e}")
            return False
    
    def _migrate_models(self, source_device_id: str):
        """Migrate models from source device to other devices."""
        if source_device_id not in self.devices:
            return
        
        source_device = self.devices[source_device_id]
        models_to_migrate = source_device.deployed_models.copy()
        
        for model_id in models_to_migrate:
            if model_id in self.models:
                # Find alternative device
                model = self.models[model_id]
                plan = self._plan_edge_deployment(model, 
                    [d for d in self.devices.keys() if d != source_device_id])
                
                if plan['suitable_devices']:
                    target_device = plan['suitable_devices'][0]
                    success = self._deploy_to_device(
                        target_device['device'],
                        target_device['compressed_model']
                    )
                    
                    if success:
                        logger.info(f"Migrated model {model_id} from {source_device_id} to {target_device['device_id']}")
    
    def start_monitoring(self):
        """Start device monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_devices)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Started edge device monitoring")
    
    def stop_monitoring(self):
        """Stop device monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped edge device monitoring")
    
    def _monitor_devices(self):
        """Monitor device health and performance."""
        while self._monitoring_active:
            try:
                current_time = time.time()
                
                for device_id, device in list(self.devices.items()):
                    # Check heartbeat
                    if current_time - device.last_heartbeat > 60:  # 1 minute timeout
                        logger.warning(f"Device {device_id} heartbeat timeout")
                        device.is_active = False
                    
                    # Check battery level
                    if device.battery_level is not None and device.battery_level < 0.1:
                        logger.warning(f"Device {device_id} low battery: {device.battery_level:.1%}")
                    
                    # Check temperature
                    if device.temperature > 80:  # 80°C threshold
                        logger.warning(f"Device {device_id} high temperature: {device.temperature}°C")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Device monitoring error: {e}")
                time.sleep(30)
    
    def get_deployment_status(self, model_id: str) -> Dict[str, Any]:
        """Get deployment status for model."""
        if model_id not in self.deployments:
            return {'status': 'not_deployed'}
        
        deployment = self.deployments[model_id]
        
        # Check device health
        deployed_devices = deployment['result']['deployed_devices']
        healthy_devices = []
        unhealthy_devices = []
        
        for device_id in deployed_devices:
            if device_id in self.devices and self.devices[device_id].is_active:
                healthy_devices.append(device_id)
            else:
                unhealthy_devices.append(device_id)
        
        return {
            'status': 'deployed',
            'strategy': deployment['strategy'].value,
            'deployed_devices': deployed_devices,
            'healthy_devices': healthy_devices,
            'unhealthy_devices': unhealthy_devices,
            'deployment_time': deployment['timestamp']
        }
    
    def get_edge_info(self) -> Dict[str, Any]:
        """Get edge computing system information."""
        total_devices = len(self.devices)
        active_devices = sum(1 for d in self.devices.values() if d.is_active)
        
        device_types = defaultdict(int)
        capabilities = defaultdict(int)
        
        for device in self.devices.values():
            device_types[device.device_type.value] += 1
            capabilities[device.get_capability().value] += 1
        
        return {
            'total_devices': total_devices,
            'active_devices': active_devices,
            'device_types': dict(device_types),
            'capabilities': dict(capabilities),
            'registered_models': len(self.models),
            'active_deployments': len(self.deployments),
            'monitoring_active': self._monitoring_active
        }


class EdgeLoadBalancer:
    """Load balancer for edge inference requests."""
    
    def __init__(self):
        self.request_queues: Dict[str, deque] = defaultdict(deque)
        self.device_loads: Dict[str, float] = defaultdict(float)
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        
    def route_request(self, model_id: str, available_devices: List[str],
                     request_data: Any) -> str:
        """Route inference request to best available device."""
        if not available_devices:
            raise ValueError("No available devices for routing")
        
        # Select device with lowest load
        best_device = min(available_devices, key=lambda d: self.device_loads[d])
        
        # Add request to queue
        self.request_queues[best_device].append({
            'model_id': model_id,
            'data': request_data,
            'timestamp': time.time()
        })
        
        # Update load estimate
        self.device_loads[best_device] += 0.1  # Rough estimate
        
        return best_device
    
    def complete_request(self, device_id: str, response_time: float):
        """Mark request as completed."""
        # Update load
        self.device_loads[device_id] = max(0, self.device_loads[device_id] - 0.1)
        
        # Record response time
        self.response_times[device_id].append(response_time)
        if len(self.response_times[device_id]) > 100:  # Keep last 100 measurements
            self.response_times[device_id].pop(0)
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        avg_response_times = {}
        for device_id, times in self.response_times.items():
            if times:
                avg_response_times[device_id] = sum(times) / len(times)
        
        return {
            'device_loads': dict(self.device_loads),
            'queue_lengths': {d: len(q) for d, q in self.request_queues.items()},
            'avg_response_times': avg_response_times
        }


class EdgeManager:
    """Main interface for edge computing capabilities."""
    
    def __init__(self):
        self.orchestrator = EdgeOrchestrator()
        self.compressor = ModelCompressor()
        self.load_balancer = EdgeLoadBalancer()
        
    def register_device(self, device_specs: Dict[str, Any]) -> str:
        """Register new edge device."""
        try:
            device = EdgeDevice(
                device_id=device_specs['device_id'],
                device_type=DeviceType(device_specs['device_type']),
                specs=DeviceSpecs(**device_specs['specs']),
                network=NetworkSpecs(**device_specs['network']),
                location=device_specs.get('location', (0.0, 0.0))
            )
            
            self.orchestrator.register_device(device)
            return device.device_id
            
        except Exception as e:
            logger.error(f"Device registration failed: {e}")
            raise
    
    def register_model(self, model_specs: Dict[str, Any]) -> str:
        """Register model for edge deployment."""
        try:
            model = ModelProfile(**model_specs)
            self.orchestrator.register_model(model)
            return model.model_id
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise
    
    def deploy_model(self, model_id: str, strategy: str = "adaptive",
                    target_devices: List[str] = None) -> Dict[str, Any]:
        """Deploy model to edge devices."""
        try:
            strategy_enum = DeploymentStrategy(strategy.upper())
            return self.orchestrator.deploy_model(model_id, strategy_enum, target_devices)
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise
    
    def compress_model(self, model_id: str, target_device_id: str,
                      techniques: List[str] = None) -> Dict[str, Any]:
        """Compress model for specific device."""
        try:
            if model_id not in self.orchestrator.models:
                raise ValueError(f"Model {model_id} not found")
            if target_device_id not in self.orchestrator.devices:
                raise ValueError(f"Device {target_device_id} not found")
            
            model = self.orchestrator.models[model_id]
            device = self.orchestrator.devices[target_device_id]
            
            compressed_model = self.compressor.compress_model(model, device, techniques)
            
            return {
                'original_size_mb': model.model_size_mb,
                'compressed_size_mb': compressed_model.model_size_mb,
                'compression_ratio': model.model_size_mb / compressed_model.model_size_mb,
                'accuracy_loss': model.accuracy - compressed_model.accuracy,
                'latency_improvement': model.inference_latency_ms - compressed_model.inference_latency_ms
            }
            
        except Exception as e:
            logger.error(f"Model compression failed: {e}")
            raise
    
    def route_inference_request(self, model_id: str, request_data: Any) -> Dict[str, Any]:
        """Route inference request to appropriate device."""
        try:
            # Find devices with deployed model
            available_devices = []
            for device_id, device in self.orchestrator.devices.items():
                if model_id in device.deployed_models and device.is_active:
                    available_devices.append(device_id)
            
            if not available_devices:
                raise ValueError(f"No available devices for model {model_id}")
            
            selected_device = self.load_balancer.route_request(
                model_id, available_devices, request_data
            )
            
            return {
                'selected_device': selected_device,
                'available_devices': available_devices,
                'routing_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            raise
    
    def update_device_status(self, device_id: str, status_update: Dict[str, Any]):
        """Update device status."""
        try:
            if device_id not in self.orchestrator.devices:
                raise ValueError(f"Device {device_id} not found")
            
            device = self.orchestrator.devices[device_id]
            device.update_status(
                load=status_update.get('load', device.current_load),
                battery=status_update.get('battery', device.battery_level),
                temperature=status_update.get('temperature', device.temperature)
            )
            
        except Exception as e:
            logger.error(f"Device status update failed: {e}")
            raise
    
    def start_monitoring(self):
        """Start edge system monitoring."""
        self.orchestrator.start_monitoring()
    
    def stop_monitoring(self):
        """Stop edge system monitoring."""
        self.orchestrator.stop_monitoring()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        edge_info = self.orchestrator.get_edge_info()
        load_stats = self.load_balancer.get_load_stats()
        
        return {
            'edge_info': edge_info,
            'load_balancing': load_stats,
            'timestamp': time.time()
        }
    
    def get_edge_computing_info(self) -> Dict[str, Any]:
        """Get information about edge computing capabilities."""
        return {
            'supported_device_types': [dt.value for dt in DeviceType],
            'compute_capabilities': [cc.value for cc in ComputeCapability],
            'network_conditions': [nc.value for nc in NetworkCondition],
            'deployment_strategies': [ds.value for ds in DeploymentStrategy],
            'compression_techniques': list(self.compressor.compression_techniques.keys()),
            'system_status': self.get_system_status()
        }


# Utility functions
def create_mobile_device(device_id: str, location: Tuple[float, float] = (0.0, 0.0)) -> EdgeDevice:
    """Create a typical mobile device configuration."""
    return EdgeDevice(
        device_id=device_id,
        device_type=DeviceType.MOBILE,
        specs=DeviceSpecs(
            cpu_cores=8,
            cpu_frequency=2.4,
            memory_gb=6.0,
            storage_gb=128.0,
            gpu_available=True,
            gpu_memory_gb=2.0,
            power_budget_watts=5.0,
            battery_capacity_wh=15.0
        ),
        network=NetworkSpecs(
            bandwidth_mbps=50.0,
            latency_ms=30.0,
            is_metered=True
        ),
        location=location,
        battery_level=0.8
    )


def create_iot_sensor(device_id: str, location: Tuple[float, float] = (0.0, 0.0)) -> EdgeDevice:
    """Create a typical IoT sensor configuration."""
    return EdgeDevice(
        device_id=device_id,
        device_type=DeviceType.IOT_SENSOR,
        specs=DeviceSpecs(
            cpu_cores=1,
            cpu_frequency=0.5,
            memory_gb=0.5,
            storage_gb=4.0,
            power_budget_watts=1.0,
            battery_capacity_wh=5.0
        ),
        network=NetworkSpecs(
            bandwidth_mbps=1.0,
            latency_ms=100.0,
            is_metered=True
        ),
        location=location,
        battery_level=0.9
    )


def create_edge_server(device_id: str, location: Tuple[float, float] = (0.0, 0.0)) -> EdgeDevice:
    """Create a typical edge server configuration."""
    return EdgeDevice(
        device_id=device_id,
        device_type=DeviceType.EDGE_SERVER,
        specs=DeviceSpecs(
            cpu_cores=16,
            cpu_frequency=3.2,
            memory_gb=64.0,
            storage_gb=1000.0,
            gpu_available=True,
            gpu_memory_gb=16.0,
            power_budget_watts=200.0
        ),
        network=NetworkSpecs(
            bandwidth_mbps=1000.0,
            latency_ms=5.0
        ),
        location=location
    )


# Export main classes and functions
__all__ = [
    'EdgeManager',
    'EdgeOrchestrator',
    'EdgeDevice',
    'ModelProfile',
    'ModelCompressor',
    'EdgeLoadBalancer',
    'DeviceSpecs',
    'NetworkSpecs',
    'DeviceType',
    'ComputeCapability',
    'NetworkCondition',
    'DeploymentStrategy',
    'create_mobile_device',
    'create_iot_sensor',
    'create_edge_server'
]