"""Neuromorphic Computing Module for AIBF Framework.

This module provides neuromorphic computing capabilities including:
- Spiking neural networks (SNNs)
- Event-driven processing
- Temporal dynamics modeling
- Bio-inspired learning algorithms
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
from collections import deque

logger = logging.getLogger(__name__)


class NeuronModel(Enum):
    """Supported neuron models."""
    LEAKY_INTEGRATE_FIRE = "LIF"
    IZHIKEVICH = "IZH"
    HODGKIN_HUXLEY = "HH"
    ADAPTIVE_EXPONENTIAL = "AEIF"
    INTEGRATE_FIRE = "IF"


class SynapseType(Enum):
    """Types of synaptic connections."""
    EXCITATORY = "EXC"
    INHIBITORY = "INH"
    MODULATORY = "MOD"


class LearningRule(Enum):
    """Supported learning rules."""
    STDP = "STDP"  # Spike-Timing Dependent Plasticity
    RSTDP = "RSTDP"  # Reward-modulated STDP
    BCM = "BCM"  # Bienenstock-Cooper-Munro
    HOMEOSTATIC = "HOMEOSTATIC"


@dataclass
class SpikeEvent:
    """Represents a spike event."""
    neuron_id: int
    timestamp: float
    amplitude: float = 1.0


@dataclass
class Synapse:
    """Synaptic connection between neurons."""
    pre_neuron: int
    post_neuron: int
    weight: float
    delay: float
    synapse_type: SynapseType
    plasticity_enabled: bool = True
    last_update: float = 0.0


class SpikingNeuron(ABC):
    """Abstract base class for spiking neuron models."""
    
    def __init__(self, neuron_id: int, **params):
        self.neuron_id = neuron_id
        self.membrane_potential = params.get('v_rest', -70.0)
        self.threshold = params.get('threshold', -55.0)
        self.reset_potential = params.get('v_reset', -80.0)
        self.refractory_period = params.get('refractory', 2.0)
        self.last_spike_time = -float('inf')
        self.spike_history: List[float] = []
        
    @abstractmethod
    def update(self, dt: float, input_current: float) -> bool:
        """Update neuron state and return True if spike occurred."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset neuron to initial state."""
        pass
    
    def is_refractory(self, current_time: float) -> bool:
        """Check if neuron is in refractory period."""
        return (current_time - self.last_spike_time) < self.refractory_period
    
    def add_spike(self, timestamp: float):
        """Record a spike event."""
        self.last_spike_time = timestamp
        self.spike_history.append(timestamp)
        # Keep only recent spikes for memory efficiency
        if len(self.spike_history) > 1000:
            self.spike_history = self.spike_history[-500:]


class LeakyIntegrateFireNeuron(SpikingNeuron):
    """Leaky Integrate-and-Fire neuron model."""
    
    def __init__(self, neuron_id: int, **params):
        super().__init__(neuron_id, **params)
        self.tau_m = params.get('tau_m', 20.0)  # Membrane time constant
        self.resistance = params.get('resistance', 1.0)
        self.capacitance = params.get('capacitance', 1.0)
        self.v_rest = params.get('v_rest', -70.0)
        self.membrane_potential = self.v_rest
        
    def update(self, dt: float, input_current: float) -> bool:
        """Update LIF neuron dynamics."""
        if self.is_refractory(time.time()):
            return False
            
        # LIF dynamics: tau_m * dV/dt = -(V - V_rest) + R*I
        dv_dt = (-(self.membrane_potential - self.v_rest) + 
                self.resistance * input_current) / self.tau_m
        
        self.membrane_potential += dv_dt * dt
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = self.reset_potential
            self.add_spike(time.time())
            return True
            
        return False
    
    def reset(self):
        """Reset neuron state."""
        self.membrane_potential = self.v_rest
        self.spike_history.clear()
        self.last_spike_time = -float('inf')


class IzhikevichNeuron(SpikingNeuron):
    """Izhikevich neuron model."""
    
    def __init__(self, neuron_id: int, **params):
        super().__init__(neuron_id, **params)
        self.a = params.get('a', 0.02)  # Recovery time constant
        self.b = params.get('b', 0.2)   # Sensitivity of recovery
        self.c = params.get('c', -65.0) # After-spike reset value
        self.d = params.get('d', 8.0)   # After-spike recovery boost
        self.v = params.get('v_init', -70.0)
        self.u = self.b * self.v
        
    def update(self, dt: float, input_current: float) -> bool:
        """Update Izhikevich neuron dynamics."""
        # Izhikevich model equations
        dv_dt = 0.04 * self.v**2 + 5 * self.v + 140 - self.u + input_current
        du_dt = self.a * (self.b * self.v - self.u)
        
        self.v += dv_dt * dt
        self.u += du_dt * dt
        
        # Check for spike
        if self.v >= 30.0:  # Spike threshold
            self.v = self.c
            self.u += self.d
            self.add_spike(time.time())
            return True
            
        return False
    
    def reset(self):
        """Reset neuron state."""
        self.v = -70.0
        self.u = self.b * self.v
        self.spike_history.clear()
        self.last_spike_time = -float('inf')


class STDPLearning:
    """Spike-Timing Dependent Plasticity learning rule."""
    
    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0, 
                 a_plus: float = 0.01, a_minus: float = 0.012):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        
    def update_weight(self, synapse: Synapse, pre_spike_time: float, 
                     post_spike_time: float) -> float:
        """Update synaptic weight based on spike timing."""
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # Post-synaptic spike after pre-synaptic
            delta_w = self.a_plus * np.exp(-dt / self.tau_plus)
        else:  # Pre-synaptic spike after post-synaptic
            delta_w = -self.a_minus * np.exp(dt / self.tau_minus)
            
        # Update weight with bounds
        new_weight = synapse.weight + delta_w
        return np.clip(new_weight, 0.0, 1.0)


class SpikingNeuralNetwork:
    """Spiking Neural Network implementation."""
    
    def __init__(self, name: str = "SNN"):
        self.name = name
        self.neurons: Dict[int, SpikingNeuron] = {}
        self.synapses: List[Synapse] = []
        self.spike_events: deque = deque(maxlen=10000)
        self.current_time = 0.0
        self.dt = 0.1  # Time step in ms
        self.learning_rules: Dict[str, Any] = {}
        
    def add_neuron(self, neuron: SpikingNeuron) -> int:
        """Add neuron to network."""
        self.neurons[neuron.neuron_id] = neuron
        return neuron.neuron_id
    
    def add_synapse(self, synapse: Synapse):
        """Add synaptic connection."""
        if (synapse.pre_neuron in self.neurons and 
            synapse.post_neuron in self.neurons):
            self.synapses.append(synapse)
        else:
            raise ValueError("Both pre and post neurons must exist in network")
    
    def connect_neurons(self, pre_id: int, post_id: int, weight: float, 
                       delay: float = 1.0, synapse_type: SynapseType = SynapseType.EXCITATORY):
        """Create synaptic connection between neurons."""
        synapse = Synapse(pre_id, post_id, weight, delay, synapse_type)
        self.add_synapse(synapse)
    
    def add_learning_rule(self, rule_name: str, learning_rule: Any):
        """Add plasticity learning rule."""
        self.learning_rules[rule_name] = learning_rule
    
    def simulate(self, duration: float, input_currents: Optional[Dict[int, List[float]]] = None) -> Dict[str, Any]:
        """Simulate network for given duration."""
        try:
            steps = int(duration / self.dt)
            spike_trains = {neuron_id: [] for neuron_id in self.neurons.keys()}
            membrane_potentials = {neuron_id: [] for neuron_id in self.neurons.keys()}
            
            for step in range(steps):
                self.current_time = step * self.dt
                
                # Calculate input currents for each neuron
                neuron_currents = self._calculate_neuron_currents(input_currents, step)
                
                # Update all neurons
                for neuron_id, neuron in self.neurons.items():
                    current = neuron_currents.get(neuron_id, 0.0)
                    
                    # Update neuron
                    spiked = neuron.update(self.dt, current)
                    
                    # Record data
                    if spiked:
                        spike_trains[neuron_id].append(self.current_time)
                        self.spike_events.append(SpikeEvent(neuron_id, self.current_time))
                    
                    membrane_potentials[neuron_id].append(neuron.membrane_potential)
                
                # Apply plasticity rules
                self._apply_plasticity()
            
            return {
                "spike_trains": spike_trains,
                "membrane_potentials": membrane_potentials,
                "simulation_time": duration,
                "time_step": self.dt,
                "network_stats": self._get_network_stats()
            }
            
        except Exception as e:
            logger.error(f"SNN simulation failed: {e}")
            raise
    
    def _calculate_neuron_currents(self, external_currents: Optional[Dict[int, List[float]]], 
                                  step: int) -> Dict[int, float]:
        """Calculate total input current for each neuron."""
        currents = {neuron_id: 0.0 for neuron_id in self.neurons.keys()}
        
        # Add external currents
        if external_currents:
            for neuron_id, current_trace in external_currents.items():
                if neuron_id in currents and step < len(current_trace):
                    currents[neuron_id] += current_trace[step]
        
        # Add synaptic currents
        for synapse in self.synapses:
            pre_neuron = self.neurons[synapse.pre_neuron]
            
            # Check for recent spikes with delay
            for spike_time in pre_neuron.spike_history:
                if (self.current_time - spike_time >= synapse.delay and 
                    self.current_time - spike_time < synapse.delay + self.dt):
                    
                    # Apply synaptic current
                    current_amplitude = synapse.weight
                    if synapse.synapse_type == SynapseType.INHIBITORY:
                        current_amplitude *= -1
                    
                    currents[synapse.post_neuron] += current_amplitude
        
        return currents
    
    def _apply_plasticity(self):
        """Apply synaptic plasticity rules."""
        if "STDP" in self.learning_rules:
            stdp_rule = self.learning_rules["STDP"]
            
            for synapse in self.synapses:
                if not synapse.plasticity_enabled:
                    continue
                    
                pre_neuron = self.neurons[synapse.pre_neuron]
                post_neuron = self.neurons[synapse.post_neuron]
                
                # Find recent spike pairs
                for pre_spike in pre_neuron.spike_history[-10:]:  # Recent spikes only
                    for post_spike in post_neuron.spike_history[-10:]:
                        if abs(pre_spike - post_spike) < 50.0:  # Within STDP window
                            synapse.weight = stdp_rule.update_weight(
                                synapse, pre_spike, post_spike
                            )
    
    def _get_network_stats(self) -> Dict[str, Any]:
        """Calculate network statistics."""
        total_spikes = sum(len(neuron.spike_history) for neuron in self.neurons.values())
        avg_firing_rate = total_spikes / (len(self.neurons) * self.current_time) * 1000  # Hz
        
        return {
            "total_neurons": len(self.neurons),
            "total_synapses": len(self.synapses),
            "total_spikes": total_spikes,
            "average_firing_rate": avg_firing_rate,
            "simulation_duration": self.current_time
        }
    
    def reset(self):
        """Reset network to initial state."""
        for neuron in self.neurons.values():
            neuron.reset()
        self.spike_events.clear()
        self.current_time = 0.0


class EventDrivenProcessor:
    """Event-driven processing for neuromorphic computing."""
    
    def __init__(self, buffer_size: int = 10000):
        self.event_buffer: deque = deque(maxlen=buffer_size)
        self.processors: List[Callable] = []
        self.active = False
        
    def add_event(self, event: SpikeEvent):
        """Add event to processing buffer."""
        self.event_buffer.append(event)
        
        if self.active:
            self._process_event(event)
    
    def add_processor(self, processor: Callable[[SpikeEvent], None]):
        """Add event processor function."""
        self.processors.append(processor)
    
    def start_processing(self):
        """Start event-driven processing."""
        self.active = True
        
        # Process any buffered events
        while self.event_buffer:
            event = self.event_buffer.popleft()
            self._process_event(event)
    
    def stop_processing(self):
        """Stop event-driven processing."""
        self.active = False
    
    def _process_event(self, event: SpikeEvent):
        """Process single event through all processors."""
        for processor in self.processors:
            try:
                processor(event)
            except Exception as e:
                logger.error(f"Event processing failed: {e}")


class NeuromorphicLearning:
    """Bio-inspired learning algorithms for neuromorphic systems."""
    
    def __init__(self):
        self.learning_history: List[Dict[str, Any]] = []
    
    def hebbian_learning(self, network: SpikingNeuralNetwork, 
                        learning_rate: float = 0.01) -> Dict[str, Any]:
        """Implement Hebbian learning rule."""
        try:
            weight_changes = []
            
            for synapse in network.synapses:
                pre_neuron = network.neurons[synapse.pre_neuron]
                post_neuron = network.neurons[synapse.post_neuron]
                
                # Calculate correlation between pre and post activity
                pre_activity = len(pre_neuron.spike_history) / network.current_time if network.current_time > 0 else 0
                post_activity = len(post_neuron.spike_history) / network.current_time if network.current_time > 0 else 0
                
                # Hebbian update: "neurons that fire together, wire together"
                delta_w = learning_rate * pre_activity * post_activity
                old_weight = synapse.weight
                synapse.weight = np.clip(synapse.weight + delta_w, 0.0, 1.0)
                
                weight_changes.append({
                    "synapse": f"{synapse.pre_neuron}->{synapse.post_neuron}",
                    "old_weight": old_weight,
                    "new_weight": synapse.weight,
                    "delta": delta_w
                })
            
            result = {
                "learning_rule": "Hebbian",
                "learning_rate": learning_rate,
                "weight_changes": weight_changes,
                "timestamp": time.time()
            }
            
            self.learning_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Hebbian learning failed: {e}")
            raise
    
    def homeostatic_plasticity(self, network: SpikingNeuralNetwork, 
                              target_rate: float = 10.0) -> Dict[str, Any]:
        """Implement homeostatic plasticity to maintain target firing rates."""
        try:
            adjustments = []
            
            for neuron_id, neuron in network.neurons.items():
                if network.current_time <= 0:
                    continue
                    
                # Calculate current firing rate
                current_rate = len(neuron.spike_history) / (network.current_time / 1000.0)  # Hz
                
                # Find synapses targeting this neuron
                incoming_synapses = [s for s in network.synapses if s.post_neuron == neuron_id]
                
                if incoming_synapses:
                    # Adjust weights to reach target rate
                    rate_error = target_rate - current_rate
                    adjustment_factor = 1.0 + (rate_error / target_rate) * 0.1
                    
                    for synapse in incoming_synapses:
                        old_weight = synapse.weight
                        synapse.weight = np.clip(synapse.weight * adjustment_factor, 0.0, 1.0)
                        
                        adjustments.append({
                            "neuron_id": neuron_id,
                            "synapse": f"{synapse.pre_neuron}->{synapse.post_neuron}",
                            "current_rate": current_rate,
                            "target_rate": target_rate,
                            "old_weight": old_weight,
                            "new_weight": synapse.weight
                        })
            
            result = {
                "learning_rule": "Homeostatic",
                "target_rate": target_rate,
                "adjustments": adjustments,
                "timestamp": time.time()
            }
            
            self.learning_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Homeostatic plasticity failed: {e}")
            raise


class NeuromorphicManager:
    """Main interface for neuromorphic computing capabilities."""
    
    def __init__(self):
        self.networks: Dict[str, SpikingNeuralNetwork] = {}
        self.event_processor = EventDrivenProcessor()
        self.learning_engine = NeuromorphicLearning()
        
    def create_network(self, name: str) -> SpikingNeuralNetwork:
        """Create a new spiking neural network."""
        network = SpikingNeuralNetwork(name)
        self.networks[name] = network
        return network
    
    def create_neuron(self, model: NeuronModel, neuron_id: int, **params) -> SpikingNeuron:
        """Create a neuron with specified model."""
        if model == NeuronModel.LEAKY_INTEGRATE_FIRE:
            return LeakyIntegrateFireNeuron(neuron_id, **params)
        elif model == NeuronModel.IZHIKEVICH:
            return IzhikevichNeuron(neuron_id, **params)
        else:
            raise NotImplementedError(f"Neuron model {model.value} not implemented")
    
    def build_feedforward_network(self, name: str, layer_sizes: List[int], 
                                 neuron_model: NeuronModel = NeuronModel.LEAKY_INTEGRATE_FIRE) -> SpikingNeuralNetwork:
        """Build a feedforward spiking neural network."""
        try:
            network = self.create_network(name)
            neuron_id = 0
            layer_neurons = []
            
            # Create neurons for each layer
            for layer_idx, size in enumerate(layer_sizes):
                layer = []
                for _ in range(size):
                    neuron = self.create_neuron(neuron_model, neuron_id)
                    network.add_neuron(neuron)
                    layer.append(neuron_id)
                    neuron_id += 1
                layer_neurons.append(layer)
            
            # Connect layers
            for layer_idx in range(len(layer_neurons) - 1):
                current_layer = layer_neurons[layer_idx]
                next_layer = layer_neurons[layer_idx + 1]
                
                for pre_id in current_layer:
                    for post_id in next_layer:
                        # Random initial weights
                        weight = np.random.uniform(0.1, 0.5)
                        network.connect_neurons(pre_id, post_id, weight)
            
            # Add STDP learning
            stdp = STDPLearning()
            network.add_learning_rule("STDP", stdp)
            
            logger.info(f"Created feedforward SNN '{name}' with {len(layer_sizes)} layers")
            return network
            
        except Exception as e:
            logger.error(f"Failed to build feedforward network: {e}")
            raise
    
    def simulate_network(self, network_name: str, duration: float, 
                        input_patterns: Optional[Dict[int, List[float]]] = None) -> Dict[str, Any]:
        """Simulate a spiking neural network."""
        if network_name not in self.networks:
            raise ValueError(f"Network '{network_name}' not found")
            
        network = self.networks[network_name]
        return network.simulate(duration, input_patterns)
    
    def apply_learning(self, network_name: str, learning_rule: LearningRule, 
                      **kwargs) -> Dict[str, Any]:
        """Apply learning rule to network."""
        if network_name not in self.networks:
            raise ValueError(f"Network '{network_name}' not found")
            
        network = self.networks[network_name]
        
        if learning_rule == LearningRule.STDP:
            # STDP is applied during simulation
            return {"message": "STDP learning enabled during simulation"}
        elif learning_rule == LearningRule.BCM:
            return self.learning_engine.hebbian_learning(network, **kwargs)
        elif learning_rule == LearningRule.HOMEOSTATIC:
            return self.learning_engine.homeostatic_plasticity(network, **kwargs)
        else:
            raise NotImplementedError(f"Learning rule {learning_rule.value} not implemented")
    
    def get_network_info(self, network_name: str) -> Dict[str, Any]:
        """Get information about a network."""
        if network_name not in self.networks:
            raise ValueError(f"Network '{network_name}' not found")
            
        network = self.networks[network_name]
        return {
            "name": network.name,
            "neurons": len(network.neurons),
            "synapses": len(network.synapses),
            "current_time": network.current_time,
            "time_step": network.dt,
            "learning_rules": list(network.learning_rules.keys())
        }
    
    def get_neuromorphic_info(self) -> Dict[str, Any]:
        """Get information about neuromorphic computing capabilities."""
        return {
            "supported_neuron_models": [model.value for model in NeuronModel],
            "supported_learning_rules": [rule.value for rule in LearningRule],
            "synapse_types": [stype.value for stype in SynapseType],
            "networks": list(self.networks.keys()),
            "event_driven_processing": True,
            "plasticity_support": True
        }


# Utility functions
def create_poisson_spike_train(rate: float, duration: float, dt: float = 0.1) -> List[float]:
    """Generate Poisson spike train."""
    spike_times = []
    t = 0.0
    
    while t < duration:
        # Inter-spike interval from exponential distribution
        isi = np.random.exponential(1000.0 / rate)  # Convert Hz to ms
        t += isi
        if t < duration:
            spike_times.append(t)
    
    return spike_times


def encode_rate_to_spikes(values: List[float], max_rate: float = 100.0, 
                         duration: float = 100.0) -> Dict[int, List[float]]:
    """Encode analog values as spike trains using rate coding."""
    spike_trains = {}
    
    for i, value in enumerate(values):
        # Normalize value to firing rate
        rate = max(0, min(max_rate, abs(value) * max_rate))
        spike_trains[i] = create_poisson_spike_train(rate, duration)
    
    return spike_trains


# Export main classes and functions
__all__ = [
    'NeuromorphicManager',
    'SpikingNeuralNetwork',
    'SpikingNeuron',
    'LeakyIntegrateFireNeuron',
    'IzhikevichNeuron',
    'STDPLearning',
    'EventDrivenProcessor',
    'NeuromorphicLearning',
    'NeuronModel',
    'SynapseType',
    'LearningRule',
    'SpikeEvent',
    'Synapse',
    'create_poisson_spike_train',
    'encode_rate_to_spikes'
]