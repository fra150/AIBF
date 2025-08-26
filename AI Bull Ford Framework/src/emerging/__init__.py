"""Emerging Technologies Module for AIBF Framework.

This module provides cutting-edge AI technologies including:
- Quantum computing integration
- Neuromorphic computing
- Bio-inspired algorithms
- Edge computing optimization
- Federated learning
"""

from .quantum import QuantumManager, QuantumCircuit, QuantumSimulator
from .neuromorphic import NeuromorphicManager, SpikingNeuralNetwork, LeakyIntegrateFireNeuron
from .bio_inspired import BioInspiredManager, GeneticAlgorithm, ParticleSwarmOptimization
from .edge_computing import EdgeManager, EdgeDevice, ModelCompressor
from .federated_learning import FederatedLearningManager, FederatedServer, FederatedClient

__all__ = [
    # Quantum Computing
    'QuantumManager',
    'QuantumCircuit', 
    'QuantumSimulator',
    
    # Neuromorphic Computing
    'NeuromorphicManager',
    'SpikingNeuralNetwork',
    'LeakyIntegrateFireNeuron',
    
    # Bio-inspired AI
    'BioInspiredManager',
    'GeneticAlgorithm',
    'ParticleSwarmOptimization',
    
    # Edge Computing
    'EdgeManager',
    'EdgeDevice',
    'ModelCompressor',
    
    # Federated Learning
    'FederatedLearningManager',
    'FederatedServer',
    'FederatedClient',
]

__version__ = "1.0.0"