"""Federated Learning Module for AIBF Framework.

This module provides federated learning capabilities including:
- Federated averaging algorithms
- Client-server coordination
- Privacy-preserving techniques
- Secure aggregation
- Differential privacy
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
import random
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Federated learning aggregation strategies."""
    FEDERATED_AVERAGING = "FED_AVG"
    FEDERATED_PROX = "FED_PROX"
    FEDERATED_NOVA = "FED_NOVA"
    SCAFFOLD = "SCAFFOLD"
    FEDERATED_OPT = "FED_OPT"


class ClientSelectionStrategy(Enum):
    """Client selection strategies."""
    RANDOM = "RANDOM"
    ROUND_ROBIN = "ROUND_ROBIN"
    RESOURCE_BASED = "RESOURCE_BASED"
    DATA_BASED = "DATA_BASED"
    PERFORMANCE_BASED = "PERFORMANCE_BASED"


class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms."""
    NONE = "NONE"
    DIFFERENTIAL_PRIVACY = "DP"
    SECURE_AGGREGATION = "SECURE_AGG"
    HOMOMORPHIC_ENCRYPTION = "HE"
    MULTI_PARTY_COMPUTATION = "MPC"


class ClientStatus(Enum):
    """Client participation status."""
    AVAILABLE = "AVAILABLE"
    TRAINING = "TRAINING"
    UPLOADING = "UPLOADING"
    OFFLINE = "OFFLINE"
    DROPPED = "DROPPED"


@dataclass
class ClientProfile:
    """Profile of a federated learning client."""
    client_id: str
    data_size: int
    compute_capability: float  # FLOPS
    bandwidth_mbps: float
    reliability_score: float = 1.0  # 0-1
    privacy_budget: float = 1.0
    location: Tuple[float, float] = (0.0, 0.0)
    device_type: str = "unknown"
    last_seen: float = field(default_factory=time.time)
    status: ClientStatus = ClientStatus.AVAILABLE
    
    def update_reliability(self, success: bool):
        """Update reliability score based on participation outcome."""
        if success:
            self.reliability_score = min(1.0, self.reliability_score + 0.01)
        else:
            self.reliability_score = max(0.0, self.reliability_score - 0.05)


@dataclass
class ModelUpdate:
    """Represents a model update from a client."""
    client_id: str
    round_number: int
    parameters: Dict[str, np.ndarray]
    data_size: int
    training_loss: float
    training_accuracy: float
    training_time: float
    upload_time: float = field(default_factory=time.time)
    privacy_budget_used: float = 0.0
    
    def get_weight(self, weighting_strategy: str = "data_size") -> float:
        """Get aggregation weight for this update."""
        if weighting_strategy == "data_size":
            return float(self.data_size)
        elif weighting_strategy == "uniform":
            return 1.0
        elif weighting_strategy == "loss_based":
            return 1.0 / (self.training_loss + 1e-8)
        else:
            return 1.0


@dataclass
class FederatedRound:
    """Information about a federated learning round."""
    round_number: int
    selected_clients: List[str]
    start_time: float
    end_time: Optional[float] = None
    global_model_accuracy: Optional[float] = None
    global_model_loss: Optional[float] = None
    client_updates: List[ModelUpdate] = field(default_factory=list)
    aggregation_time: float = 0.0
    
    @property
    def duration(self) -> float:
        """Get round duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def participation_rate(self) -> float:
        """Get client participation rate."""
        if not self.selected_clients:
            return 0.0
        return len(self.client_updates) / len(self.selected_clients)


class DifferentialPrivacy:
    """Differential privacy implementation for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
    def add_noise(self, parameters: Dict[str, np.ndarray], 
                 privacy_budget: float) -> Dict[str, np.ndarray]:
        """Add differential privacy noise to model parameters."""
        try:
            noisy_parameters = {}
            
            # Calculate noise scale
            effective_epsilon = min(self.epsilon, privacy_budget)
            noise_scale = self.sensitivity / effective_epsilon
            
            for name, param in parameters.items():
                # Add Gaussian noise
                noise = np.random.normal(0, noise_scale, param.shape)
                noisy_parameters[name] = param + noise
            
            return noisy_parameters
            
        except Exception as e:
            logger.error(f"Differential privacy noise addition failed: {e}")
            raise
    
    def calculate_privacy_cost(self, num_rounds: int, 
                              participation_rate: float) -> float:
        """Calculate total privacy cost."""
        # Simplified privacy accounting
        return self.epsilon * num_rounds * participation_rate
    
    def is_budget_available(self, client_profile: ClientProfile, 
                           cost: float) -> bool:
        """Check if client has sufficient privacy budget."""
        return client_profile.privacy_budget >= cost


class SecureAggregation:
    """Secure aggregation for federated learning."""
    
    def __init__(self, threshold: int = 3):
        self.threshold = threshold  # Minimum clients for secure aggregation
        self.client_keys: Dict[str, bytes] = {}
        
    def generate_client_key(self, client_id: str) -> bytes:
        """Generate encryption key for client."""
        key = hashlib.sha256(f"{client_id}_{time.time()}".encode()).digest()
        self.client_keys[client_id] = key
        return key
    
    def encrypt_update(self, client_id: str, parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Encrypt model update (simplified implementation)."""
        try:
            if client_id not in self.client_keys:
                raise ValueError(f"No key found for client {client_id}")
            
            # Simplified encryption - in practice, use proper cryptographic libraries
            encrypted_params = {}
            key = self.client_keys[client_id]
            
            for name, param in parameters.items():
                # Simple XOR encryption for demonstration
                param_bytes = param.tobytes()
                key_repeated = (key * (len(param_bytes) // len(key) + 1))[:len(param_bytes)]
                encrypted_bytes = bytes(a ^ b for a, b in zip(param_bytes, key_repeated))
                encrypted_params[name] = {
                    'data': encrypted_bytes,
                    'shape': param.shape,
                    'dtype': str(param.dtype)
                }
            
            return encrypted_params
            
        except Exception as e:
            logger.error(f"Update encryption failed: {e}")
            raise
    
    def aggregate_encrypted_updates(self, encrypted_updates: List[Dict[str, Any]], 
                                   client_ids: List[str]) -> Dict[str, np.ndarray]:
        """Aggregate encrypted updates securely."""
        try:
            if len(encrypted_updates) < self.threshold:
                raise ValueError(f"Insufficient updates for secure aggregation: {len(encrypted_updates)} < {self.threshold}")
            
            # Decrypt and aggregate
            decrypted_updates = []
            
            for i, (encrypted_update, client_id) in enumerate(zip(encrypted_updates, client_ids)):
                if client_id not in self.client_keys:
                    continue
                    
                decrypted_params = {}
                key = self.client_keys[client_id]
                
                for name, encrypted_param in encrypted_update.items():
                    encrypted_bytes = encrypted_param['data']
                    shape = encrypted_param['shape']
                    dtype = encrypted_param['dtype']
                    
                    # Decrypt
                    key_repeated = (key * (len(encrypted_bytes) // len(key) + 1))[:len(encrypted_bytes)]
                    decrypted_bytes = bytes(a ^ b for a, b in zip(encrypted_bytes, key_repeated))
                    
                    # Reconstruct parameter
                    param = np.frombuffer(decrypted_bytes, dtype=dtype).reshape(shape)
                    decrypted_params[name] = param
                
                decrypted_updates.append(decrypted_params)
            
            # Aggregate decrypted updates
            if not decrypted_updates:
                raise ValueError("No valid decrypted updates")
            
            aggregated_params = {}
            for name in decrypted_updates[0].keys():
                params = [update[name] for update in decrypted_updates]
                aggregated_params[name] = np.mean(params, axis=0)
            
            return aggregated_params
            
        except Exception as e:
            logger.error(f"Secure aggregation failed: {e}")
            raise


class FederatedAggregator:
    """Aggregates model updates from federated clients."""
    
    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING):
        self.strategy = strategy
        self.differential_privacy = DifferentialPrivacy()
        self.secure_aggregation = SecureAggregation()
        
    def aggregate_updates(self, updates: List[ModelUpdate], 
                         global_model: Dict[str, np.ndarray],
                         weighting_strategy: str = "data_size") -> Dict[str, np.ndarray]:
        """Aggregate client updates into new global model."""
        try:
            if not updates:
                return global_model
            
            if self.strategy == AggregationStrategy.FEDERATED_AVERAGING:
                return self._federated_averaging(updates, weighting_strategy)
            elif self.strategy == AggregationStrategy.FEDERATED_PROX:
                return self._federated_prox(updates, global_model, weighting_strategy)
            else:
                raise NotImplementedError(f"Aggregation strategy {self.strategy.value} not implemented")
                
        except Exception as e:
            logger.error(f"Update aggregation failed: {e}")
            raise
    
    def _federated_averaging(self, updates: List[ModelUpdate], 
                           weighting_strategy: str) -> Dict[str, np.ndarray]:
        """Implement FedAvg algorithm."""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Calculate weights
        weights = [update.get_weight(weighting_strategy) for update in updates]
        total_weight = sum(weights)
        
        if total_weight == 0:
            raise ValueError("Total weight is zero")
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Aggregate parameters
        aggregated_params = {}
        param_names = updates[0].parameters.keys()
        
        for name in param_names:
            weighted_params = []
            for update, weight in zip(updates, weights):
                if name in update.parameters:
                    weighted_params.append(weight * update.parameters[name])
            
            if weighted_params:
                aggregated_params[name] = sum(weighted_params)
        
        return aggregated_params
    
    def _federated_prox(self, updates: List[ModelUpdate], 
                       global_model: Dict[str, np.ndarray],
                       weighting_strategy: str, mu: float = 0.01) -> Dict[str, np.ndarray]:
        """Implement FedProx algorithm with proximal term."""
        # Start with FedAvg
        fedavg_result = self._federated_averaging(updates, weighting_strategy)
        
        # Apply proximal term
        proximal_params = {}
        for name in fedavg_result.keys():
            if name in global_model:
                proximal_params[name] = (
                    fedavg_result[name] + mu * global_model[name]
                ) / (1 + mu)
            else:
                proximal_params[name] = fedavg_result[name]
        
        return proximal_params


class ClientSelector:
    """Selects clients for federated learning rounds."""
    
    def __init__(self, strategy: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM):
        self.strategy = strategy
        self.selection_history: List[List[str]] = []
        
    def select_clients(self, available_clients: Dict[str, ClientProfile], 
                      num_clients: int, round_number: int) -> List[str]:
        """Select clients for the current round."""
        try:
            if not available_clients:
                return []
            
            available_ids = [cid for cid, profile in available_clients.items() 
                           if profile.status == ClientStatus.AVAILABLE]
            
            if len(available_ids) <= num_clients:
                selected = available_ids
            else:
                if self.strategy == ClientSelectionStrategy.RANDOM:
                    selected = self._random_selection(available_ids, num_clients)
                elif self.strategy == ClientSelectionStrategy.RESOURCE_BASED:
                    selected = self._resource_based_selection(available_clients, num_clients)
                elif self.strategy == ClientSelectionStrategy.DATA_BASED:
                    selected = self._data_based_selection(available_clients, num_clients)
                elif self.strategy == ClientSelectionStrategy.PERFORMANCE_BASED:
                    selected = self._performance_based_selection(available_clients, num_clients)
                else:
                    selected = self._random_selection(available_ids, num_clients)
            
            self.selection_history.append(selected)
            return selected
            
        except Exception as e:
            logger.error(f"Client selection failed: {e}")
            raise
    
    def _random_selection(self, client_ids: List[str], num_clients: int) -> List[str]:
        """Random client selection."""
        return random.sample(client_ids, min(num_clients, len(client_ids)))
    
    def _resource_based_selection(self, clients: Dict[str, ClientProfile], 
                                num_clients: int) -> List[str]:
        """Select clients based on compute resources."""
        # Sort by compute capability and reliability
        sorted_clients = sorted(
            clients.items(),
            key=lambda x: x[1].compute_capability * x[1].reliability_score,
            reverse=True
        )
        
        return [cid for cid, _ in sorted_clients[:num_clients]]
    
    def _data_based_selection(self, clients: Dict[str, ClientProfile], 
                            num_clients: int) -> List[str]:
        """Select clients based on data size."""
        # Sort by data size
        sorted_clients = sorted(
            clients.items(),
            key=lambda x: x[1].data_size,
            reverse=True
        )
        
        return [cid for cid, _ in sorted_clients[:num_clients]]
    
    def _performance_based_selection(self, clients: Dict[str, ClientProfile], 
                                   num_clients: int) -> List[str]:
        """Select clients based on past performance."""
        # Sort by reliability score
        sorted_clients = sorted(
            clients.items(),
            key=lambda x: x[1].reliability_score,
            reverse=True
        )
        
        return [cid for cid, _ in sorted_clients[:num_clients]]
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get client selection statistics."""
        if not self.selection_history:
            return {}
        
        # Count selections per client
        selection_counts = defaultdict(int)
        for round_selections in self.selection_history:
            for client_id in round_selections:
                selection_counts[client_id] += 1
        
        total_rounds = len(self.selection_history)
        avg_clients_per_round = sum(len(selections) for selections in self.selection_history) / total_rounds
        
        return {
            'total_rounds': total_rounds,
            'avg_clients_per_round': avg_clients_per_round,
            'selection_counts': dict(selection_counts),
            'unique_clients_selected': len(selection_counts)
        }


class FederatedServer:
    """Federated learning server coordinator."""
    
    def __init__(self, aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING,
                 selection_strategy: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM,
                 privacy_mechanism: PrivacyMechanism = PrivacyMechanism.NONE):
        self.aggregator = FederatedAggregator(aggregation_strategy)
        self.client_selector = ClientSelector(selection_strategy)
        self.privacy_mechanism = privacy_mechanism
        
        self.clients: Dict[str, ClientProfile] = {}
        self.global_model: Dict[str, np.ndarray] = {}
        self.rounds: List[FederatedRound] = []
        self.current_round: Optional[FederatedRound] = None
        
        self._server_running = False
        self._round_timeout = 300  # 5 minutes
        
    def register_client(self, client_profile: ClientProfile):
        """Register a new federated learning client."""
        self.clients[client_profile.client_id] = client_profile
        
        # Generate security keys if needed
        if self.privacy_mechanism == PrivacyMechanism.SECURE_AGGREGATION:
            self.aggregator.secure_aggregation.generate_client_key(client_profile.client_id)
        
        logger.info(f"Registered client: {client_profile.client_id}")
    
    def unregister_client(self, client_id: str):
        """Unregister a client."""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Unregistered client: {client_id}")
    
    def initialize_global_model(self, model_parameters: Dict[str, np.ndarray]):
        """Initialize the global model."""
        self.global_model = copy.deepcopy(model_parameters)
        logger.info("Initialized global model")
    
    def start_round(self, num_clients: int = 10) -> Dict[str, Any]:
        """Start a new federated learning round."""
        try:
            if self.current_round and not self.current_round.end_time:
                raise RuntimeError("Previous round not completed")
            
            round_number = len(self.rounds) + 1
            
            # Select clients
            selected_clients = self.client_selector.select_clients(
                self.clients, num_clients, round_number
            )
            
            if not selected_clients:
                raise ValueError("No clients available for selection")
            
            # Create new round
            self.current_round = FederatedRound(
                round_number=round_number,
                selected_clients=selected_clients,
                start_time=time.time()
            )
            
            # Update client status
            for client_id in selected_clients:
                if client_id in self.clients:
                    self.clients[client_id].status = ClientStatus.TRAINING
            
            logger.info(f"Started round {round_number} with {len(selected_clients)} clients")
            
            return {
                'round_number': round_number,
                'selected_clients': selected_clients,
                'global_model': copy.deepcopy(self.global_model),
                'start_time': self.current_round.start_time
            }
            
        except Exception as e:
            logger.error(f"Round start failed: {e}")
            raise
    
    def submit_update(self, update: ModelUpdate) -> bool:
        """Submit a client update for the current round."""
        try:
            if not self.current_round:
                raise RuntimeError("No active round")
            
            if update.client_id not in self.current_round.selected_clients:
                raise ValueError(f"Client {update.client_id} not selected for current round")
            
            # Check if update already submitted
            existing_updates = [u for u in self.current_round.client_updates 
                              if u.client_id == update.client_id]
            if existing_updates:
                logger.warning(f"Update from {update.client_id} already submitted")
                return False
            
            # Apply privacy mechanisms
            if self.privacy_mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
                client_profile = self.clients[update.client_id]
                privacy_cost = 0.1  # Simplified cost calculation
                
                if self.aggregator.differential_privacy.is_budget_available(client_profile, privacy_cost):
                    update.parameters = self.aggregator.differential_privacy.add_noise(
                        update.parameters, client_profile.privacy_budget
                    )
                    client_profile.privacy_budget -= privacy_cost
                    update.privacy_budget_used = privacy_cost
                else:
                    logger.warning(f"Insufficient privacy budget for client {update.client_id}")
                    return False
            
            # Add update to round
            self.current_round.client_updates.append(update)
            
            # Update client status
            if update.client_id in self.clients:
                self.clients[update.client_id].status = ClientStatus.UPLOADING
                self.clients[update.client_id].last_seen = time.time()
            
            logger.info(f"Received update from client {update.client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Update submission failed: {e}")
            return False
    
    def complete_round(self, min_updates: int = 1) -> Dict[str, Any]:
        """Complete the current federated learning round."""
        try:
            if not self.current_round:
                raise RuntimeError("No active round")
            
            if len(self.current_round.client_updates) < min_updates:
                raise ValueError(f"Insufficient updates: {len(self.current_round.client_updates)} < {min_updates}")
            
            start_time = time.time()
            
            # Aggregate updates
            if self.privacy_mechanism == PrivacyMechanism.SECURE_AGGREGATION:
                # Secure aggregation
                encrypted_updates = []
                client_ids = []
                
                for update in self.current_round.client_updates:
                    encrypted_update = self.aggregator.secure_aggregation.encrypt_update(
                        update.client_id, update.parameters
                    )
                    encrypted_updates.append(encrypted_update)
                    client_ids.append(update.client_id)
                
                self.global_model = self.aggregator.secure_aggregation.aggregate_encrypted_updates(
                    encrypted_updates, client_ids
                )
            else:
                # Standard aggregation
                self.global_model = self.aggregator.aggregate_updates(
                    self.current_round.client_updates, self.global_model
                )
            
            aggregation_time = time.time() - start_time
            
            # Complete round
            self.current_round.end_time = time.time()
            self.current_round.aggregation_time = aggregation_time
            
            # Calculate round statistics
            avg_loss = np.mean([u.training_loss for u in self.current_round.client_updates])
            avg_accuracy = np.mean([u.training_accuracy for u in self.current_round.client_updates])
            
            self.current_round.global_model_loss = avg_loss
            self.current_round.global_model_accuracy = avg_accuracy
            
            # Update client reliability scores
            for client_id in self.current_round.selected_clients:
                if client_id in self.clients:
                    participated = any(u.client_id == client_id for u in self.current_round.client_updates)
                    self.clients[client_id].update_reliability(participated)
                    self.clients[client_id].status = ClientStatus.AVAILABLE
            
            # Archive round
            self.rounds.append(self.current_round)
            completed_round = self.current_round
            self.current_round = None
            
            logger.info(f"Completed round {completed_round.round_number}")
            
            return {
                'round_number': completed_round.round_number,
                'participation_rate': completed_round.participation_rate,
                'aggregation_time': aggregation_time,
                'global_model_loss': avg_loss,
                'global_model_accuracy': avg_accuracy,
                'global_model': copy.deepcopy(self.global_model)
            }
            
        except Exception as e:
            logger.error(f"Round completion failed: {e}")
            raise
    
    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        """Get information about a specific client."""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found")
        
        client = self.clients[client_id]
        
        # Calculate participation statistics
        total_selections = sum(1 for round_info in self.rounds 
                             if client_id in round_info.selected_clients)
        total_participations = sum(1 for round_info in self.rounds 
                                 if any(u.client_id == client_id for u in round_info.client_updates))
        
        participation_rate = total_participations / total_selections if total_selections > 0 else 0
        
        return {
            'client_id': client.client_id,
            'data_size': client.data_size,
            'compute_capability': client.compute_capability,
            'reliability_score': client.reliability_score,
            'privacy_budget': client.privacy_budget,
            'status': client.status.value,
            'last_seen': client.last_seen,
            'total_selections': total_selections,
            'total_participations': total_participations,
            'participation_rate': participation_rate
        }
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get federated learning server statistics."""
        total_clients = len(self.clients)
        active_clients = sum(1 for c in self.clients.values() 
                           if c.status == ClientStatus.AVAILABLE)
        
        if self.rounds:
            avg_participation = np.mean([r.participation_rate for r in self.rounds])
            avg_round_duration = np.mean([r.duration for r in self.rounds if r.end_time])
            latest_accuracy = self.rounds[-1].global_model_accuracy
            latest_loss = self.rounds[-1].global_model_loss
        else:
            avg_participation = 0
            avg_round_duration = 0
            latest_accuracy = None
            latest_loss = None
        
        return {
            'total_clients': total_clients,
            'active_clients': active_clients,
            'completed_rounds': len(self.rounds),
            'current_round_active': self.current_round is not None,
            'avg_participation_rate': avg_participation,
            'avg_round_duration': avg_round_duration,
            'latest_global_accuracy': latest_accuracy,
            'latest_global_loss': latest_loss,
            'aggregation_strategy': self.aggregator.strategy.value,
            'privacy_mechanism': self.privacy_mechanism.value
        }


class FederatedClient:
    """Federated learning client implementation."""
    
    def __init__(self, client_id: str, data_size: int, 
                 compute_capability: float = 1.0, bandwidth_mbps: float = 10.0):
        self.profile = ClientProfile(
            client_id=client_id,
            data_size=data_size,
            compute_capability=compute_capability,
            bandwidth_mbps=bandwidth_mbps
        )
        self.local_model: Dict[str, np.ndarray] = {}
        self.training_history: List[Dict[str, Any]] = []
        
    def receive_global_model(self, global_model: Dict[str, np.ndarray]):
        """Receive global model from server."""
        self.local_model = copy.deepcopy(global_model)
        logger.info(f"Client {self.profile.client_id} received global model")
    
    def train_local_model(self, num_epochs: int = 5, 
                         learning_rate: float = 0.01) -> ModelUpdate:
        """Simulate local model training."""
        try:
            start_time = time.time()
            
            # Simulate training process
            initial_loss = random.uniform(0.5, 2.0)
            initial_accuracy = random.uniform(0.6, 0.9)
            
            # Simulate improvement over epochs
            final_loss = initial_loss * (0.9 ** num_epochs)
            final_accuracy = min(0.99, initial_accuracy + 0.01 * num_epochs)
            
            # Simulate parameter updates
            updated_parameters = {}
            for name, param in self.local_model.items():
                # Add small random updates
                noise = np.random.normal(0, 0.01, param.shape)
                updated_parameters[name] = param + noise * learning_rate
            
            training_time = time.time() - start_time
            
            # Create model update
            update = ModelUpdate(
                client_id=self.profile.client_id,
                round_number=0,  # Will be set by server
                parameters=updated_parameters,
                data_size=self.profile.data_size,
                training_loss=final_loss,
                training_accuracy=final_accuracy,
                training_time=training_time
            )
            
            # Record training history
            self.training_history.append({
                'epochs': num_epochs,
                'learning_rate': learning_rate,
                'final_loss': final_loss,
                'final_accuracy': final_accuracy,
                'training_time': training_time,
                'timestamp': time.time()
            })
            
            logger.info(f"Client {self.profile.client_id} completed training")
            return update
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
            raise
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client training statistics."""
        if not self.training_history:
            return {'training_rounds': 0}
        
        avg_loss = np.mean([h['final_loss'] for h in self.training_history])
        avg_accuracy = np.mean([h['final_accuracy'] for h in self.training_history])
        avg_training_time = np.mean([h['training_time'] for h in self.training_history])
        
        return {
            'client_id': self.profile.client_id,
            'training_rounds': len(self.training_history),
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'avg_training_time': avg_training_time,
            'data_size': self.profile.data_size,
            'compute_capability': self.profile.compute_capability
        }


class FederatedLearningManager:
    """Main interface for federated learning capabilities."""
    
    def __init__(self):
        self.servers: Dict[str, FederatedServer] = {}
        self.clients: Dict[str, FederatedClient] = {}
        self.experiments: Dict[str, Dict[str, Any]] = {}
        
    def create_server(self, server_id: str, 
                     aggregation_strategy: str = "federated_averaging",
                     selection_strategy: str = "random",
                     privacy_mechanism: str = "none") -> str:
        """Create federated learning server."""
        try:
            agg_strategy = AggregationStrategy(aggregation_strategy.upper())
            sel_strategy = ClientSelectionStrategy(selection_strategy.upper())
            priv_mechanism = PrivacyMechanism(privacy_mechanism.upper())
            
            server = FederatedServer(agg_strategy, sel_strategy, priv_mechanism)
            self.servers[server_id] = server
            
            logger.info(f"Created federated server: {server_id}")
            return server_id
            
        except Exception as e:
            logger.error(f"Server creation failed: {e}")
            raise
    
    def create_client(self, client_id: str, data_size: int,
                     compute_capability: float = 1.0, 
                     bandwidth_mbps: float = 10.0) -> str:
        """Create federated learning client."""
        try:
            client = FederatedClient(client_id, data_size, compute_capability, bandwidth_mbps)
            self.clients[client_id] = client
            
            logger.info(f"Created federated client: {client_id}")
            return client_id
            
        except Exception as e:
            logger.error(f"Client creation failed: {e}")
            raise
    
    def register_client_to_server(self, server_id: str, client_id: str):
        """Register client to server."""
        try:
            if server_id not in self.servers:
                raise ValueError(f"Server {server_id} not found")
            if client_id not in self.clients:
                raise ValueError(f"Client {client_id} not found")
            
            server = self.servers[server_id]
            client = self.clients[client_id]
            
            server.register_client(client.profile)
            
        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            raise
    
    def run_federated_experiment(self, server_id: str, 
                               num_rounds: int = 10,
                               clients_per_round: int = 5,
                               local_epochs: int = 5) -> Dict[str, Any]:
        """Run complete federated learning experiment."""
        try:
            if server_id not in self.servers:
                raise ValueError(f"Server {server_id} not found")
            
            server = self.servers[server_id]
            
            # Initialize global model (dummy model for demonstration)
            dummy_model = {
                'layer1_weights': np.random.normal(0, 0.1, (10, 5)),
                'layer1_bias': np.zeros(5),
                'layer2_weights': np.random.normal(0, 0.1, (5, 1)),
                'layer2_bias': np.zeros(1)
            }
            server.initialize_global_model(dummy_model)
            
            experiment_results = {
                'server_id': server_id,
                'num_rounds': num_rounds,
                'clients_per_round': clients_per_round,
                'local_epochs': local_epochs,
                'rounds': [],
                'start_time': time.time()
            }
            
            for round_num in range(num_rounds):
                logger.info(f"Starting federated round {round_num + 1}/{num_rounds}")
                
                # Start round
                round_info = server.start_round(clients_per_round)
                selected_clients = round_info['selected_clients']
                
                # Simulate client training
                updates = []
                for client_id in selected_clients:
                    if client_id in self.clients:
                        client = self.clients[client_id]
                        
                        # Send global model to client
                        client.receive_global_model(round_info['global_model'])
                        
                        # Train local model
                        update = client.train_local_model(local_epochs)
                        update.round_number = round_info['round_number']
                        
                        # Submit update to server
                        success = server.submit_update(update)
                        if success:
                            updates.append(update)
                
                # Complete round
                if updates:
                    round_result = server.complete_round(min_updates=1)
                    experiment_results['rounds'].append(round_result)
                    
                    logger.info(f"Round {round_num + 1} completed: "
                              f"Accuracy={round_result['global_model_accuracy']:.4f}, "
                              f"Loss={round_result['global_model_loss']:.4f}")
                else:
                    logger.warning(f"Round {round_num + 1} failed: no valid updates")
            
            experiment_results['end_time'] = time.time()
            experiment_results['duration'] = experiment_results['end_time'] - experiment_results['start_time']
            
            # Store experiment
            experiment_id = f"{server_id}_exp_{int(time.time())}"
            self.experiments[experiment_id] = experiment_results
            
            return experiment_results
            
        except Exception as e:
            logger.error(f"Federated experiment failed: {e}")
            raise
    
    def get_server_status(self, server_id: str) -> Dict[str, Any]:
        """Get server status and statistics."""
        if server_id not in self.servers:
            raise ValueError(f"Server {server_id} not found")
        
        return self.servers[server_id].get_server_stats()
    
    def get_client_status(self, client_id: str) -> Dict[str, Any]:
        """Get client status and statistics."""
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found")
        
        return self.clients[client_id].get_client_stats()
    
    def get_federated_learning_info(self) -> Dict[str, Any]:
        """Get information about federated learning capabilities."""
        return {
            'aggregation_strategies': [strategy.value for strategy in AggregationStrategy],
            'selection_strategies': [strategy.value for strategy in ClientSelectionStrategy],
            'privacy_mechanisms': [mechanism.value for mechanism in PrivacyMechanism],
            'active_servers': len(self.servers),
            'active_clients': len(self.clients),
            'completed_experiments': len(self.experiments),
            'server_ids': list(self.servers.keys()),
            'client_ids': list(self.clients.keys())
        }


# Utility functions
def create_iid_data_distribution(total_samples: int, num_clients: int) -> List[int]:
    """Create IID data distribution across clients."""
    base_size = total_samples // num_clients
    remainder = total_samples % num_clients
    
    distribution = [base_size] * num_clients
    for i in range(remainder):
        distribution[i] += 1
    
    return distribution


def create_non_iid_data_distribution(total_samples: int, num_clients: int, 
                                    alpha: float = 0.5) -> List[int]:
    """Create non-IID data distribution using Dirichlet distribution."""
    # Generate Dirichlet distribution
    proportions = np.random.dirichlet([alpha] * num_clients)
    
    # Convert to sample counts
    distribution = [int(prop * total_samples) for prop in proportions]
    
    # Adjust for rounding errors
    diff = total_samples - sum(distribution)
    for i in range(abs(diff)):
        if diff > 0:
            distribution[i % num_clients] += 1
        else:
            if distribution[i % num_clients] > 0:
                distribution[i % num_clients] -= 1
    
    return distribution


# Export main classes and functions
__all__ = [
    'FederatedLearningManager',
    'FederatedServer',
    'FederatedClient',
    'FederatedAggregator',
    'ClientSelector',
    'DifferentialPrivacy',
    'SecureAggregation',
    'ClientProfile',
    'ModelUpdate',
    'FederatedRound',
    'AggregationStrategy',
    'ClientSelectionStrategy',
    'PrivacyMechanism',
    'ClientStatus',
    'create_iid_data_distribution',
    'create_non_iid_data_distribution'
]