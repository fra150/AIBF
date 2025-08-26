"""Performance tests for core module components."""

import pytest
import torch
import numpy as np
import time
from typing import Dict, Any, List
from unittest.mock import patch

# Import performance testing utilities
from . import (
    benchmark, PERF_CONFIG, create_test_data, create_large_model,
    save_benchmark_results
)

# Import core modules
from core.neural_networks import (
    FeedForward, CNN, RNN, Attention, NeuralNetworkConfig
)
from core.transformers import (
    TransformerEncoder, TransformerDecoder, MultiHeadAttention,
    PositionalEncoding, TransformerConfig
)
from core.reinforcement_learning import (
    ActorCritic, ExperienceReplay, RLAgent, Environment,
    RLConfig, PolicyNetwork, ValueNetwork
)
from core.tensor_ops import (
    TensorOperations, MatrixOperations, ActivationFunctions,
    LossCalculator, TensorConfig
)
from core.model_utils import (
    ModelManager, CheckpointManager, ModelOptimizer,
    ModelEvaluator, ModelConfig
)


class TestNeuralNetworkPerformance:
    """Performance tests for neural network components."""
    
    @pytest.mark.performance
    def test_feedforward_performance(self):
        """Test FeedForward network performance across different sizes."""
        results = {}
        
        # Test different network sizes
        test_configs = [
            {"input_size": 128, "hidden_sizes": [64, 32], "output_size": 10, "name": "small"},
            {"input_size": 512, "hidden_sizes": [256, 128, 64], "output_size": 50, "name": "medium"},
            {"input_size": 2048, "hidden_sizes": [1024, 512, 256], "output_size": 100, "name": "large"}
        ]
        
        for config in test_configs:
            with benchmark.benchmark_context(f"feedforward_{config['name']}"):
                # Create network
                nn_config = NeuralNetworkConfig(
                    input_size=config["input_size"],
                    hidden_sizes=config["hidden_sizes"],
                    output_size=config["output_size"],
                    activation="relu",
                    dropout_rate=0.1
                )
                
                network = FeedForward(nn_config)
                
                # Test different batch sizes
                batch_results = {}
                for batch_size in [PERF_CONFIG.small_batch_size, PERF_CONFIG.medium_batch_size, PERF_CONFIG.large_batch_size]:
                    # Create test data
                    input_data = create_test_data(batch_size, config["input_size"])
                    
                    # Benchmark forward pass
                    def forward_pass():
                        with torch.no_grad():
                            return network(input_data)
                    
                    timing_results = benchmark.time_function(forward_pass)
                    memory_results = benchmark.profile_memory_usage(forward_pass)
                    
                    batch_results[f"batch_{batch_size}"] = {
                        "timing": timing_results,
                        "memory": memory_results
                    }
                    
                    # Check performance thresholds
                    violations = benchmark.check_performance_thresholds({
                        "memory_mb": {"max": memory_results["memory_increase_mb"]}
                    })
                    
                    if violations:
                        pytest.fail(f"Performance violations in {config['name']} batch {batch_size}: {violations}")
                
                results[config["name"]] = batch_results
        
        # Save results
        save_benchmark_results("feedforward_performance", results)
        
        # Verify performance scaling
        small_time = results["small"]["batch_1"]["timing"]["mean_time"]
        large_time = results["large"]["batch_1"]["timing"]["mean_time"]
        
        # Large network should not be more than 100x slower than small
        assert large_time / small_time < 100, f"Performance scaling issue: {large_time / small_time}x slower"
    
    @pytest.mark.performance
    def test_cnn_performance(self):
        """Test CNN performance with different input sizes."""
        results = {}
        
        # Test different CNN configurations
        test_configs = [
            {"channels": [3, 16, 32], "kernel_sizes": [3, 3], "input_size": (3, 64, 64), "name": "small"},
            {"channels": [3, 32, 64, 128], "kernel_sizes": [3, 3, 3], "input_size": (3, 128, 128), "name": "medium"},
            {"channels": [3, 64, 128, 256, 512], "kernel_sizes": [3, 3, 3, 3], "input_size": (3, 224, 224), "name": "large"}
        ]
        
        for config in test_configs:
            with benchmark.benchmark_context(f"cnn_{config['name']}"):
                # Create CNN
                nn_config = NeuralNetworkConfig(
                    input_size=config["input_size"],
                    channels=config["channels"],
                    kernel_sizes=config["kernel_sizes"],
                    activation="relu",
                    dropout_rate=0.1
                )
                
                cnn = CNN(nn_config)
                
                # Test different batch sizes
                batch_results = {}
                for batch_size in [1, 4, 8]:  # Smaller batches for CNN due to memory
                    # Create test data
                    input_data = create_test_data(batch_size, *config["input_size"])
                    
                    # Benchmark forward pass
                    def forward_pass():
                        with torch.no_grad():
                            return cnn(input_data)
                    
                    timing_results = benchmark.time_function(forward_pass)
                    memory_results = benchmark.profile_memory_usage(forward_pass)
                    
                    batch_results[f"batch_{batch_size}"] = {
                        "timing": timing_results,
                        "memory": memory_results
                    }
                
                results[config["name"]] = batch_results
        
        save_benchmark_results("cnn_performance", results)
    
    @pytest.mark.performance
    def test_rnn_performance(self):
        """Test RNN performance with different sequence lengths."""
        results = {}
        
        # Test different RNN configurations
        test_configs = [
            {"input_size": 64, "hidden_size": 128, "num_layers": 1, "seq_len": 50, "name": "small"},
            {"input_size": 128, "hidden_size": 256, "num_layers": 2, "seq_len": 100, "name": "medium"},
            {"input_size": 256, "hidden_size": 512, "num_layers": 3, "seq_len": 200, "name": "large"}
        ]
        
        for config in test_configs:
            with benchmark.benchmark_context(f"rnn_{config['name']}"):
                # Create RNN
                nn_config = NeuralNetworkConfig(
                    input_size=config["input_size"],
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    rnn_type="LSTM",
                    dropout_rate=0.1
                )
                
                rnn = RNN(nn_config)
                
                # Test different batch sizes
                batch_results = {}
                for batch_size in [1, 4, 8]:
                    # Create test data (batch_size, seq_len, input_size)
                    input_data = create_test_data(batch_size, config["seq_len"], config["input_size"])
                    
                    # Benchmark forward pass
                    def forward_pass():
                        with torch.no_grad():
                            return rnn(input_data)
                    
                    timing_results = benchmark.time_function(forward_pass)
                    memory_results = benchmark.profile_memory_usage(forward_pass)
                    
                    batch_results[f"batch_{batch_size}"] = {
                        "timing": timing_results,
                        "memory": memory_results
                    }
                
                results[config["name"]] = batch_results
        
        save_benchmark_results("rnn_performance", results)
    
    @pytest.mark.performance
    def test_attention_performance(self):
        """Test Attention mechanism performance."""
        results = {}
        
        # Test different attention configurations
        test_configs = [
            {"embed_dim": 128, "num_heads": 4, "seq_len": 50, "name": "small"},
            {"embed_dim": 256, "num_heads": 8, "seq_len": 100, "name": "medium"},
            {"embed_dim": 512, "num_heads": 16, "seq_len": 200, "name": "large"}
        ]
        
        for config in test_configs:
            with benchmark.benchmark_context(f"attention_{config['name']}"):
                # Create Attention
                nn_config = NeuralNetworkConfig(
                    embed_dim=config["embed_dim"],
                    num_heads=config["num_heads"],
                    dropout_rate=0.1
                )
                
                attention = Attention(nn_config)
                
                # Test different batch sizes
                batch_results = {}
                for batch_size in [1, 4, 8]:
                    # Create test data (batch_size, seq_len, embed_dim)
                    query = create_test_data(batch_size, config["seq_len"], config["embed_dim"])
                    key = create_test_data(batch_size, config["seq_len"], config["embed_dim"])
                    value = create_test_data(batch_size, config["seq_len"], config["embed_dim"])
                    
                    # Benchmark forward pass
                    def forward_pass():
                        with torch.no_grad():
                            return attention(query, key, value)
                    
                    timing_results = benchmark.time_function(forward_pass)
                    memory_results = benchmark.profile_memory_usage(forward_pass)
                    
                    batch_results[f"batch_{batch_size}"] = {
                        "timing": timing_results,
                        "memory": memory_results
                    }
                
                results[config["name"]] = batch_results
        
        save_benchmark_results("attention_performance", results)


class TestTransformerPerformance:
    """Performance tests for transformer components."""
    
    @pytest.mark.performance
    def test_transformer_encoder_performance(self):
        """Test TransformerEncoder performance."""
        results = {}
        
        # Test different transformer configurations
        test_configs = [
            {"d_model": 256, "num_heads": 4, "num_layers": 2, "seq_len": 50, "name": "small"},
            {"d_model": 512, "num_heads": 8, "num_layers": 4, "seq_len": 100, "name": "medium"},
            {"d_model": 768, "num_heads": 12, "num_layers": 6, "seq_len": 200, "name": "large"}
        ]
        
        for config in test_configs:
            with benchmark.benchmark_context(f"transformer_encoder_{config['name']}"):
                # Create TransformerEncoder
                transformer_config = TransformerConfig(
                    d_model=config["d_model"],
                    num_heads=config["num_heads"],
                    num_layers=config["num_layers"],
                    d_ff=config["d_model"] * 4,
                    dropout_rate=0.1
                )
                
                encoder = TransformerEncoder(transformer_config)
                
                # Test different batch sizes
                batch_results = {}
                for batch_size in [1, 4, 8]:
                    # Create test data (batch_size, seq_len, d_model)
                    input_data = create_test_data(batch_size, config["seq_len"], config["d_model"])
                    
                    # Benchmark forward pass
                    def forward_pass():
                        with torch.no_grad():
                            return encoder(input_data)
                    
                    timing_results = benchmark.time_function(forward_pass)
                    memory_results = benchmark.profile_memory_usage(forward_pass)
                    
                    batch_results[f"batch_{batch_size}"] = {
                        "timing": timing_results,
                        "memory": memory_results
                    }
                
                results[config["name"]] = batch_results
        
        save_benchmark_results("transformer_encoder_performance", results)
    
    @pytest.mark.performance
    def test_multihead_attention_performance(self):
        """Test MultiHeadAttention performance with different configurations."""
        results = {}
        
        # Test different attention configurations
        test_configs = [
            {"d_model": 256, "num_heads": 4, "seq_len": 50, "name": "small"},
            {"d_model": 512, "num_heads": 8, "seq_len": 100, "name": "medium"},
            {"d_model": 768, "num_heads": 12, "seq_len": 200, "name": "large"},
            {"d_model": 1024, "num_heads": 16, "seq_len": 500, "name": "xlarge"}
        ]
        
        for config in test_configs:
            with benchmark.benchmark_context(f"multihead_attention_{config['name']}"):
                # Create MultiHeadAttention
                transformer_config = TransformerConfig(
                    d_model=config["d_model"],
                    num_heads=config["num_heads"],
                    dropout_rate=0.1
                )
                
                attention = MultiHeadAttention(transformer_config)
                
                # Test different batch sizes
                batch_results = {}
                for batch_size in [1, 2, 4]:  # Smaller batches for large sequences
                    # Create test data
                    query = create_test_data(batch_size, config["seq_len"], config["d_model"])
                    key = create_test_data(batch_size, config["seq_len"], config["d_model"])
                    value = create_test_data(batch_size, config["seq_len"], config["d_model"])
                    
                    # Benchmark forward pass
                    def forward_pass():
                        with torch.no_grad():
                            return attention(query, key, value)
                    
                    timing_results = benchmark.time_function(forward_pass)
                    memory_results = benchmark.profile_memory_usage(forward_pass)
                    
                    batch_results[f"batch_{batch_size}"] = {
                        "timing": timing_results,
                        "memory": memory_results
                    }
                    
                    # Check quadratic scaling for attention
                    if config["name"] == "xlarge" and batch_size == 1:
                        # Attention should scale quadratically with sequence length
                        expected_complexity = (config["seq_len"] ** 2) * config["d_model"]
                        actual_time = timing_results["mean_time"]
                        
                        # Log complexity metrics
                        print(f"Attention complexity - Seq len: {config['seq_len']}, "
                              f"Time: {actual_time:.4f}s, Expected complexity: {expected_complexity}")
                
                results[config["name"]] = batch_results
        
        save_benchmark_results("multihead_attention_performance", results)


class TestReinforcementLearningPerformance:
    """Performance tests for reinforcement learning components."""
    
    @pytest.mark.performance
    def test_actor_critic_performance(self):
        """Test ActorCritic performance."""
        results = {}
        
        # Test different AC configurations
        test_configs = [
            {"state_dim": 64, "action_dim": 4, "hidden_dim": 128, "name": "small"},
            {"state_dim": 256, "action_dim": 16, "hidden_dim": 512, "name": "medium"},
            {"state_dim": 1024, "action_dim": 64, "hidden_dim": 2048, "name": "large"}
        ]
        
        for config in test_configs:
            with benchmark.benchmark_context(f"actor_critic_{config['name']}"):
                # Create ActorCritic
                rl_config = RLConfig(
                    state_dim=config["state_dim"],
                    action_dim=config["action_dim"],
                    hidden_dim=config["hidden_dim"],
                    learning_rate=0.001
                )
                
                actor_critic = ActorCritic(rl_config)
                
                # Test different batch sizes
                batch_results = {}
                for batch_size in [1, 16, 64]:
                    # Create test data
                    states = create_test_data(batch_size, config["state_dim"])
                    
                    # Benchmark forward pass
                    def forward_pass():
                        with torch.no_grad():
                            return actor_critic(states)
                    
                    timing_results = benchmark.time_function(forward_pass)
                    memory_results = benchmark.profile_memory_usage(forward_pass)
                    
                    batch_results[f"batch_{batch_size}"] = {
                        "timing": timing_results,
                        "memory": memory_results
                    }
                
                results[config["name"]] = batch_results
        
        save_benchmark_results("actor_critic_performance", results)
    
    @pytest.mark.performance
    def test_experience_replay_performance(self):
        """Test ExperienceReplay performance with large buffers."""
        results = {}
        
        # Test different buffer sizes
        buffer_sizes = [1000, 10000, 100000]
        
        for buffer_size in buffer_sizes:
            with benchmark.benchmark_context(f"experience_replay_{buffer_size}"):
                # Create ExperienceReplay
                rl_config = RLConfig(
                    state_dim=64,
                    action_dim=4,
                    buffer_size=buffer_size
                )
                
                replay_buffer = ExperienceReplay(rl_config)
                
                # Fill buffer with experiences
                def fill_buffer():
                    for i in range(buffer_size):
                        state = torch.randn(64)
                        action = torch.randint(0, 4, (1,))
                        reward = torch.randn(1)
                        next_state = torch.randn(64)
                        done = torch.randint(0, 2, (1,)).bool()
                        
                        replay_buffer.add_experience(state, action, reward, next_state, done)
                
                # Benchmark buffer filling
                timing_results = benchmark.time_function(fill_buffer)
                memory_results = benchmark.profile_memory_usage(fill_buffer)
                
                # Benchmark sampling
                def sample_batch():
                    return replay_buffer.sample_batch(32)
                
                sampling_timing = benchmark.time_function(sample_batch)
                sampling_memory = benchmark.profile_memory_usage(sample_batch)
                
                results[f"buffer_{buffer_size}"] = {
                    "filling": {
                        "timing": timing_results,
                        "memory": memory_results
                    },
                    "sampling": {
                        "timing": sampling_timing,
                        "memory": sampling_memory
                    }
                }
        
        save_benchmark_results("experience_replay_performance", results)


class TestTensorOperationsPerformance:
    """Performance tests for tensor operations."""
    
    @pytest.mark.performance
    def test_matrix_operations_performance(self):
        """Test matrix operations performance."""
        results = {}
        
        # Test different matrix sizes
        matrix_sizes = [
            (128, 128),
            (512, 512),
            (1024, 1024),
            (2048, 2048)
        ]
        
        tensor_config = TensorConfig(device="cpu", dtype=torch.float32)
        matrix_ops = MatrixOperations(tensor_config)
        
        for size in matrix_sizes:
            with benchmark.benchmark_context(f"matrix_ops_{size[0]}x{size[1]}"):
                # Create test matrices
                matrix_a = torch.randn(size)
                matrix_b = torch.randn(size)
                
                # Test different operations
                operations = {
                    "multiply": lambda: matrix_ops.multiply(matrix_a, matrix_b),
                    "transpose": lambda: matrix_ops.transpose(matrix_a),
                    "inverse": lambda: matrix_ops.inverse(matrix_a + torch.eye(size[0]) * 0.1),  # Add regularization
                    "eigenvalues": lambda: matrix_ops.eigenvalues(matrix_a @ matrix_a.T)  # Symmetric matrix
                }
                
                size_results = {}
                for op_name, op_func in operations.items():
                    try:
                        timing_results = benchmark.time_function(op_func)
                        memory_results = benchmark.profile_memory_usage(op_func)
                        
                        size_results[op_name] = {
                            "timing": timing_results,
                            "memory": memory_results
                        }
                    except Exception as e:
                        size_results[op_name] = {"error": str(e)}
                
                results[f"{size[0]}x{size[1]}"] = size_results
        
        save_benchmark_results("matrix_operations_performance", results)
    
    @pytest.mark.performance
    def test_activation_functions_performance(self):
        """Test activation functions performance."""
        results = {}
        
        tensor_config = TensorConfig(device="cpu", dtype=torch.float32)
        activations = ActivationFunctions(tensor_config)
        
        # Test different tensor sizes
        tensor_sizes = [
            (1000,),
            (10000,),
            (100000,),
            (1000000,)
        ]
        
        activation_functions = [
            "relu", "sigmoid", "tanh", "gelu", "swish"
        ]
        
        for size in tensor_sizes:
            with benchmark.benchmark_context(f"activations_{size[0]}"):
                # Create test tensor
                input_tensor = torch.randn(size)
                
                size_results = {}
                for activation_name in activation_functions:
                    def activation_func():
                        return activations.apply_activation(input_tensor, activation_name)
                    
                    timing_results = benchmark.time_function(activation_func)
                    memory_results = benchmark.profile_memory_usage(activation_func)
                    
                    size_results[activation_name] = {
                        "timing": timing_results,
                        "memory": memory_results
                    }
                
                results[f"size_{size[0]}"] = size_results
        
        save_benchmark_results("activation_functions_performance", results)


class TestModelUtilsPerformance:
    """Performance tests for model utilities."""
    
    @pytest.mark.performance
    def test_model_manager_performance(self):
        """Test ModelManager performance with different model sizes."""
        results = {}
        
        # Test different model sizes
        model_configs = [
            {"params": PERF_CONFIG.small_model_params, "name": "small"},
            {"params": PERF_CONFIG.medium_model_params, "name": "medium"},
            {"params": PERF_CONFIG.large_model_params, "name": "large"}
        ]
        
        for config in model_configs:
            with benchmark.benchmark_context(f"model_manager_{config['name']}"):
                # Create model
                model = create_large_model(config["params"])
                
                # Create ModelManager
                model_config = ModelConfig(
                    model_name=f"test_model_{config['name']}",
                    model_path=str(PERF_CONFIG.temp_dir / f"model_{config['name']}.pt")
                )
                
                model_manager = ModelManager(model_config)
                
                # Test operations
                operations = {}
                
                # Test model saving
                def save_model():
                    return model_manager.save_model(model)
                
                operations["save"] = {
                    "timing": benchmark.time_function(save_model),
                    "memory": benchmark.profile_memory_usage(save_model)
                }
                
                # Test model loading
                def load_model():
                    return model_manager.load_model()
                
                operations["load"] = {
                    "timing": benchmark.time_function(load_model),
                    "memory": benchmark.profile_memory_usage(load_model)
                }
                
                # Test model optimization
                optimizer = torch.optim.Adam(model.parameters())
                
                def optimize_step():
                    # Simulate training step
                    input_data = torch.randn(32, model[0].in_features)
                    target = torch.randn(32, 1)
                    
                    optimizer.zero_grad()
                    output = model(input_data)
                    loss = torch.nn.functional.mse_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    return loss.item()
                
                operations["optimize"] = {
                    "timing": benchmark.time_function(optimize_step),
                    "memory": benchmark.profile_memory_usage(optimize_step)
                }
                
                results[config["name"]] = operations
        
        save_benchmark_results("model_manager_performance", results)
    
    @pytest.mark.performance
    def test_checkpoint_manager_performance(self):
        """Test CheckpointManager performance."""
        results = {}
        
        with benchmark.benchmark_context("checkpoint_manager"):
            # Create model and optimizer
            model = create_large_model(PERF_CONFIG.medium_model_params)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Create CheckpointManager
            model_config = ModelConfig(
                model_name="test_checkpoint_model",
                model_path=str(PERF_CONFIG.temp_dir / "checkpoint_test.pt"),
                checkpoint_dir=str(PERF_CONFIG.temp_dir / "checkpoints")
            )
            
            checkpoint_manager = CheckpointManager(model_config)
            
            # Test checkpoint saving
            def save_checkpoint():
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": 10,
                    "loss": 0.5
                }
                return checkpoint_manager.save_checkpoint(checkpoint_data, "test_checkpoint")
            
            save_timing = benchmark.time_function(save_checkpoint)
            save_memory = benchmark.profile_memory_usage(save_checkpoint)
            
            # Test checkpoint loading
            def load_checkpoint():
                return checkpoint_manager.load_checkpoint("test_checkpoint")
            
            load_timing = benchmark.time_function(load_checkpoint)
            load_memory = benchmark.profile_memory_usage(load_checkpoint)
            
            results = {
                "save": {
                    "timing": save_timing,
                    "memory": save_memory
                },
                "load": {
                    "timing": load_timing,
                    "memory": load_memory
                }
            }
        
        save_benchmark_results("checkpoint_manager_performance", results)


class TestEndToEndPerformance:
    """End-to-end performance tests."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_training_pipeline_performance(self):
        """Test complete training pipeline performance."""
        results = {}
        
        with benchmark.benchmark_context("training_pipeline"):
            # Create model, data, and training components
            model = create_large_model(PERF_CONFIG.medium_model_params)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()
            
            # Simulate training data
            batch_size = 32
            input_size = model[0].in_features
            num_batches = 100
            
            # Training loop
            def training_epoch():
                total_loss = 0.0
                model.train()
                
                for batch_idx in range(num_batches):
                    # Generate batch data
                    inputs = torch.randn(batch_size, input_size)
                    targets = torch.randn(batch_size, 1)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                return total_loss / num_batches
            
            # Benchmark training epoch
            timing_results = benchmark.time_function(training_epoch)
            memory_results = benchmark.profile_memory_usage(training_epoch)
            
            results = {
                "timing": timing_results,
                "memory": memory_results,
                "throughput": {
                    "samples_per_second": (num_batches * batch_size) / timing_results["mean_time"],
                    "batches_per_second": num_batches / timing_results["mean_time"]
                }
            }
        
        save_benchmark_results("training_pipeline_performance", results)
        
        # Check performance thresholds
        violations = benchmark.check_performance_thresholds({
            "memory_mb": {"max": memory_results["memory_increase_mb"]}
        })
        
        if violations:
            pytest.fail(f"Training pipeline performance violations: {violations}")
        
        # Verify reasonable throughput
        throughput = results["throughput"]["samples_per_second"]
        assert throughput > 100, f"Training throughput too low: {throughput} samples/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])