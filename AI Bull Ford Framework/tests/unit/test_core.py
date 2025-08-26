"""Unit tests for core module components."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import Dict, Any, List

# Import core modules
from core.architectures.neural_networks import (
    NeuralNetworkConfig, BaseNeuralNetwork, FeedForwardNetwork,
    ConvolutionalNetwork, RecurrentNetwork, ResidualNetwork
)
from core.architectures.transformers import (
    TransformerConfig, MultiHeadAttention, TransformerBlock,
    TransformerEncoder, TransformerDecoder, VisionTransformer
)
from core.architectures.reinforcement_learning import (
    RLConfig, PolicyNetwork, ValueNetwork, ActorCriticNetwork,
    DQNNetwork, PPOAgent, SACAgent
)
from core.utils.tensor_ops import (
    safe_tensor_operation, batch_tensor_operation, tensor_memory_efficient_operation
)
from core.utils.model_utils import (
    ModelInfo, count_parameters, get_model_size, save_model_checkpoint,
    load_model_checkpoint, freeze_layers, unfreeze_layers
)


class TestNeuralNetworks:
    """Test cases for neural network architectures."""
    
    def test_neural_network_config(self):
        """Test NeuralNetworkConfig creation and validation."""
        config = NeuralNetworkConfig(
            input_size=784,
            hidden_sizes=[256, 128],
            output_size=10,
            activation="relu",
            dropout_rate=0.2
        )
        
        assert config.input_size == 784
        assert config.hidden_sizes == [256, 128]
        assert config.output_size == 10
        assert config.activation == "relu"
        assert config.dropout_rate == 0.2
    
    def test_feedforward_network(self):
        """Test FeedForwardNetwork creation and forward pass."""
        config = NeuralNetworkConfig(
            input_size=10,
            hidden_sizes=[20, 15],
            output_size=5
        )
        
        network = FeedForwardNetwork(config)
        
        # Test forward pass
        x = torch.randn(32, 10)  # batch_size=32, input_size=10
        output = network(x)
        
        assert output.shape == (32, 5)
        assert not torch.isnan(output).any()
    
    def test_convolutional_network(self):
        """Test ConvolutionalNetwork creation and forward pass."""
        config = NeuralNetworkConfig(
            input_channels=3,
            conv_layers=[(32, 3, 1), (64, 3, 1)],
            fc_layers=[128, 10]
        )
        
        network = ConvolutionalNetwork(config)
        
        # Test forward pass
        x = torch.randn(16, 3, 32, 32)  # batch_size=16, channels=3, height=32, width=32
        output = network(x)
        
        assert output.shape == (16, 10)
        assert not torch.isnan(output).any()
    
    def test_recurrent_network(self):
        """Test RecurrentNetwork creation and forward pass."""
        config = NeuralNetworkConfig(
            input_size=50,
            hidden_size=100,
            num_layers=2,
            output_size=20,
            rnn_type="LSTM"
        )
        
        network = RecurrentNetwork(config)
        
        # Test forward pass
        x = torch.randn(10, 32, 50)  # seq_len=10, batch_size=32, input_size=50
        output = network(x)
        
        assert output.shape == (32, 20)
        assert not torch.isnan(output).any()


class TestTransformers:
    """Test cases for transformer architectures."""
    
    def test_transformer_config(self):
        """Test TransformerConfig creation and validation."""
        config = TransformerConfig(
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_length=1024,
            dropout_rate=0.1
        )
        
        assert config.d_model == 512
        assert config.num_heads == 8
        assert config.num_layers == 6
        assert config.d_ff == 2048
        assert config.max_seq_length == 1024
        assert config.dropout_rate == 0.1
    
    def test_multi_head_attention(self):
        """Test MultiHeadAttention mechanism."""
        config = TransformerConfig(d_model=512, num_heads=8)
        attention = MultiHeadAttention(config)
        
        # Test forward pass
        seq_len, batch_size = 20, 16
        x = torch.randn(seq_len, batch_size, 512)
        
        output, attention_weights = attention(x, x, x)
        
        assert output.shape == (seq_len, batch_size, 512)
        assert attention_weights.shape == (batch_size, 8, seq_len, seq_len)
        assert not torch.isnan(output).any()
    
    def test_transformer_encoder(self):
        """Test TransformerEncoder."""
        config = TransformerConfig(
            d_model=256,
            num_heads=4,
            num_layers=3,
            d_ff=1024
        )
        
        encoder = TransformerEncoder(config)
        
        # Test forward pass
        seq_len, batch_size = 15, 8
        x = torch.randn(seq_len, batch_size, 256)
        
        output = encoder(x)
        
        assert output.shape == (seq_len, batch_size, 256)
        assert not torch.isnan(output).any()
    
    def test_vision_transformer(self):
        """Test VisionTransformer."""
        config = TransformerConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=12,
            num_classes=1000
        )
        
        vit = VisionTransformer(config)
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        output = vit(x)
        
        assert output.shape == (batch_size, 1000)
        assert not torch.isnan(output).any()


class TestReinforcementLearning:
    """Test cases for reinforcement learning components."""
    
    def test_rl_config(self):
        """Test RLConfig creation and validation."""
        config = RLConfig(
            state_dim=84,
            action_dim=4,
            hidden_dims=[256, 256],
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005
        )
        
        assert config.state_dim == 84
        assert config.action_dim == 4
        assert config.hidden_dims == [256, 256]
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.tau == 0.005
    
    def test_policy_network(self):
        """Test PolicyNetwork."""
        config = RLConfig(
            state_dim=10,
            action_dim=4,
            hidden_dims=[64, 32]
        )
        
        policy = PolicyNetwork(config)
        
        # Test forward pass
        batch_size = 16
        states = torch.randn(batch_size, 10)
        
        actions = policy(states)
        
        assert actions.shape == (batch_size, 4)
        assert not torch.isnan(actions).any()
    
    def test_value_network(self):
        """Test ValueNetwork."""
        config = RLConfig(
            state_dim=10,
            hidden_dims=[64, 32]
        )
        
        value_net = ValueNetwork(config)
        
        # Test forward pass
        batch_size = 16
        states = torch.randn(batch_size, 10)
        
        values = value_net(states)
        
        assert values.shape == (batch_size, 1)
        assert not torch.isnan(values).any()
    
    def test_dqn_network(self):
        """Test DQNNetwork."""
        config = RLConfig(
            state_dim=84,
            action_dim=4,
            hidden_dims=[512, 256]
        )
        
        dqn = DQNNetwork(config)
        
        # Test forward pass
        batch_size = 32
        states = torch.randn(batch_size, 84)
        
        q_values = dqn(states)
        
        assert q_values.shape == (batch_size, 4)
        assert not torch.isnan(q_values).any()


class TestTensorOps:
    """Test cases for tensor operations utilities."""
    
    def test_safe_tensor_operation(self):
        """Test safe tensor operations."""
        # Test normal operation
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        
        result = safe_tensor_operation(torch.add, x, y)
        expected = torch.add(x, y)
        
        assert torch.allclose(result, expected)
        
        # Test with invalid operation
        with pytest.raises(RuntimeError):
            safe_tensor_operation(torch.matmul, x, torch.randn(5, 5))
    
    def test_batch_tensor_operation(self):
        """Test batch tensor operations."""
        tensors = [torch.randn(100, 100) for _ in range(10)]
        
        def add_one(x):
            return x + 1
        
        results = batch_tensor_operation(add_one, tensors, batch_size=3)
        
        assert len(results) == 10
        for i, result in enumerate(results):
            expected = tensors[i] + 1
            assert torch.allclose(result, expected)
    
    def test_tensor_memory_efficient_operation(self):
        """Test memory efficient tensor operations."""
        x = torch.randn(1000, 1000)
        
        def square_operation(tensor):
            return tensor ** 2
        
        result = tensor_memory_efficient_operation(square_operation, x)
        expected = x ** 2
        
        assert torch.allclose(result, expected)


class TestModelUtils:
    """Test cases for model utilities."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Linear(10, 5)
        
        total_params, trainable_params = count_parameters(model)
        
        # Linear layer: (10 * 5) + 5 = 55 parameters
        assert total_params == 55
        assert trainable_params == 55
        
        # Test with frozen parameters
        for param in model.parameters():
            param.requires_grad = False
        
        total_params, trainable_params = count_parameters(model)
        assert total_params == 55
        assert trainable_params == 0
    
    def test_get_model_size(self):
        """Test model size calculation."""
        model = nn.Linear(100, 50)
        
        size_mb = get_model_size(model)
        
        # Should return size in MB
        assert isinstance(size_mb, float)
        assert size_mb > 0
    
    @patch('torch.save')
    def test_save_model_checkpoint(self, mock_save):
        """Test model checkpoint saving."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = "test_checkpoint.pth"
        epoch = 10
        loss = 0.5
        
        save_model_checkpoint(
            model, optimizer, checkpoint_path, epoch, loss
        )
        
        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        
        assert 'model_state_dict' in args[0]
        assert 'optimizer_state_dict' in args[0]
        assert 'epoch' in args[0]
        assert 'loss' in args[0]
        assert args[0]['epoch'] == epoch
        assert args[0]['loss'] == loss
    
    def test_freeze_unfreeze_layers(self):
        """Test layer freezing and unfreezing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Initially all parameters should be trainable
        assert all(p.requires_grad for p in model.parameters())
        
        # Freeze all layers
        freeze_layers(model)
        assert all(not p.requires_grad for p in model.parameters())
        
        # Unfreeze all layers
        unfreeze_layers(model)
        assert all(p.requires_grad for p in model.parameters())
        
        # Freeze specific layers
        freeze_layers(model, layer_names=['0'])  # Freeze first linear layer
        
        # Check that first layer is frozen, others are not
        first_layer_params = list(model[0].parameters())
        other_params = list(model[2].parameters())
        
        assert all(not p.requires_grad for p in first_layer_params)
        assert all(p.requires_grad for p in other_params)


# Integration tests for core module
class TestCoreIntegration:
    """Integration tests for core module components."""
    
    def test_neural_network_training_loop(self):
        """Test a complete training loop with neural network."""
        config = NeuralNetworkConfig(
            input_size=10,
            hidden_sizes=[20],
            output_size=2
        )
        
        model = FeedForwardNetwork(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Generate dummy data
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        
        # Training loop
        model.train()
        initial_loss = None
        
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss.item()
        
        # Loss should decrease
        final_loss = loss.item()
        assert final_loss < initial_loss
    
    def test_transformer_attention_mechanism(self):
        """Test transformer attention mechanism integration."""
        config = TransformerConfig(
            d_model=128,
            num_heads=4,
            num_layers=2
        )
        
        encoder = TransformerEncoder(config)
        
        # Test with different sequence lengths
        for seq_len in [10, 20, 50]:
            x = torch.randn(seq_len, 8, 128)
            output = encoder(x)
            
            assert output.shape == (seq_len, 8, 128)
            assert not torch.isnan(output).any()
    
    def test_rl_agent_action_selection(self):
        """Test RL agent action selection."""
        config = RLConfig(
            state_dim=4,
            action_dim=2,
            hidden_dims=[32, 32]
        )
        
        agent = PPOAgent(config)
        
        # Test action selection
        state = torch.randn(1, 4)
        action = agent.select_action(state)
        
        assert action.shape == (1, 2)
        assert not torch.isnan(action).any()
        
        # Test batch action selection
        states = torch.randn(16, 4)
        actions = agent.select_action(states)
        
        assert actions.shape == (16, 2)
        assert not torch.isnan(actions).any()


if __name__ == "__main__":
    pytest.main([__file__])