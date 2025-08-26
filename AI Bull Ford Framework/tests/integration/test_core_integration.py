"""Integration tests for core module interactions."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple
import tempfile
from pathlib import Path

# Import core modules
from core.neural_networks import (
    NeuralNetworkConfig, FeedForwardNetwork, ConvolutionalNetwork,
    RecurrentNetwork, ResidualNetwork, AttentionNetwork
)
from core.transformers import (
    TransformerConfig, TransformerEncoder, TransformerDecoder,
    MultiHeadAttention, PositionalEncoding, TransformerBlock
)
from core.reinforcement_learning import (
    RLConfig, PolicyNetwork, ValueNetwork, ActorCriticNetwork,
    ExperienceReplay, RLAgent, Environment
)
from core.tensor_ops import (
    TensorConfig, TensorOperations, MatrixOperations,
    ConvolutionOperations, ActivationFunctions, LossFunction
)
from core.model_utils import (
    ModelConfig, ModelManager, CheckpointManager,
    ModelOptimizer, ModelEvaluator, ModelExporter
)

# Import assembly line for integration
from assembly_line.pipeline import (
    PipelineConfig, Pipeline, PipelineStage, PipelineManager
)
from assembly_line.module_registry import (
    ModuleRegistry, ModuleInfo, ModuleManager
)


class TestNeuralNetworkIntegration:
    """Test integration between different neural network components."""
    
    def test_feedforward_with_tensor_ops(self):
        """Test FeedForwardNetwork with custom tensor operations."""
        # Create configs
        nn_config = NeuralNetworkConfig(
            input_size=784,
            hidden_sizes=[512, 256, 128],
            output_size=10,
            activation="relu",
            dropout_rate=0.2
        )
        
        tensor_config = TensorConfig(
            device="cpu",
            dtype=torch.float32,
            enable_mixed_precision=False
        )
        
        # Create components
        network = FeedForwardNetwork(nn_config)
        tensor_ops = TensorOperations(tensor_config)
        
        # Test integration
        batch_size = 32
        input_data = torch.randn(batch_size, 784)
        
        # Forward pass through network
        output = network(input_data)
        
        # Apply tensor operations
        normalized_output = tensor_ops.normalize(output, dim=1)
        softmax_output = tensor_ops.softmax(normalized_output, dim=1)
        
        assert output.shape == (batch_size, 10)
        assert normalized_output.shape == (batch_size, 10)
        assert softmax_output.shape == (batch_size, 10)
        assert torch.allclose(softmax_output.sum(dim=1), torch.ones(batch_size))
    
    def test_cnn_with_attention(self):
        """Test ConvolutionalNetwork with AttentionNetwork integration."""
        # Create configs
        cnn_config = NeuralNetworkConfig(
            input_channels=3,
            output_channels=[32, 64, 128],
            kernel_sizes=[3, 3, 3],
            stride=1,
            padding=1
        )
        
        attention_config = NeuralNetworkConfig(
            input_size=128,
            hidden_size=256,
            num_heads=8,
            dropout_rate=0.1
        )
        
        # Create networks
        cnn = ConvolutionalNetwork(cnn_config)
        attention = AttentionNetwork(attention_config)
        
        # Test integration
        batch_size = 16
        input_images = torch.randn(batch_size, 3, 32, 32)
        
        # CNN feature extraction
        cnn_features = cnn(input_images)
        
        # Reshape for attention
        seq_len = cnn_features.shape[2] * cnn_features.shape[3]
        attention_input = cnn_features.view(batch_size, 128, seq_len).transpose(1, 2)
        
        # Apply attention
        attended_features = attention(attention_input)
        
        assert cnn_features.shape[0] == batch_size
        assert cnn_features.shape[1] == 128
        assert attended_features.shape == (batch_size, seq_len, 128)
    
    def test_rnn_with_transformer(self):
        """Test RecurrentNetwork with TransformerEncoder integration."""
        # Create configs
        rnn_config = NeuralNetworkConfig(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            rnn_type="LSTM",
            bidirectional=True
        )
        
        transformer_config = TransformerConfig(
            d_model=512,  # 256 * 2 for bidirectional
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Create networks
        rnn = RecurrentNetwork(rnn_config)
        transformer = TransformerEncoder(transformer_config)
        
        # Test integration
        batch_size = 8
        seq_len = 50
        input_seq = torch.randn(batch_size, seq_len, 512)
        
        # RNN processing
        rnn_output, (hidden, cell) = rnn(input_seq)
        
        # Transformer processing
        # Transpose for transformer (seq_len, batch_size, d_model)
        transformer_input = rnn_output.transpose(0, 1)
        transformer_output = transformer(transformer_input)
        
        # Transpose back
        final_output = transformer_output.transpose(0, 1)
        
        assert rnn_output.shape == (batch_size, seq_len, 512)
        assert final_output.shape == (batch_size, seq_len, 512)


class TestTransformerIntegration:
    """Test integration between transformer components."""
    
    def test_encoder_decoder_integration(self):
        """Test TransformerEncoder and TransformerDecoder integration."""
        config = TransformerConfig(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            vocab_size=10000
        )
        
        encoder = TransformerEncoder(config)
        decoder = TransformerDecoder(config)
        
        # Test sequence-to-sequence
        batch_size = 4
        src_seq_len = 20
        tgt_seq_len = 15
        
        src_tokens = torch.randint(0, config.vocab_size, (batch_size, src_seq_len))
        tgt_tokens = torch.randint(0, config.vocab_size, (batch_size, tgt_seq_len))
        
        # Encode source sequence
        src_embeddings = encoder.embedding(src_tokens)
        src_embeddings = encoder.pos_encoding(src_embeddings.transpose(0, 1))
        encoder_output = encoder.transformer_encoder(src_embeddings)
        
        # Decode target sequence
        tgt_embeddings = decoder.embedding(tgt_tokens)
        tgt_embeddings = decoder.pos_encoding(tgt_embeddings.transpose(0, 1))
        
        # Create causal mask for decoder
        tgt_mask = decoder._generate_square_subsequent_mask(tgt_seq_len)
        
        decoder_output = decoder.transformer_decoder(
            tgt_embeddings, encoder_output, tgt_mask=tgt_mask
        )
        
        # Final projection
        logits = decoder.output_projection(decoder_output.transpose(0, 1))
        
        assert encoder_output.shape == (src_seq_len, batch_size, config.d_model)
        assert decoder_output.shape == (tgt_seq_len, batch_size, config.d_model)
        assert logits.shape == (batch_size, tgt_seq_len, config.vocab_size)
    
    def test_multihead_attention_with_positional_encoding(self):
        """Test MultiHeadAttention with PositionalEncoding."""
        config = TransformerConfig(
            d_model=512,
            nhead=8,
            dropout=0.1,
            max_seq_length=1000
        )
        
        attention = MultiHeadAttention(config)
        pos_encoding = PositionalEncoding(config)
        
        # Test attention with positional encoding
        batch_size = 6
        seq_len = 100
        
        # Create input embeddings
        embeddings = torch.randn(batch_size, seq_len, config.d_model)
        
        # Add positional encoding
        embeddings_with_pos = pos_encoding(embeddings.transpose(0, 1))
        embeddings_with_pos = embeddings_with_pos.transpose(0, 1)
        
        # Apply attention
        attended_output, attention_weights = attention(
            embeddings_with_pos,
            embeddings_with_pos,
            embeddings_with_pos
        )
        
        assert attended_output.shape == (batch_size, seq_len, config.d_model)
        assert attention_weights.shape == (batch_size, config.nhead, seq_len, seq_len)


class TestReinforcementLearningIntegration:
    """Test integration between RL components."""
    
    def test_actor_critic_with_experience_replay(self):
        """Test ActorCriticNetwork with ExperienceReplay integration."""
        rl_config = RLConfig(
            state_dim=84,
            action_dim=4,
            hidden_dim=512,
            learning_rate=0.001,
            gamma=0.99,
            buffer_size=10000
        )
        
        # Create components
        actor_critic = ActorCriticNetwork(rl_config)
        experience_replay = ExperienceReplay(rl_config)
        
        # Generate sample experiences
        batch_size = 32
        for _ in range(100):
            state = torch.randn(rl_config.state_dim)
            action = torch.randint(0, rl_config.action_dim, (1,)).item()
            reward = torch.randn(1).item()
            next_state = torch.randn(rl_config.state_dim)
            done = torch.randint(0, 2, (1,)).item() == 1
            
            experience_replay.add(state, action, reward, next_state, done)
        
        # Sample batch and train
        if len(experience_replay) >= batch_size:
            batch = experience_replay.sample(batch_size)
            
            states = torch.stack([exp[0] for exp in batch])
            actions = torch.tensor([exp[1] for exp in batch])
            rewards = torch.tensor([exp[2] for exp in batch])
            next_states = torch.stack([exp[3] for exp in batch])
            dones = torch.tensor([exp[4] for exp in batch])
            
            # Forward pass
            action_probs, state_values = actor_critic(states)
            next_action_probs, next_state_values = actor_critic(next_states)
            
            assert action_probs.shape == (batch_size, rl_config.action_dim)
            assert state_values.shape == (batch_size, 1)
            assert next_state_values.shape == (batch_size, 1)
    
    def test_rl_agent_with_environment(self):
        """Test RLAgent with Environment integration."""
        rl_config = RLConfig(
            state_dim=4,
            action_dim=2,
            hidden_dim=128,
            learning_rate=0.001
        )
        
        # Create mock environment
        class MockEnvironment(Environment):
            def __init__(self):
                super().__init__()
                self.state = torch.randn(4)
                self.step_count = 0
            
            def reset(self):
                self.state = torch.randn(4)
                self.step_count = 0
                return self.state
            
            def step(self, action):
                self.step_count += 1
                reward = torch.randn(1).item()
                done = self.step_count >= 100
                self.state = torch.randn(4)
                return self.state, reward, done, {}
            
            def get_state_dim(self):
                return 4
            
            def get_action_dim(self):
                return 2
        
        env = MockEnvironment()
        agent = RLAgent(rl_config)
        
        # Test training episode
        state = env.reset()
        total_reward = 0
        
        for _ in range(10):  # Short episode for testing
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Train agent
        if len(agent.memory) > 0:
            loss = agent.train()
            assert isinstance(loss, (int, float, torch.Tensor))
        
        assert total_reward is not None


class TestModelUtilsIntegration:
    """Test integration between model utilities."""
    
    def test_model_manager_with_checkpoint_manager(self):
        """Test ModelManager with CheckpointManager integration."""
        model_config = ModelConfig(
            model_type="feedforward",
            input_size=784,
            output_size=10,
            hidden_sizes=[512, 256]
        )
        
        # Create temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            checkpoint_dir.mkdir()
            
            model_manager = ModelManager(model_config)
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(checkpoint_dir),
                max_checkpoints=3
            )
            
            # Create and register model
            model = FeedForwardNetwork(NeuralNetworkConfig(
                input_size=784,
                hidden_sizes=[512, 256],
                output_size=10
            ))
            
            model_manager.register_model("test_model", model)
            
            # Save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=torch.optim.Adam(model.parameters()),
                epoch=1,
                loss=0.5,
                metadata={"accuracy": 0.85}
            )
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_path)
            
            assert "model_state_dict" in checkpoint_data
            assert "optimizer_state_dict" in checkpoint_data
            assert "epoch" in checkpoint_data
            assert checkpoint_data["epoch"] == 1
    
    def test_model_optimizer_with_evaluator(self):
        """Test ModelOptimizer with ModelEvaluator integration."""
        model_config = ModelConfig(
            model_type="feedforward",
            input_size=784,
            output_size=10
        )
        
        optimizer = ModelOptimizer(model_config)
        evaluator = ModelEvaluator(model_config)
        
        # Create model and data
        model = FeedForwardNetwork(NeuralNetworkConfig(
            input_size=784,
            hidden_sizes=[256, 128],
            output_size=10
        ))
        
        # Generate sample data
        batch_size = 64
        train_data = torch.randn(batch_size, 784)
        train_labels = torch.randint(0, 10, (batch_size,))
        
        test_data = torch.randn(32, 784)
        test_labels = torch.randint(0, 10, (32,))
        
        # Training step
        optimizer.setup_optimizer(model, lr=0.001)
        
        model.train()
        outputs = model(train_data)
        loss = nn.CrossEntropyLoss()(outputs, train_labels)
        
        optimizer.step(loss)
        
        # Evaluation step
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data)
            test_loss = nn.CrossEntropyLoss()(test_outputs, test_labels)
            
            metrics = evaluator.evaluate(
                model=model,
                data_loader=[(test_data, test_labels)],
                metrics=["accuracy", "loss"]
            )
        
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["loss"] >= 0


class TestAssemblyLineIntegration:
    """Test integration with assembly line components."""
    
    def test_pipeline_with_neural_networks(self):
        """Test Pipeline with neural network stages."""
        pipeline_config = PipelineConfig(
            name="neural_pipeline",
            stages=[
                {"name": "preprocessing", "type": "data"},
                {"name": "feature_extraction", "type": "neural"},
                {"name": "classification", "type": "neural"},
                {"name": "postprocessing", "type": "data"}
            ],
            parallel_execution=False
        )
        
        pipeline = Pipeline(pipeline_config)
        
        # Create neural network stages
        feature_extractor = ConvolutionalNetwork(NeuralNetworkConfig(
            input_channels=3,
            output_channels=[32, 64],
            kernel_sizes=[3, 3]
        ))
        
        classifier = FeedForwardNetwork(NeuralNetworkConfig(
            input_size=64,
            hidden_sizes=[128],
            output_size=10
        ))
        
        # Mock stage implementations
        def preprocessing_stage(data):
            # Normalize data
            return (data - data.mean()) / data.std()
        
        def feature_extraction_stage(data):
            # Extract features using CNN
            features = feature_extractor(data)
            # Global average pooling
            return features.mean(dim=[2, 3])
        
        def classification_stage(features):
            # Classify using feedforward network
            return classifier(features)
        
        def postprocessing_stage(logits):
            # Apply softmax
            return torch.softmax(logits, dim=1)
        
        # Register stages
        pipeline.add_stage("preprocessing", preprocessing_stage)
        pipeline.add_stage("feature_extraction", feature_extraction_stage)
        pipeline.add_stage("classification", classification_stage)
        pipeline.add_stage("postprocessing", postprocessing_stage)
        
        # Test pipeline execution
        batch_size = 8
        input_data = torch.randn(batch_size, 3, 32, 32)
        
        result = pipeline.execute(input_data)
        
        assert result.shape == (batch_size, 10)
        assert torch.allclose(result.sum(dim=1), torch.ones(batch_size))
    
    def test_module_registry_with_transformers(self):
        """Test ModuleRegistry with transformer components."""
        registry = ModuleRegistry()
        
        # Register transformer modules
        transformer_config = TransformerConfig(
            d_model=512,
            nhead=8,
            num_encoder_layers=6
        )
        
        encoder = TransformerEncoder(transformer_config)
        attention = MultiHeadAttention(transformer_config)
        
        # Register modules
        registry.register_module(
            "transformer_encoder",
            encoder,
            ModuleInfo(
                name="transformer_encoder",
                version="1.0.0",
                description="Transformer encoder for sequence processing",
                dependencies=["torch"],
                config=transformer_config.__dict__
            )
        )
        
        registry.register_module(
            "multihead_attention",
            attention,
            ModuleInfo(
                name="multihead_attention",
                version="1.0.0",
                description="Multi-head attention mechanism",
                dependencies=["torch"],
                config=transformer_config.__dict__
            )
        )
        
        # Test module retrieval
        retrieved_encoder = registry.get_module("transformer_encoder")
        retrieved_attention = registry.get_module("multihead_attention")
        
        assert retrieved_encoder is not None
        assert retrieved_attention is not None
        
        # Test module info
        encoder_info = registry.get_module_info("transformer_encoder")
        assert encoder_info.name == "transformer_encoder"
        assert encoder_info.version == "1.0.0"
        
        # Test module listing
        modules = registry.list_modules()
        assert "transformer_encoder" in modules
        assert "multihead_attention" in modules


class TestCrossModuleIntegration:
    """Test integration across different core modules."""
    
    def test_end_to_end_training_pipeline(self):
        """Test complete training pipeline integration."""
        # Create configs
        nn_config = NeuralNetworkConfig(
            input_size=784,
            hidden_sizes=[512, 256],
            output_size=10,
            activation="relu",
            dropout_rate=0.2
        )
        
        model_config = ModelConfig(
            model_type="feedforward",
            input_size=784,
            output_size=10
        )
        
        tensor_config = TensorConfig(
            device="cpu",
            dtype=torch.float32
        )
        
        # Create components
        model = FeedForwardNetwork(nn_config)
        model_manager = ModelManager(model_config)
        optimizer = ModelOptimizer(model_config)
        evaluator = ModelEvaluator(model_config)
        tensor_ops = TensorOperations(tensor_config)
        
        # Register model
        model_manager.register_model("classifier", model)
        
        # Setup training
        optimizer.setup_optimizer(model, lr=0.001)
        
        # Generate training data
        num_batches = 5
        batch_size = 32
        
        for batch_idx in range(num_batches):
            # Generate batch
            data = torch.randn(batch_size, 784)
            labels = torch.randint(0, 10, (batch_size,))
            
            # Normalize data using tensor ops
            normalized_data = tensor_ops.normalize(data, dim=1)
            
            # Forward pass
            model.train()
            outputs = model(normalized_data)
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Backward pass
            optimizer.step(loss)
            
            # Evaluate periodically
            if batch_idx % 2 == 0:
                model.eval()
                with torch.no_grad():
                    eval_data = torch.randn(16, 784)
                    eval_labels = torch.randint(0, 10, (16,))
                    
                    eval_data = tensor_ops.normalize(eval_data, dim=1)
                    eval_outputs = model(eval_data)
                    
                    metrics = evaluator.evaluate(
                        model=model,
                        data_loader=[(eval_data, eval_labels)],
                        metrics=["accuracy"]
                    )
                    
                    assert "accuracy" in metrics
        
        # Final evaluation
        final_metrics = evaluator.evaluate(
            model=model,
            data_loader=[(normalized_data, labels)],
            metrics=["accuracy", "loss"]
        )
        
        assert "accuracy" in final_metrics
        assert "loss" in final_metrics
        assert 0 <= final_metrics["accuracy"] <= 1
    
    def test_multimodal_transformer_rl_integration(self):
        """Test integration of transformers with RL for multimodal tasks."""
        # Create configs
        transformer_config = TransformerConfig(
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            vocab_size=1000
        )
        
        rl_config = RLConfig(
            state_dim=256,  # Transformer output dimension
            action_dim=4,
            hidden_dim=128,
            learning_rate=0.001
        )
        
        # Create components
        transformer = TransformerEncoder(transformer_config)
        rl_agent = RLAgent(rl_config)
        
        # Simulate multimodal environment
        batch_size = 4
        seq_len = 20
        
        # Text input (token sequences)
        text_tokens = torch.randint(0, transformer_config.vocab_size, (batch_size, seq_len))
        
        # Process text with transformer
        text_embeddings = transformer.embedding(text_tokens)
        text_embeddings = transformer.pos_encoding(text_embeddings.transpose(0, 1))
        transformer_output = transformer.transformer_encoder(text_embeddings)
        
        # Use transformer output as state for RL agent
        # Take mean over sequence length to get fixed-size state
        rl_state = transformer_output.mean(dim=0)  # (batch_size, d_model)
        
        # RL agent selects actions based on transformer-processed text
        actions = []
        for i in range(batch_size):
            action = rl_agent.select_action(rl_state[i])
            actions.append(action)
        
        actions = torch.tensor(actions)
        
        assert transformer_output.shape == (seq_len, batch_size, transformer_config.d_model)
        assert rl_state.shape == (batch_size, transformer_config.d_model)
        assert actions.shape == (batch_size,)
        assert all(0 <= action < rl_config.action_dim for action in actions)


if __name__ == "__main__":
    pytest.main([__file__])