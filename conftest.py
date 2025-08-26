"""Global pytest configuration and fixtures for AIBF framework tests."""

import pytest
import asyncio
import os
import sys
import tempfile
import shutil
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Generator, AsyncGenerator
from unittest.mock import Mock, patch

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable verbose logging from external libraries during tests
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker for tests that typically take longer
        if any(keyword in item.name.lower() for keyword in ["integration", "e2e", "performance", "workflow"]):
            item.add_marker(pytest.mark.slow)
        
        # Add network marker for tests that require network
        if any(keyword in item.name.lower() for keyword in ["api", "client", "server", "websocket", "grpc"]):
            item.add_marker(pytest.mark.network)


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Temporary directory fixtures
@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test use."""
    temp_path = Path(tempfile.mkdtemp(prefix="aibf_test_"))
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def session_temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for the entire test session."""
    temp_path = Path(tempfile.mkdtemp(prefix="aibf_session_"))
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


# Sample data fixtures
@pytest.fixture(scope="session")
def sample_image_data():
    """Generate sample image data for testing."""
    import numpy as np
    return {
        "small_image": np.random.rand(64, 64, 3),
        "medium_image": np.random.rand(224, 224, 3),
        "large_image": np.random.rand(512, 512, 3),
        "grayscale_image": np.random.rand(224, 224, 1),
        "batch_images": np.random.rand(8, 224, 224, 3)
    }


@pytest.fixture(scope="session")
def sample_audio_data():
    """Generate sample audio data for testing."""
    import numpy as np
    return {
        "short_audio": np.random.rand(1000),  # 1 second at 1kHz
        "medium_audio": np.random.rand(16000),  # 1 second at 16kHz
        "long_audio": np.random.rand(48000),  # 1 second at 48kHz
        "stereo_audio": np.random.rand(16000, 2),  # Stereo
        "mel_spectrogram": np.random.rand(80, 100),  # 80 mel bins, 100 time steps
        "mfcc_features": np.random.rand(13, 100)  # 13 MFCC coefficients, 100 time steps
    }


@pytest.fixture(scope="session")
def sample_text_data():
    """Generate sample text data for testing."""
    return {
        "short_text": "This is a short test sentence.",
        "medium_text": "This is a medium length text that contains multiple sentences. It is used for testing text processing capabilities of the framework.",
        "long_text": "This is a much longer text that spans multiple paragraphs and contains various types of content. " * 10,
        "multilingual_text": {
            "english": "Hello, how are you today?",
            "spanish": "Hola, ¿cómo estás hoy?",
            "french": "Bonjour, comment allez-vous aujourd'hui?",
            "german": "Hallo, wie geht es dir heute?"
        },
        "technical_text": "The neural network architecture consists of multiple layers including convolutional, pooling, and fully connected layers.",
        "medical_text": "Patient presents with acute chest pain, shortness of breath, and elevated cardiac enzymes consistent with myocardial infarction.",
        "financial_text": "The portfolio shows strong performance with a Sharpe ratio of 1.2 and maximum drawdown of 8.5% over the past year."
    }


@pytest.fixture(scope="session")
def sample_tabular_data():
    """Generate sample tabular data for testing."""
    import numpy as np
    import pandas as pd
    
    # Generate synthetic tabular data
    n_samples = 1000
    
    data = {
        "feature_1": np.random.normal(0, 1, n_samples),
        "feature_2": np.random.uniform(-1, 1, n_samples),
        "feature_3": np.random.exponential(1, n_samples),
        "category_1": np.random.choice(["A", "B", "C"], n_samples),
        "category_2": np.random.choice(["X", "Y", "Z"], n_samples),
        "target_regression": np.random.normal(10, 2, n_samples),
        "target_classification": np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


# Configuration fixtures
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "timeout": 30,  # seconds
        "slow_test_threshold": 5,  # seconds
        "temp_dir_prefix": "aibf_test_",
        "log_level": "INFO",
        "enable_gpu_tests": os.environ.get("ENABLE_GPU_TESTS", "false").lower() == "true",
        "enable_network_tests": os.environ.get("ENABLE_NETWORK_TESTS", "false").lower() == "true",
        "enable_slow_tests": os.environ.get("ENABLE_SLOW_TESTS", "false").lower() == "true",
        "test_data_size": {
            "small": 100,
            "medium": 1000,
            "large": 10000
        },
        "model_config": {
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        "api_config": {
            "host": "localhost",
            "rest_port": 8080,
            "websocket_port": 8081,
            "grpc_port": 8082,
            "timeout": 30
        }
    }


# Mock fixtures
@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.forward.return_value = Mock()
    model.train.return_value = None
    model.eval.return_value = None
    model.parameters.return_value = []
    model.state_dict.return_value = {}
    model.load_state_dict.return_value = None
    return model


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = Mock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = (Mock(), Mock())
    return dataset


@pytest.fixture
def mock_dataloader():
    """Mock dataloader for testing."""
    dataloader = Mock()
    dataloader.__iter__.return_value = iter([(Mock(), Mock()) for _ in range(10)])
    dataloader.__len__.return_value = 10
    return dataloader


# Framework component fixtures
@pytest.fixture
def neural_network_config():
    """Configuration for neural network components."""
    return {
        "feedforward": {
            "input_dim": 784,
            "hidden_dims": [256, 128],
            "output_dim": 10,
            "activation": "relu",
            "dropout": 0.1
        },
        "cnn": {
            "input_shape": (224, 224, 3),
            "num_classes": 10,
            "filters": [32, 64, 128],
            "kernel_sizes": [3, 3, 3],
            "pool_sizes": [2, 2, 2]
        },
        "rnn": {
            "input_size": 100,
            "hidden_size": 128,
            "num_layers": 2,
            "output_size": 10,
            "rnn_type": "LSTM",
            "bidirectional": True
        },
        "transformer": {
            "d_model": 512,
            "nhead": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "dim_feedforward": 2048,
            "dropout": 0.1
        }
    }


@pytest.fixture
def multimodal_config():
    """Configuration for multimodal components."""
    return {
        "vision": {
            "model_type": "cnn",
            "input_shape": (224, 224, 3),
            "embedding_dim": 512
        },
        "audio": {
            "model_type": "transformer",
            "input_features": 80,
            "embedding_dim": 512
        },
        "text": {
            "model_type": "transformer",
            "vocab_size": 10000,
            "embedding_dim": 512
        },
        "fusion": {
            "fusion_method": "attention",
            "output_dim": 256
        }
    }


@pytest.fixture
def application_config():
    """Configuration for application components."""
    return {
        "healthcare": {
            "medical_analyzer": {
                "model_path": "test_models/medical",
                "confidence_threshold": 0.8,
                "supported_modalities": ["xray", "ct", "mri"]
            },
            "patient_processor": {
                "feature_extraction": True,
                "normalization": True,
                "privacy_mode": True
            }
        },
        "financial": {
            "risk_manager": {
                "risk_models": ["var", "cvar", "sharpe"],
                "confidence_level": 0.95,
                "time_horizon": 252
            },
            "portfolio_optimizer": {
                "optimization_method": "mean_variance",
                "constraints": {
                    "max_weight": 0.3,
                    "min_weight": 0.01
                }
            }
        },
        "educational": {
            "content_recommender": {
                "recommendation_algorithms": ["collaborative", "content_based"],
                "max_recommendations": 10,
                "diversity_factor": 0.3
            },
            "learner_profiler": {
                "profiling_methods": ["performance_based", "preference_based"],
                "update_frequency": "real_time"
            }
        }
    }


# Skip conditions
def pytest_runtest_setup(item):
    """Setup function to skip tests based on conditions."""
    # Skip GPU tests if GPU not available or not enabled
    if "gpu" in item.keywords:
        if not os.environ.get("ENABLE_GPU_TESTS", "false").lower() == "true":
            pytest.skip("GPU tests disabled (set ENABLE_GPU_TESTS=true to enable)")
    
    # Skip network tests if network tests not enabled
    if "network" in item.keywords:
        if not os.environ.get("ENABLE_NETWORK_TESTS", "false").lower() == "true":
            pytest.skip("Network tests disabled (set ENABLE_NETWORK_TESTS=true to enable)")
    
    # Skip slow tests if slow tests not enabled
    if "slow" in item.keywords:
        if not os.environ.get("ENABLE_SLOW_TESTS", "false").lower() == "true":
            pytest.skip("Slow tests disabled (set ENABLE_SLOW_TESTS=true to enable)")


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Cleanup environment after each test."""
    yield
    
    # Clear any global state
    import gc
    gc.collect()
    
    # Reset logging level
    logging.getLogger().setLevel(logging.INFO)


# Performance monitoring fixture
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time
    import psutil
    import threading
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.peak_memory = 0
            self.monitoring = False
            self.monitor_thread = None
        
        def start(self):
            self.start_time = time.time()
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_memory)
            self.monitor_thread.start()
        
        def stop(self):
            self.end_time = time.time()
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
        
        def _monitor_memory(self):
            process = psutil.Process()
            while self.monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    time.sleep(0.1)
                except:
                    break
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        def get_metrics(self):
            return {
                "duration": self.duration,
                "peak_memory_mb": self.peak_memory
            }
    
    return PerformanceMonitor()