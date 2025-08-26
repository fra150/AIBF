"""Integration tests configuration for AIBF Framework."""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add src directory to Python path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Integration test configuration
INTEGRATION_TEST_CONFIG = {
    "timeout": 300,  # 5 minutes for integration tests
    "slow_test_threshold": 60,  # 1 minute
    "temp_dir": tempfile.gettempdir(),
    "test_data_dir": project_root / "tests" / "data",
    "coverage_threshold": 70,  # Lower threshold for integration tests
    "max_memory_usage": "2GB",
    "enable_gpu_tests": False,  # Set to True if GPU available
    "enable_network_tests": True,
    "test_databases": {
        "sqlite": ":memory:",
        "redis": "redis://localhost:6379/1"
    },
    "mock_external_services": True
}

# Ensure test directories exist
test_dirs = [
    INTEGRATION_TEST_CONFIG["test_data_dir"],
    Path(INTEGRATION_TEST_CONFIG["temp_dir"]) / "aibf_integration_tests"
]

for test_dir in test_dirs:
    test_dir.mkdir(parents=True, exist_ok=True)

# Configure logging for integration tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            Path(INTEGRATION_TEST_CONFIG["temp_dir"]) / "aibf_integration_tests" / "integration_tests.log"
        )
    ]
)

logger = logging.getLogger(__name__)
logger.info("Integration tests configuration loaded")

# Test fixtures and utilities
def get_test_config() -> Dict[str, Any]:
    """Get integration test configuration."""
    return INTEGRATION_TEST_CONFIG.copy()

def setup_test_environment():
    """Setup test environment for integration tests."""
    logger.info("Setting up integration test environment")
    
    # Create temporary directories
    temp_base = Path(INTEGRATION_TEST_CONFIG["temp_dir"]) / "aibf_integration_tests"
    temp_base.mkdir(exist_ok=True)
    
    # Set environment variables for testing
    os.environ["AIBF_TEST_MODE"] = "integration"
    os.environ["AIBF_LOG_LEVEL"] = "INFO"
    os.environ["AIBF_TEMP_DIR"] = str(temp_base)
    
    return temp_base

def cleanup_test_environment(temp_dir: Path):
    """Cleanup test environment after integration tests."""
    logger.info("Cleaning up integration test environment")
    
    # Clean up temporary files
    import shutil
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
    
    # Remove test environment variables
    test_env_vars = ["AIBF_TEST_MODE", "AIBF_LOG_LEVEL", "AIBF_TEMP_DIR"]
    for var in test_env_vars:
        os.environ.pop(var, None)

# Import test utilities
try:
    import pytest
    import torch
    import numpy as np
    from unittest.mock import Mock, patch, MagicMock
except ImportError as e:
    logger.warning(f"Some test dependencies not available: {e}")

logger.info("Integration tests module initialized")