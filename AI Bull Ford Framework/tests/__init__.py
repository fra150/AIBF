"""Test suite for AIBF (AI Building Framework).

This package contains comprehensive tests for all framework components:
- Unit tests for individual modules and functions
- Integration tests for component interactions
- Performance tests for benchmarking
- End-to-end tests for complete workflows
"""

import os
import sys
import pytest
import logging
from pathlib import Path
from typing import Any, Dict, Generator

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests.log')
    ]
)

logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    'timeout': 30,  # Default test timeout in seconds
    'slow_test_threshold': 5.0,  # Tests slower than this are marked as slow
    'temp_dir': project_root / 'tests' / 'temp',
    'fixtures_dir': project_root / 'tests' / 'fixtures',
    'data_dir': project_root / 'tests' / 'data'
}

# Ensure test directories exist
for dir_path in [TEST_CONFIG['temp_dir'], TEST_CONFIG['fixtures_dir'], TEST_CONFIG['data_dir']]:
    dir_path.mkdir(parents=True, exist_ok=True)


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m "not slow"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return TEST_CONFIG.copy()


@pytest.fixture(scope="function")
def temp_dir(test_config: Dict[str, Any]) -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    import tempfile
    import shutil
    
    temp_path = Path(tempfile.mkdtemp(dir=test_config['temp_dir']))
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def sample_data_dir(test_config: Dict[str, Any]) -> Path:
    """Provide path to sample test data."""
    return test_config['data_dir']


class TestBase:
    """Base class for all test classes with common utilities."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.info(f"Setting up test class: {cls.__name__}")
    
    @classmethod
    def teardown_class(cls):
        """Tear down test class."""
        cls.logger.info(f"Tearing down test class: {cls.__name__}")
    
    def setup_method(self, method):
        """Set up individual test method."""
        self.logger.info(f"Starting test: {method.__name__}")
    
    def teardown_method(self, method):
        """Tear down individual test method."""
        self.logger.info(f"Finished test: {method.__name__}")


# Export commonly used testing utilities
__all__ = [
    'TestBase',
    'TEST_CONFIG',
    'logger',
    'pytest_configure'
]