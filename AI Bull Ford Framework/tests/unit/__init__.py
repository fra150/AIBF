"""Unit tests package for AIBF framework.

This package contains unit tests for all core modules of the AIBF framework.
Tests are organized by module and follow pytest conventions.
"""

import sys
import os
from pathlib import Path

# Add src to Python path for testing
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Test configuration
TEST_CONFIG = {
    "timeout": 30,  # seconds
    "slow_test_threshold": 5.0,  # seconds
    "temp_dir": project_root / "tests" / "temp",
    "fixtures_dir": project_root / "tests" / "fixtures",
    "coverage_threshold": 90,  # percentage
}

# Ensure test directories exist
TEST_CONFIG["temp_dir"].mkdir(exist_ok=True)
TEST_CONFIG["fixtures_dir"].mkdir(exist_ok=True)

__all__ = ["TEST_CONFIG"]