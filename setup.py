#!/usr/bin/env python3
"""
AI Bull Ford (AIBF) Setup Script

Advanced AI Framework for Intelligent Systems Development
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind11_build_ext
from pybind11 import get_cmake_dir

# Ensure minimum Python version
if sys.version_info < (3, 10):
    raise RuntimeError("AI Bull Ford requires Python 3.10 or higher")

# Get the long description from README
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Version information
version_file = here / "src" / "aibf" / "__version__.py"
version_info = {}
if version_file.exists():
    exec(version_file.read_text(), version_info)
    version = version_info["__version__"]
else:
    version = "0.1.0"

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_file = here / filename
    if requirements_file.exists():
        with open(requirements_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

# Core requirements (essential for basic functionality)
core_requirements = [
    "torch>=2.1.0",
    "transformers>=4.35.0",
    "numpy>=1.24.0",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "PyYAML>=6.0.1",
    "requests>=2.31.0",
    "aiohttp>=3.9.0",
    "sqlalchemy>=2.0.0",
    "redis>=5.0.0",
    "psutil>=5.9.0",
    "structlog>=23.2.0",
    "click>=8.1.0",
    "python-dotenv>=1.0.0",
]

# Optional requirements for different features
extra_requirements = {
    "ml": [
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
    "nlp": [
        "spacy>=3.7.0",
        "nltk>=3.8.1",
        "sentence-transformers>=2.2.0",
        "langchain>=0.0.350",
        "openai>=1.3.0",
    ],
    "vision": [
        "opencv-python>=4.8.0",
        "Pillow>=10.1.0",
        "albumentations>=1.3.0",
        "torchvision>=0.16.0",
    ],
    "audio": [
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "torchaudio>=2.1.0",
    ],
    "rl": [
        "gym>=0.29.0",
        "stable-baselines3>=2.2.0",
        "ray[rllib]>=2.8.0",
    ],
    "quantum": [
        "qiskit>=0.45.0",
        "cirq>=1.3.0",
    ],
    "graph": [
        "torch-geometric>=2.4.0",
        "networkx>=3.2.0",
        "dgl>=1.1.0",
    ],
    "distributed": [
        "ray>=2.8.0",
        "dask[complete]>=2023.10.0",
        "celery>=5.3.0",
    ],
    "monitoring": [
        "prometheus-client>=0.19.0",
        "grafana-api>=1.0.3",
        "sentry-sdk>=1.38.0",
    ],
    "database": [
        "psycopg2-binary>=2.9.0",
        "pymongo>=4.6.0",
        "elasticsearch>=8.11.0",
        "chromadb>=0.4.0",
    ],
    "cloud": [
        "boto3>=1.34.0",
        "azure-storage-blob>=12.19.0",
        "google-cloud-storage>=2.10.0",
    ],
    "deployment": [
        "docker>=6.1.0",
        "kubernetes>=28.1.0",
        "gunicorn>=21.2.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "black>=23.10.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.7.0",
        "pre-commit>=3.5.0",
    ],
    "docs": [
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=1.3.0",
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.4.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "ipykernel>=6.26.0",
        "nbconvert>=7.11.0",
    ],
}

# All optional requirements
extra_requirements["all"] = [
    req for reqs in extra_requirements.values() for req in reqs
]

# Development requirements
extra_requirements["dev-all"] = (
    extra_requirements["dev"] + 
    extra_requirements["docs"] + 
    extra_requirements["jupyter"]
)

# C++ Extensions for performance-critical components
ext_modules = []

# Optional: Add C++ extensions if pybind11 is available
try:
    import pybind11
    
    # Performance-critical neural network operations
    ext_modules.append(
        Pybind11Extension(
            "aibf.core._neural_ops",
            [
                "src/aibf/core/extensions/neural_ops.cpp",
            ],
            include_dirs=[
                # Path to pybind11 headers
                pybind11.get_include(),
            ],
            language="c++",
            cxx_std=17,
        )
    )
    
    # Fast tensor operations
    ext_modules.append(
        Pybind11Extension(
            "aibf.core._tensor_ops",
            [
                "src/aibf/core/extensions/tensor_ops.cpp",
            ],
            include_dirs=[
                pybind11.get_include(),
            ],
            language="c++",
            cxx_std=17,
        )
    )
    
except ImportError:
    print("Warning: pybind11 not found. C++ extensions will not be built.")
    ext_modules = []

# Custom build command
class CustomBuildExt(pybind11_build_ext if ext_modules else build_ext):
    """Custom build extension command."""
    
    def build_extensions(self):
        # Set C++ standard
        for ext in self.extensions:
            ext.cxx_std = 17
        
        # Platform-specific optimizations
        if sys.platform == "win32":
            for ext in self.extensions:
                ext.extra_compile_args = ["/O2", "/std:c++17"]
        else:
            for ext in self.extensions:
                ext.extra_compile_args = ["-O3", "-std=c++17", "-march=native"]
        
        super().build_extensions()

# Entry points for command-line tools
entry_points = {
    "console_scripts": [
        "aibf=aibf.cli:main",
        "aibf-server=aibf.server:main",
        "aibf-train=aibf.training:main",
        "aibf-deploy=aibf.deployment:main",
        "aibf-monitor=aibf.monitoring.cli:main",
        "aibf-config=aibf.config.cli:main",
    ],
}

# Package data
package_data = {
    "aibf": [
        "config/*.yaml",
        "config/*.json",
        "templates/*.html",
        "templates/*.js",
        "templates/*.css",
        "static/*",
        "models/*.json",
        "schemas/*.json",
        "data/*.json",
        "data/*.csv",
    ],
}

# Classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: C++",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Distributed Computing",
    "Framework :: AsyncIO",
    "Framework :: FastAPI",
]

# Keywords
keywords = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural networks",
    "transformers",
    "reinforcement learning",
    "multi-agent systems",
    "natural language processing",
    "computer vision",
    "multimodal ai",
    "rag",
    "fine-tuning",
    "api framework",
    "microservices",
    "monitoring",
    "mlops",
    "ai framework",
    "intelligent systems",
]

# Project URLs
project_urls = {
    "Homepage": "https://github.com/your-org/ai-bull-ford",
    "Documentation": "https://aibf.readthedocs.io",
    "Repository": "https://github.com/your-org/ai-bull-ford",
    "Bug Reports": "https://github.com/your-org/ai-bull-ford/issues",
    "Funding": "https://github.com/sponsors/your-org",
    "Say Thanks!": "https://github.com/your-org/ai-bull-ford/discussions",
}

# Setup configuration
setup(
    name="ai-bull-ford",
    version=version,
    description="Advanced AI Framework for Intelligent Systems Development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AIBF Team",
    author_email="team@aibf.ai",
    url="https://github.com/your-org/ai-bull-ford",
    project_urls=project_urls,
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data=package_data,
    include_package_data=True,
    
    # Requirements
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require=extra_requirements,
    
    # Extensions
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    
    # Entry points
    entry_points=entry_points,
    
    # Metadata
    classifiers=classifiers,
    keywords=keywords,
    license="MIT",
    
    # Options
    zip_safe=False,
    platforms=["any"],
    
    # Additional metadata
    maintainer="AIBF Team",
    maintainer_email="maintainers@aibf.ai",
    
    # Test configuration
    test_suite="tests",
    tests_require=extra_requirements["dev"],
    
    # Options for different installation methods
    options={
        "build_ext": {
            "parallel": True,
        },
        "bdist_wheel": {
            "universal": False,
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
)

# Post-installation setup
def post_install():
    """Post-installation setup tasks."""
    print("\n🚀 AI Bull Ford installation completed!")
    print("\nNext steps:")
    print("1. Initialize the framework: aibf --init")
    print("2. Configure your settings: aibf config --setup")
    print("3. Start the server: aibf-server")
    print("4. Check the documentation: https://aibf.readthedocs.io")
    print("\nFor help: aibf --help")
    print("\nWelcome to the future of AI development! 🤖")

if __name__ == "__main__":
    # Run post-installation tasks if this is being installed
    import sys
    if "install" in sys.argv:
        import atexit
        atexit.register(post_install)