"""Performance test configuration and utilities."""

import os
import sys
import logging
import time
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import gc
import tracemalloc

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Performance test configuration
@dataclass
class PerformanceConfig:
    """Configuration for performance tests."""
    # Test timeouts
    test_timeout: int = 300  # 5 minutes
    benchmark_timeout: int = 600  # 10 minutes
    
    # Performance thresholds
    max_memory_mb: int = 2048  # 2GB
    max_cpu_percent: float = 80.0
    max_gpu_memory_mb: int = 4096  # 4GB
    
    # Benchmark parameters
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    min_samples: int = 5
    
    # Resource monitoring
    monitor_interval: float = 0.1  # seconds
    enable_memory_profiling: bool = True
    enable_gpu_monitoring: bool = torch.cuda.is_available()
    
    # Test data sizes
    small_batch_size: int = 1
    medium_batch_size: int = 8
    large_batch_size: int = 32
    
    # Model sizes
    small_model_params: int = 1_000_000  # 1M parameters
    medium_model_params: int = 10_000_000  # 10M parameters
    large_model_params: int = 100_000_000  # 100M parameters
    
    # Temporary directories
    temp_dir: Path = Path("/tmp/aibf_performance_tests")
    benchmark_results_dir: Path = Path("benchmark_results")
    
    # Logging
    log_level: str = "INFO"
    enable_detailed_logging: bool = False


# Global performance configuration
PERF_CONFIG = PerformanceConfig()

# Ensure test directories exist
PERF_CONFIG.temp_dir.mkdir(parents=True, exist_ok=True)
PERF_CONFIG.benchmark_results_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, PERF_CONFIG.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PERF_CONFIG.temp_dir / "performance_tests.log")
    ]
)

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources during performance tests."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'gpu_memory_mb': [],
            'timestamps': []
        }
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'gpu_memory_mb': [],
            'timestamps': []
        }
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop resource monitoring and return metrics."""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Calculate statistics
        stats = {}
        for metric, values in self.metrics.items():
            if metric == 'timestamps' or not values:
                continue
            
            stats[metric] = {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'samples': len(values)
            }
        
        logger.info(f"Stopped resource monitoring. Collected {len(self.metrics['timestamps'])} samples")
        return stats
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = process.cpu_percent()
                
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # GPU memory usage
                gpu_memory_mb = 0
                if self.config.enable_gpu_monitoring and torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                
                # Store metrics
                self.metrics['timestamps'].append(timestamp)
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['memory_mb'].append(memory_mb)
                self.metrics['gpu_memory_mb'].append(gpu_memory_mb)
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")
                time.sleep(self.config.monitor_interval)


class PerformanceBenchmark:
    """Performance benchmark utilities."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.monitor = ResourceMonitor(config)
    
    @contextmanager
    def benchmark_context(self, test_name: str):
        """Context manager for performance benchmarking."""
        logger.info(f"Starting benchmark: {test_name}")
        
        # Start memory profiling if enabled
        if self.config.enable_memory_profiling:
            tracemalloc.start()
        
        # Start resource monitoring
        self.monitor.start_monitoring()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_time = time.time()
        
        try:
            yield self
        finally:
            end_time = time.time()
            
            # Stop resource monitoring
            resource_stats = self.monitor.stop_monitoring()
            
            # Get memory profiling results
            memory_stats = {}
            if self.config.enable_memory_profiling:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_stats = {
                    'current_mb': current / 1024 / 1024,
                    'peak_mb': peak / 1024 / 1024
                }
            
            # Log results
            duration = end_time - start_time
            logger.info(f"Benchmark {test_name} completed in {duration:.3f}s")
            
            if resource_stats:
                logger.info(f"Resource usage - CPU: {resource_stats.get('cpu_percent', {}).get('max', 0):.1f}%, "
                           f"Memory: {resource_stats.get('memory_mb', {}).get('max', 0):.1f}MB")
            
            if memory_stats:
                logger.info(f"Memory profiling - Current: {memory_stats['current_mb']:.1f}MB, "
                           f"Peak: {memory_stats['peak_mb']:.1f}MB")
    
    def time_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Time a function execution with multiple iterations."""
        times = []
        
        # Warmup iterations
        for _ in range(self.config.warmup_iterations):
            func(*args, **kwargs)
        
        # Benchmark iterations
        for _ in range(self.config.benchmark_iterations):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        return {
            'min_time': min(times),
            'max_time': max(times),
            'mean_time': sum(times) / len(times),
            'median_time': sorted(times)[len(times) // 2],
            'total_time': sum(times),
            'iterations': len(times),
            'times': times
        }
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage of a function."""
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        initial_gpu_memory = 0
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Start memory tracing
        tracemalloc.start()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Get peak memory
            current_traced, peak_traced = tracemalloc.get_traced_memory()
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024
            
            final_gpu_memory = 0
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            return {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': final_memory - initial_memory,
                'peak_traced_mb': peak_traced / 1024 / 1024,
                'current_traced_mb': current_traced / 1024 / 1024,
                'initial_gpu_memory_mb': initial_gpu_memory,
                'final_gpu_memory_mb': final_gpu_memory,
                'gpu_memory_increase_mb': final_gpu_memory - initial_gpu_memory
            }
        
        finally:
            tracemalloc.stop()
    
    def check_performance_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """Check if performance metrics exceed thresholds."""
        violations = []
        
        # Check memory usage
        if 'memory_mb' in metrics:
            max_memory = metrics['memory_mb'].get('max', 0)
            if max_memory > self.config.max_memory_mb:
                violations.append(
                    f"Memory usage {max_memory:.1f}MB exceeds threshold {self.config.max_memory_mb}MB"
                )
        
        # Check CPU usage
        if 'cpu_percent' in metrics:
            max_cpu = metrics['cpu_percent'].get('max', 0)
            if max_cpu > self.config.max_cpu_percent:
                violations.append(
                    f"CPU usage {max_cpu:.1f}% exceeds threshold {self.config.max_cpu_percent}%"
                )
        
        # Check GPU memory usage
        if 'gpu_memory_mb' in metrics:
            max_gpu_memory = metrics['gpu_memory_mb'].get('max', 0)
            if max_gpu_memory > self.config.max_gpu_memory_mb:
                violations.append(
                    f"GPU memory usage {max_gpu_memory:.1f}MB exceeds threshold {self.config.max_gpu_memory_mb}MB"
                )
        
        return violations


def create_test_data(batch_size: int, *dimensions) -> torch.Tensor:
    """Create test data tensor."""
    shape = (batch_size,) + dimensions
    return torch.randn(shape)


def create_large_model(num_params: int) -> torch.nn.Module:
    """Create a model with approximately the specified number of parameters."""
    # Calculate layer sizes to approximate target parameters
    # For a simple linear model: params ≈ input_size * output_size + output_size
    
    if num_params < 10000:
        # Small model
        hidden_size = int((num_params / 2) ** 0.5)
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
    elif num_params < 1000000:
        # Medium model
        hidden_size = int((num_params / 4) ** 0.5)
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
    else:
        # Large model
        hidden_size = int((num_params / 8) ** 0.5)
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )
    
    return model


def save_benchmark_results(test_name: str, results: Dict[str, Any]):
    """Save benchmark results to file."""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().isoformat()
    results_file = PERF_CONFIG.benchmark_results_dir / f"{test_name}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'test_name': test_name,
            'timestamp': timestamp,
            'results': results,
            'config': {
                'max_memory_mb': PERF_CONFIG.max_memory_mb,
                'max_cpu_percent': PERF_CONFIG.max_cpu_percent,
                'benchmark_iterations': PERF_CONFIG.benchmark_iterations
            }
        }, f, indent=2)
    
    logger.info(f"Saved benchmark results to {results_file}")


# Global benchmark instance
benchmark = PerformanceBenchmark(PERF_CONFIG)

logger.info("Performance test environment initialized")
logger.info(f"Configuration: {PERF_CONFIG}")