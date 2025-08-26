#!/usr/bin/env python3
"""
AIBF Benchmark Test Suite

Tests for the benchmark infrastructure to ensure reliability
and correctness of performance measurements.

Usage:
    python -m pytest test_benchmarks.py -v
    python test_benchmarks.py --run-integration
"""

import unittest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from tests.benchmarks.benchmark_suite import BenchmarkSuite
    from tests.benchmarks.performance_analysis import PerformanceAnalyzer, PerformanceDatabase, PerformanceRegression
    from tests.benchmarks.run_benchmarks import BenchmarkRunner
except ImportError as e:
    print(f"Warning: Could not import benchmark modules: {e}")
    print("Some tests will be skipped")

class TestBenchmarkSuite(unittest.TestCase):
    """Test the benchmark suite functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Create minimal test config
        test_config = """
global:
  iterations: 2
  timeout_seconds: 30
  memory_limit_gb: 1
  enable_gpu: false
  enable_profiling: false

thresholds:
  core:
    test_neural_network:
      max_time_ms: 1000
      max_memory_mb: 100
"""
        with open(self.config_file, 'w') as f:
            f.write(test_config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite can be initialized."""
        try:
            suite = BenchmarkSuite(self.config_file)
            self.assertIsNotNone(suite)
            self.assertIsNotNone(suite.config)
        except NameError:
            self.skipTest("BenchmarkSuite not available")
    
    def test_mock_benchmark_execution(self):
        """Test mock benchmark execution."""
        # Create a mock benchmark function
        def mock_benchmark():
            time.sleep(0.01)  # Simulate work
            return {
                'test_name': 'mock_test',
                'module_name': 'test_module',
                'iterations': 2,
                'mean_time': 0.01,
                'std_time': 0.001,
                'min_time': 0.009,
                'max_time': 0.011,
                'throughput': 100.0,
                'memory_usage_mb': 10.0,
                'cpu_usage_percent': 50.0,
                'error_rate': 0.0
            }
        
        # Execute mock benchmark
        result = mock_benchmark()
        
        # Validate result structure
        required_fields = [
            'test_name', 'module_name', 'iterations', 'mean_time',
            'std_time', 'min_time', 'max_time', 'throughput',
            'memory_usage_mb', 'cpu_usage_percent', 'error_rate'
        ]
        
        for field in required_fields:
            self.assertIn(field, result)
            self.assertIsNotNone(result[field])
        
        # Validate data types
        self.assertIsInstance(result['test_name'], str)
        self.assertIsInstance(result['module_name'], str)
        self.assertIsInstance(result['iterations'], int)
        self.assertIsInstance(result['mean_time'], (int, float))
        self.assertIsInstance(result['throughput'], (int, float))
        self.assertIsInstance(result['memory_usage_mb'], (int, float))
        self.assertIsInstance(result['error_rate'], (int, float))
        
        # Validate ranges
        self.assertGreaterEqual(result['error_rate'], 0.0)
        self.assertLessEqual(result['error_rate'], 1.0)
        self.assertGreater(result['mean_time'], 0)
        self.assertGreater(result['throughput'], 0)

class TestPerformanceAnalyzer(unittest.TestCase):
    """Test performance analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_file = os.path.join(self.temp_dir, "test_benchmark.db")
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_database_initialization(self):
        """Test performance database initialization."""
        try:
            db = PerformanceDatabase(self.db_file)
            self.assertIsNotNone(db)
            self.assertTrue(os.path.exists(self.db_file))
        except NameError:
            self.skipTest("PerformanceDatabase not available")
    
    def test_regression_detection(self):
        """Test performance regression detection."""
        try:
            # Create test data
            baseline_data = {
                'results': [
                    {
                        'module_name': 'test_module',
                        'test_name': 'test_function',
                        'mean_time': 1.0,
                        'throughput': 100.0,
                        'memory_usage_mb': 50.0
                    }
                ]
            }
            
            current_data = {
                'results': [
                    {
                        'module_name': 'test_module',
                        'test_name': 'test_function',
                        'mean_time': 1.2,  # 20% slower
                        'throughput': 80.0,  # 20% slower
                        'memory_usage_mb': 60.0  # 20% more memory
                    }
                ]
            }
            
            # Create temporary files
            baseline_file = os.path.join(self.temp_dir, "baseline.json")
            current_file = os.path.join(self.temp_dir, "current.json")
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f)
            
            with open(current_file, 'w') as f:
                json.dump(current_data, f)
            
            # Test regression detection
            analyzer = PerformanceAnalyzer()
            regressions = analyzer.compare_results(baseline_file, current_file)
            
            # Should detect regressions
            self.assertGreater(len(regressions), 0)
            
            # Check regression details
            time_regression = next(
                (r for r in regressions if r.metric == 'mean_time'), None
            )
            self.assertIsNotNone(time_regression)
            self.assertEqual(time_regression.test_name, 'test_function')
            self.assertEqual(time_regression.module_name, 'test_module')
            self.assertAlmostEqual(time_regression.change_percent, 20.0, places=1)
            
        except NameError:
            self.skipTest("PerformanceAnalyzer not available")
    
    def test_performance_regression_classification(self):
        """Test performance regression severity classification."""
        try:
            from datetime import datetime
            
            # Test different severity levels
            minor_regression = PerformanceRegression(
                test_name="test",
                module_name="module",
                metric="mean_time",
                baseline_value=1.0,
                current_value=1.1,  # 10% increase
                change_percent=10.0,
                severity="minor",
                timestamp=datetime.now()
            )
            
            major_regression = PerformanceRegression(
                test_name="test",
                module_name="module",
                metric="mean_time",
                baseline_value=1.0,
                current_value=1.3,  # 30% increase
                change_percent=30.0,
                severity="major",
                timestamp=datetime.now()
            )
            
            critical_regression = PerformanceRegression(
                test_name="test",
                module_name="module",
                metric="mean_time",
                baseline_value=1.0,
                current_value=1.6,  # 60% increase
                change_percent=60.0,
                severity="critical",
                timestamp=datetime.now()
            )
            
            # Test is_regression property
            self.assertTrue(minor_regression.is_regression)
            self.assertTrue(major_regression.is_regression)
            self.assertTrue(critical_regression.is_regression)
            
            # Test severity classification
            self.assertEqual(minor_regression.severity, "minor")
            self.assertEqual(major_regression.severity, "major")
            self.assertEqual(critical_regression.severity, "critical")
            
        except NameError:
            self.skipTest("PerformanceRegression not available")

class TestBenchmarkRunner(unittest.TestCase):
    """Test benchmark runner functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Create minimal test config
        test_config = """
global:
  iterations: 1
  timeout_seconds: 10
  memory_limit_gb: 1
  enable_gpu: false
  enable_profiling: false

suites:
  test: ['core']
  
ci_cd:
  fail_on_regression: true
  regression_threshold: 15
"""
        with open(self.config_file, 'w') as f:
            f.write(test_config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_runner_initialization(self):
        """Test benchmark runner initialization."""
        try:
            runner = BenchmarkRunner(self.config_file)
            self.assertIsNotNone(runner)
            self.assertIsNotNone(runner.config)
            self.assertIsNotNone(runner.suite)
            self.assertIsNotNone(runner.analyzer)
        except NameError:
            self.skipTest("BenchmarkRunner not available")
    
    def test_system_info_collection(self):
        """Test system information collection."""
        try:
            runner = BenchmarkRunner(self.config_file)
            system_info = runner.get_system_info()
            
            # Check required fields
            self.assertIn('timestamp', system_info)
            self.assertIn('platform', system_info)
            self.assertIn('hardware', system_info)
            self.assertIn('environment', system_info)
            
            # Check platform info
            platform = system_info['platform']
            self.assertIn('system', platform)
            self.assertIn('python_version', platform)
            
            # Check hardware info
            hardware = system_info['hardware']
            self.assertIn('cpu_count', hardware)
            self.assertIn('memory_total_gb', hardware)
            
            # Check environment info
            environment = system_info['environment']
            self.assertIn('ci', environment)
            
        except NameError:
            self.skipTest("BenchmarkRunner not available")
    
    @patch('subprocess.run')
    def test_git_info_collection(self, mock_run):
        """Test git information collection."""
        try:
            # Mock git commands
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc123\n"
            
            runner = BenchmarkRunner(self.config_file)
            
            commit = runner._get_git_commit()
            self.assertEqual(commit, "abc123")
            
            branch = runner._get_git_branch()
            self.assertEqual(branch, "abc123")
            
        except NameError:
            self.skipTest("BenchmarkRunner not available")
    
    def test_results_saving(self):
        """Test benchmark results saving."""
        try:
            runner = BenchmarkRunner(self.config_file)
            
            # Create mock results
            runner.results = {
                'suite_name': 'test',
                'modules': ['core'],
                'total_tests': 1,
                'results': [
                    {
                        'test_name': 'mock_test',
                        'module_name': 'core',
                        'mean_time': 0.1,
                        'throughput': 10.0,
                        'memory_usage_mb': 5.0,
                        'error_rate': 0.0
                    }
                ]
            }
            
            # Test JSON saving
            json_file = os.path.join(self.temp_dir, "results.json")
            runner.save_results(json_file, 'json')
            
            self.assertTrue(os.path.exists(json_file))
            
            # Verify saved content
            with open(json_file, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data['suite_name'], 'test')
            self.assertEqual(len(saved_data['results']), 1)
            
        except NameError:
            self.skipTest("BenchmarkRunner not available")

class TestBenchmarkIntegration(unittest.TestCase):
    """Integration tests for the complete benchmark system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_benchmark_flow(self):
        """Test complete benchmark execution flow."""
        try:
            # This test requires all benchmark modules to be available
            from tests.benchmarks.benchmark_suite import BenchmarkSuite
            from tests.benchmarks.run_benchmarks import BenchmarkRunner
            
            # Create test config
            config_file = os.path.join(self.temp_dir, "integration_config.yaml")
            config_content = """
global:
  iterations: 1
  timeout_seconds: 5
  memory_limit_gb: 1
  enable_gpu: false
  enable_profiling: false

suites:
  minimal: ['core']
"""
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Initialize runner
            runner = BenchmarkRunner(config_file)
            
            # Mock the actual benchmark execution to avoid dependencies
            with patch.object(runner, '_run_module_benchmarks') as mock_benchmark:
                mock_benchmark.return_value = [
                    {
                        'test_name': 'integration_test',
                        'module_name': 'core',
                        'iterations': 1,
                        'mean_time': 0.05,
                        'std_time': 0.001,
                        'min_time': 0.049,
                        'max_time': 0.051,
                        'throughput': 20.0,
                        'memory_usage_mb': 8.0,
                        'cpu_usage_percent': 25.0,
                        'error_rate': 0.0
                    }
                ]
                
                # Run benchmark suite
                results = runner.run_benchmark_suite('minimal')
                
                # Verify results structure
                self.assertIn('suite_name', results)
                self.assertIn('modules', results)
                self.assertIn('system_info', results)
                self.assertIn('results', results)
                self.assertIn('summary', results)
                
                # Verify results content
                self.assertEqual(results['suite_name'], 'minimal')
                self.assertEqual(len(results['results']), 1)
                
                # Test saving results
                output_file = os.path.join(self.temp_dir, "integration_results.json")
                runner.save_results(output_file)
                self.assertTrue(os.path.exists(output_file))
                
                # Test report generation
                report_file = os.path.join(self.temp_dir, "integration_report.md")
                runner.generate_ci_report(report_file)
                self.assertTrue(os.path.exists(report_file))
                
                # Verify report content
                with open(report_file, 'r') as f:
                    report_content = f.read()
                
                self.assertIn('AIBF Benchmark Report', report_content)
                self.assertIn('integration_test', report_content)
                
        except ImportError:
            self.skipTest("Benchmark modules not available for integration test")

def run_mock_benchmarks():
    """Run mock benchmarks for testing purposes."""
    print("Running mock benchmark tests...")
    
    # Simulate benchmark execution
    mock_results = {
        'suite_name': 'mock_test',
        'modules': ['core'],
        'execution_time_seconds': 1.5,
        'total_tests': 3,
        'results': [
            {
                'test_name': 'mock_neural_network',
                'module_name': 'core',
                'iterations': 5,
                'mean_time': 0.1,
                'std_time': 0.01,
                'min_time': 0.09,
                'max_time': 0.11,
                'throughput': 10.0,
                'memory_usage_mb': 15.0,
                'cpu_usage_percent': 30.0,
                'error_rate': 0.0
            },
            {
                'test_name': 'mock_transformer',
                'module_name': 'core',
                'iterations': 5,
                'mean_time': 0.2,
                'std_time': 0.02,
                'min_time': 0.18,
                'max_time': 0.22,
                'throughput': 5.0,
                'memory_usage_mb': 25.0,
                'cpu_usage_percent': 45.0,
                'error_rate': 0.0
            },
            {
                'test_name': 'mock_tensor_ops',
                'module_name': 'core',
                'iterations': 5,
                'mean_time': 0.05,
                'std_time': 0.005,
                'min_time': 0.045,
                'max_time': 0.055,
                'throughput': 20.0,
                'memory_usage_mb': 8.0,
                'cpu_usage_percent': 20.0,
                'error_rate': 0.0
            }
        ],
        'summary': {
            'total_execution_time': 0.35,
            'average_execution_time': 0.117,
            'total_throughput': 35.0,
            'total_memory_usage_mb': 48.0,
            'average_error_rate': 0.0,
            'fastest_test': {'name': 'mock_tensor_ops', 'time': 0.05},
            'slowest_test': {'name': 'mock_transformer', 'time': 0.2},
            'median_time': 0.1,
            'tests_with_errors': 0
        }
    }
    
    # Save mock results
    with open('mock_benchmark_results.json', 'w') as f:
        json.dump(mock_results, f, indent=2)
    
    print("Mock benchmark results saved to mock_benchmark_results.json")
    print(f"Total tests: {mock_results['total_tests']}")
    print(f"Execution time: {mock_results['execution_time_seconds']:.2f}s")
    print(f"Average test time: {mock_results['summary']['average_execution_time']:.3f}s")
    print(f"Total throughput: {mock_results['summary']['total_throughput']:.1f} ops/s")
    
    return mock_results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="AIBF Benchmark Tests")
    parser.add_argument('--run-integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--run-mock', action='store_true',
                       help='Run mock benchmarks')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.run_mock:
        run_mock_benchmarks()
    else:
        # Run unit tests
        test_loader = unittest.TestLoader()
        test_suite = unittest.TestSuite()
        
        # Add test classes
        test_suite.addTests(test_loader.loadTestsFromTestCase(TestBenchmarkSuite))
        test_suite.addTests(test_loader.loadTestsFromTestCase(TestPerformanceAnalyzer))
        test_suite.addTests(test_loader.loadTestsFromTestCase(TestBenchmarkRunner))
        
        if args.run_integration:
            test_suite.addTests(test_loader.loadTestsFromTestCase(TestBenchmarkIntegration))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(test_suite)
        
        # Exit with appropriate code
        sys.exit(0 if result.wasSuccessful() else 1)