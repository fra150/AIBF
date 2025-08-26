#!/usr/bin/env python3
"""
AIBF Automated Benchmark Runner

Executes comprehensive benchmark suites with CI/CD integration,
automated reporting, and performance regression detection.

Usage:
    python run_benchmarks.py --suite all --output results/
    python run_benchmarks.py --suite core --compare-baseline baseline.json
    python run_benchmarks.py --ci-mode --fail-on-regression
    python run_benchmarks.py --profile --modules core,vision
"""

import argparse
import json
import logging
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.benchmark_suite import BenchmarkSuite
from tests.benchmarks.performance_analysis import PerformanceAnalyzer, PerformanceDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Automated benchmark execution and management."""
    
    def __init__(self, config_path: str = "benchmark_config.yaml"):
        self.config = self._load_config(config_path)
        self.suite = BenchmarkSuite(config_path)
        self.analyzer = PerformanceAnalyzer(config_path)
        self.db = PerformanceDatabase()
        self.start_time = None
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load benchmark configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if file not found."""
        return {
            'global': {
                'iterations': 10,
                'timeout_seconds': 300,
                'memory_limit_gb': 8,
                'enable_gpu': True,
                'enable_profiling': False
            },
            'suites': {
                'all': ['core', 'vision', 'healthcare', 'finance', 'education', 'multimodal', 'pipeline'],
                'core': ['core'],
                'ai': ['vision', 'healthcare', 'finance', 'education'],
                'integration': ['multimodal', 'pipeline']
            },
            'ci_cd': {
                'fail_on_regression': True,
                'regression_threshold': 15,
                'upload_artifacts': True
            }
        }
    
    def get_system_info(self) -> Dict:
        """Collect comprehensive system information."""
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'system': os.name,
                'platform': sys.platform,
                'python_version': sys.version,
                'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'environment': {
                'ci': os.getenv('CI', 'false').lower() == 'true',
                'github_actions': os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true',
                'git_commit': self._get_git_commit(),
                'git_branch': self._get_git_branch()
            }
        }
        
        # Add GPU information if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                system_info['hardware']['gpus'] = [
                    {
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_free_mb': gpu.memoryFree,
                        'temperature': gpu.temperature,
                        'load': gpu.load
                    }
                    for gpu in gpus
                ]
        except Exception as e:
            logger.debug(f"Could not get GPU info: {e}")
            system_info['hardware']['gpus'] = []
        
        return system_info
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def run_benchmark_suite(self, suite_name: str, modules: List[str] = None, 
                           parallel: bool = True) -> Dict:
        """Run a complete benchmark suite."""
        logger.info(f"Starting benchmark suite: {suite_name}")
        self.start_time = time.time()
        
        # Determine modules to benchmark
        if modules:
            target_modules = modules
        else:
            target_modules = self.config.get('suites', {}).get(suite_name, ['core'])
        
        logger.info(f"Benchmarking modules: {target_modules}")
        
        # Collect system information
        system_info = self.get_system_info()
        
        # Run benchmarks
        all_results = []
        
        if parallel and len(target_modules) > 1:
            all_results = self._run_parallel_benchmarks(target_modules)
        else:
            all_results = self._run_sequential_benchmarks(target_modules)
        
        # Compile final results
        total_time = time.time() - self.start_time
        
        final_results = {
            'suite_name': suite_name,
            'modules': target_modules,
            'system_info': system_info,
            'execution_time_seconds': total_time,
            'total_tests': len(all_results),
            'results': all_results,
            'summary': self._generate_summary(all_results)
        }
        
        self.results = final_results
        logger.info(f"Benchmark suite completed in {total_time:.2f} seconds")
        
        return final_results
    
    def _run_parallel_benchmarks(self, modules: List[str]) -> List[Dict]:
        """Run benchmarks in parallel across modules."""
        all_results = []
        max_workers = min(len(modules), psutil.cpu_count())
        
        logger.info(f"Running benchmarks in parallel with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit benchmark tasks
            future_to_module = {
                executor.submit(self._run_module_benchmarks, module): module
                for module in modules
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                try:
                    module_results = future.result()
                    all_results.extend(module_results)
                    logger.info(f"Completed benchmarks for module: {module}")
                except Exception as e:
                    logger.error(f"Benchmark failed for module {module}: {e}")
        
        return all_results
    
    def _run_sequential_benchmarks(self, modules: List[str]) -> List[Dict]:
        """Run benchmarks sequentially."""
        all_results = []
        
        for module in modules:
            logger.info(f"Running benchmarks for module: {module}")
            try:
                module_results = self._run_module_benchmarks(module)
                all_results.extend(module_results)
                logger.info(f"Completed benchmarks for module: {module}")
            except Exception as e:
                logger.error(f"Benchmark failed for module {module}: {e}")
        
        return all_results
    
    def _run_module_benchmarks(self, module: str) -> List[Dict]:
        """Run all benchmarks for a specific module."""
        if module == 'core':
            return self.suite.benchmark_core_modules()
        elif module == 'vision':
            return self.suite.benchmark_vision_modules()
        elif module == 'healthcare':
            return self.suite.benchmark_healthcare_modules()
        elif module == 'finance':
            return self.suite.benchmark_finance_modules()
        elif module == 'education':
            return self.suite.benchmark_education_modules()
        elif module == 'multimodal':
            return self.suite.benchmark_multimodal_modules()
        elif module == 'pipeline':
            return self.suite.benchmark_pipeline_modules()
        else:
            logger.warning(f"Unknown module: {module}")
            return []
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics for benchmark results."""
        if not results:
            return {}
        
        # Calculate aggregate statistics
        total_time = sum(r['mean_time'] for r in results)
        total_throughput = sum(r['throughput'] for r in results)
        total_memory = sum(r['memory_usage_mb'] for r in results)
        avg_error_rate = sum(r['error_rate'] for r in results) / len(results)
        
        # Find performance outliers
        times = [r['mean_time'] for r in results]
        times.sort()
        
        summary = {
            'total_execution_time': total_time,
            'average_execution_time': total_time / len(results),
            'total_throughput': total_throughput,
            'total_memory_usage_mb': total_memory,
            'average_error_rate': avg_error_rate,
            'fastest_test': {
                'name': min(results, key=lambda x: x['mean_time'])['test_name'],
                'time': min(times)
            },
            'slowest_test': {
                'name': max(results, key=lambda x: x['mean_time'])['test_name'],
                'time': max(times)
            },
            'median_time': times[len(times) // 2],
            'tests_with_errors': len([r for r in results if r['error_rate'] > 0])
        }
        
        return summary
    
    def save_results(self, output_path: str, format: str = 'json'):
        """Save benchmark results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        elif format.lower() == 'yaml':
            with open(output_file, 'w') as f:
                yaml.dump(self.results, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {output_file}")
    
    def compare_with_baseline(self, baseline_file: str) -> List:
        """Compare current results with baseline."""
        if not self.results:
            raise ValueError("No current results to compare")
        
        # Save current results temporarily
        current_file = "temp_current_results.json"
        self.save_results(current_file)
        
        try:
            # Perform comparison
            regressions = self.analyzer.compare_results(baseline_file, current_file)
            
            if regressions:
                logger.warning(f"Detected {len(regressions)} performance regressions")
                for regression in regressions:
                    logger.warning(
                        f"  {regression.module_name}.{regression.test_name} - "
                        f"{regression.metric}: {regression.change_percent:+.1f}% "
                        f"({regression.severity})"
                    )
            else:
                logger.info("No performance regressions detected")
            
            return regressions
        
        finally:
            # Clean up temporary file
            if os.path.exists(current_file):
                os.remove(current_file)
    
    def store_in_database(self):
        """Store results in performance database."""
        if not self.results:
            raise ValueError("No results to store")
        
        metadata = {
            'git_commit': self.results['system_info']['environment']['git_commit'],
            'branch': self.results['system_info']['environment']['git_branch'],
            'environment': 'ci' if self.results['system_info']['environment']['ci'] else 'local'
        }
        
        run_id = self.db.store_benchmark_run(self.results, metadata)
        logger.info(f"Results stored in database with run_id: {run_id}")
        
        return run_id
    
    def generate_ci_report(self, output_file: str = "benchmark_report.md"):
        """Generate CI-friendly markdown report."""
        if not self.results:
            raise ValueError("No results to report")
        
        summary = self.results['summary']
        system_info = self.results['system_info']
        
        report = f"""
# AIBF Benchmark Report

**Suite:** {self.results['suite_name']}  
**Timestamp:** {system_info['timestamp']}  
**Commit:** {system_info['environment']['git_commit'][:8] if system_info['environment']['git_commit'] else 'unknown'}  
**Branch:** {system_info['environment']['git_branch'] or 'unknown'}  

## Summary

- **Total Tests:** {self.results['total_tests']}
- **Total Execution Time:** {summary['total_execution_time']:.2f}s
- **Average Test Time:** {summary['average_execution_time']:.4f}s
- **Total Throughput:** {summary['total_throughput']:.2f} ops/s
- **Average Error Rate:** {summary['average_error_rate']:.2%}
- **Tests with Errors:** {summary['tests_with_errors']}

## Performance Highlights

- **Fastest Test:** {summary['fastest_test']['name']} ({summary['fastest_test']['time']:.4f}s)
- **Slowest Test:** {summary['slowest_test']['name']} ({summary['slowest_test']['time']:.4f}s)
- **Median Time:** {summary['median_time']:.4f}s

## System Information

- **Platform:** {system_info['platform']['system']} ({system_info['platform']['architecture']})
- **Python:** {system_info['platform']['python_version'].split()[0]}
- **CPU Cores:** {system_info['hardware']['cpu_count']}
- **Memory:** {system_info['hardware']['memory_total_gb']:.1f}GB total, {system_info['hardware']['memory_available_gb']:.1f}GB available
"""
        
        # Add GPU information if available
        if system_info['hardware'].get('gpus'):
            report += "\n## GPU Information\n\n"
            for i, gpu in enumerate(system_info['hardware']['gpus']):
                report += f"- **GPU {i}:** {gpu['name']} ({gpu['memory_total_mb']}MB)\n"
        
        # Add detailed results table
        report += "\n## Detailed Results\n\n"
        report += "| Module | Test | Mean Time (s) | Throughput (ops/s) | Memory (MB) | Error Rate |\n"
        report += "|--------|------|---------------|-------------------|-------------|------------|\n"
        
        for result in self.results['results']:
            report += f"| {result['module_name']} | {result['test_name']} | "
            report += f"{result['mean_time']:.4f} | {result['throughput']:.2f} | "
            report += f"{result['memory_usage_mb']:.2f} | {result['error_rate']:.2%} |\n"
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"CI report generated: {output_file}")
    
    def run_ci_mode(self, suite_name: str, baseline_file: str = None, 
                   fail_on_regression: bool = True) -> int:
        """Run benchmarks in CI mode with regression detection."""
        logger.info("Running benchmarks in CI mode")
        
        try:
            # Run benchmark suite
            results = self.run_benchmark_suite(suite_name)
            
            # Save results with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"benchmark_results_{timestamp}.json"
            self.save_results(results_file)
            
            # Store in database
            self.store_in_database()
            
            # Generate CI report
            self.generate_ci_report()
            
            # Check for regressions if baseline provided
            regressions = []
            if baseline_file and os.path.exists(baseline_file):
                regressions = self.compare_with_baseline(baseline_file)
            
            # Determine exit code
            if fail_on_regression and regressions:
                critical_regressions = [r for r in regressions if r.severity == 'critical']
                if critical_regressions:
                    logger.error(f"Critical performance regressions detected: {len(critical_regressions)}")
                    return 1
            
            # Check for test failures
            failed_tests = [r for r in results['results'] if r['error_rate'] > 0]
            if failed_tests:
                logger.error(f"Tests with errors: {len(failed_tests)}")
                return 1
            
            logger.info("All benchmarks completed successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            return 1

def main():
    """Main benchmark runner execution."""
    parser = argparse.ArgumentParser(description="AIBF Automated Benchmark Runner")
    parser.add_argument("--suite", default="core", 
                       help="Benchmark suite to run (all, core, ai, integration)")
    parser.add_argument("--modules", 
                       help="Comma-separated list of specific modules to benchmark")
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--format", default="json", choices=["json", "yaml"],
                       help="Output format")
    parser.add_argument("--baseline", 
                       help="Baseline file for regression comparison")
    parser.add_argument("--ci-mode", action="store_true",
                       help="Run in CI mode with automated reporting")
    parser.add_argument("--fail-on-regression", action="store_true",
                       help="Fail if performance regressions detected")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run benchmarks in parallel")
    parser.add_argument("--config", default="benchmark_config.yaml",
                       help="Configuration file")
    parser.add_argument("--profile", action="store_true",
                       help="Enable detailed profiling")
    parser.add_argument("--store-db", action="store_true",
                       help="Store results in performance database")
    
    args = parser.parse_args()
    
    # Parse modules if provided
    modules = None
    if args.modules:
        modules = [m.strip() for m in args.modules.split(',')]
    
    runner = BenchmarkRunner(args.config)
    
    try:
        if args.ci_mode:
            # Run in CI mode
            exit_code = runner.run_ci_mode(
                args.suite, 
                args.baseline, 
                args.fail_on_regression
            )
            sys.exit(exit_code)
        else:
            # Run normal benchmark
            results = runner.run_benchmark_suite(
                args.suite, 
                modules, 
                args.parallel
            )
            
            # Save results
            runner.save_results(args.output, args.format)
            
            # Store in database if requested
            if args.store_db:
                runner.store_in_database()
            
            # Compare with baseline if provided
            if args.baseline:
                regressions = runner.compare_with_baseline(args.baseline)
                
                if regressions and args.fail_on_regression:
                    critical_regressions = [r for r in regressions if r.severity == 'critical']
                    if critical_regressions:
                        logger.error("Critical regressions detected")
                        sys.exit(1)
            
            # Generate report
            runner.generate_ci_report()
            
            print(f"\nBenchmark completed successfully!")
            print(f"Results saved to: {args.output}")
            print(f"Total tests: {results['total_tests']}")
            print(f"Execution time: {results['execution_time_seconds']:.2f}s")
            
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()