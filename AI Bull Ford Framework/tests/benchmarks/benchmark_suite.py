#!/usr/bin/env python3
"""
AIBF Performance Benchmark Suite

Comprehensive performance testing for all AIBF modules including:
- Core neural networks
- Computer vision
- Healthcare AI
- Financial analysis
- Educational systems
- Multimodal processing

Usage:
    python benchmark_suite.py --module all
    python benchmark_suite.py --module core --iterations 100
    python benchmark_suite.py --module healthcare --profile
"""

import argparse
import time
import json
import logging
import statistics
import psutil
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

# AIBF imports
from aibf.core.neural_networks import NeuralNetwork, Transformer
from aibf.core.reinforcement_learning import RLAgent
from aibf.vision.encoders import VisionEncoder
from aibf.healthcare.medical_imaging import MedicalImageAnalyzer
from aibf.healthcare.patient_data import PatientDataProcessor
from aibf.financial.risk_management import RiskManager
from aibf.financial.portfolio_optimization import PortfolioOptimizer
from aibf.educational.learner_profiling import LearnerProfiler
from aibf.educational.content_recommendation import ContentRecommendationEngine
from aibf.multimodal.fusion import MultimodalFusionEngine
from aibf.utils.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    module_name: str
    test_name: str
    iterations: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    error_rate: float = 0.0
    metadata: Dict[str, Any] = None

class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
    
    @contextmanager
    def monitor(self):
        """Context manager for performance monitoring."""
        # Initial measurements
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            
            # Final measurements
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self.process.cpu_percent()
            
            self.last_duration = end_time - start_time
            self.last_memory_usage = end_memory - start_memory
            self.last_cpu_usage = end_cpu
            
            if self.gpu_available:
                self.last_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            else:
                self.last_gpu_memory = None

class BenchmarkSuite:
    """Main benchmark suite for AIBF framework."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.monitor = PerformanceMonitor()
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, 
                     func: Callable, 
                     module_name: str, 
                     test_name: str, 
                     iterations: int = 10,
                     warmup_iterations: int = 3) -> BenchmarkResult:
        """Run a single benchmark test."""
        logger.info(f"Running benchmark: {module_name}.{test_name} ({iterations} iterations)")
        
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                func()
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Actual benchmark runs
        times = []
        errors = 0
        memory_usage = []
        cpu_usage = []
        gpu_usage = []
        
        for i in range(iterations):
            try:
                with self.monitor.monitor():
                    func()
                
                times.append(self.monitor.last_duration)
                memory_usage.append(self.monitor.last_memory_usage)
                cpu_usage.append(self.monitor.last_cpu_usage)
                
                if self.monitor.last_gpu_memory is not None:
                    gpu_usage.append(self.monitor.last_gpu_memory)
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i+1} failed: {e}")
                errors += 1
        
        if not times:
            raise RuntimeError(f"All benchmark iterations failed for {module_name}.{test_name}")
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        throughput = 1.0 / mean_time if mean_time > 0 else 0.0
        error_rate = errors / iterations
        
        avg_memory = statistics.mean(memory_usage) if memory_usage else 0.0
        avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0.0
        avg_gpu = statistics.mean(gpu_usage) if gpu_usage else None
        
        result = BenchmarkResult(
            module_name=module_name,
            test_name=test_name,
            iterations=iterations,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            throughput=throughput,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=avg_cpu,
            gpu_usage_percent=avg_gpu,
            error_rate=error_rate
        )
        
        self.results.append(result)
        logger.info(f"Completed: {mean_time:.4f}s ± {std_time:.4f}s, {throughput:.2f} ops/s")
        
        return result
    
    def benchmark_core_neural_networks(self, iterations: int = 10):
        """Benchmark core neural network operations."""
        logger.info("Benchmarking core neural networks...")
        
        # Neural Network Forward Pass
        def nn_forward():
            nn = NeuralNetwork({
                "layers": [784, 256, 128, 10],
                "activation": "relu",
                "dropout": 0.2
            })
            x = torch.randn(32, 784)  # Batch of 32
            with torch.no_grad():
                output = nn.forward(x)
            return output
        
        self.run_benchmark(nn_forward, "core", "neural_network_forward", iterations)
        
        # Transformer Attention
        def transformer_attention():
            transformer = Transformer({
                "d_model": 512,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 2048
            })
            x = torch.randn(10, 32, 512)  # seq_len=10, batch=32, d_model=512
            with torch.no_grad():
                output = transformer.forward(x)
            return output
        
        self.run_benchmark(transformer_attention, "core", "transformer_attention", iterations)
        
        # Reinforcement Learning Step
        def rl_step():
            agent = RLAgent({
                "state_dim": 84,
                "action_dim": 4,
                "algorithm": "dqn",
                "learning_rate": 0.001
            })
            state = np.random.rand(84)
            action = agent.select_action(state)
            return action
        
        self.run_benchmark(rl_step, "core", "rl_action_selection", iterations)
    
    def benchmark_computer_vision(self, iterations: int = 10):
        """Benchmark computer vision operations."""
        logger.info("Benchmarking computer vision...")
        
        # Vision Encoder
        def vision_encoding():
            encoder = VisionEncoder({
                "model_type": "resnet",
                "pretrained": False,
                "feature_dim": 512
            })
            images = torch.randn(8, 3, 224, 224)  # Batch of 8 images
            with torch.no_grad():
                features = encoder.encode(images)
            return features
        
        self.run_benchmark(vision_encoding, "vision", "image_encoding", iterations)
        
        # Batch Image Processing
        def batch_processing():
            encoder = VisionEncoder({"model_type": "cnn"})
            batch_sizes = [1, 4, 8, 16]
            for batch_size in batch_sizes:
                images = torch.randn(batch_size, 3, 224, 224)
                with torch.no_grad():
                    encoder.encode(images)
        
        self.run_benchmark(batch_processing, "vision", "batch_processing", iterations)
    
    def benchmark_healthcare_ai(self, iterations: int = 10):
        """Benchmark healthcare AI operations."""
        logger.info("Benchmarking healthcare AI...")
        
        # Medical Image Analysis
        def medical_analysis():
            analyzer = MedicalImageAnalyzer({
                "model_type": "medical_cnn",
                "num_classes": 5,
                "confidence_threshold": 0.8
            })
            medical_image = np.random.rand(512, 512, 3).astype(np.float32)
            result = analyzer.analyze_image(medical_image)
            return result
        
        self.run_benchmark(medical_analysis, "healthcare", "medical_image_analysis", iterations)
        
        # Patient Data Processing
        def patient_processing():
            processor = PatientDataProcessor({
                "anonymization": True,
                "validation": True,
                "feature_extraction": True
            })
            patient_data = {
                "age": 45,
                "gender": "M",
                "symptoms": ["chest_pain", "shortness_of_breath"],
                "vitals": {"bp": "140/90", "hr": 85, "temp": 98.6},
                "lab_results": {"glucose": 120, "cholesterol": 200}
            }
            result = processor.process_patient_data(patient_data)
            return result
        
        self.run_benchmark(patient_processing, "healthcare", "patient_data_processing", iterations)
    
    def benchmark_financial_analysis(self, iterations: int = 10):
        """Benchmark financial analysis operations."""
        logger.info("Benchmarking financial analysis...")
        
        # Risk Management
        def risk_analysis():
            risk_manager = RiskManager({
                "var_confidence": 0.95,
                "time_horizon": 252,
                "monte_carlo_simulations": 1000
            })
            portfolio_data = {
                "assets": [
                    {"symbol": "AAPL", "weight": 0.3, "returns": np.random.normal(0.001, 0.02, 252)},
                    {"symbol": "GOOGL", "weight": 0.4, "returns": np.random.normal(0.0008, 0.025, 252)},
                    {"symbol": "MSFT", "weight": 0.3, "returns": np.random.normal(0.0012, 0.018, 252)}
                ]
            }
            risk_metrics = risk_manager.calculate_portfolio_risk(portfolio_data)
            return risk_metrics
        
        self.run_benchmark(risk_analysis, "financial", "risk_analysis", iterations)
        
        # Portfolio Optimization
        def portfolio_optimization():
            optimizer = PortfolioOptimizer({
                "optimization_method": "mean_variance",
                "constraints": {"max_weight": 0.4, "min_weight": 0.05},
                "target_return": 0.10
            })
            market_data = {
                "returns": np.random.normal(0.001, 0.02, (252, 10)),  # 252 days, 10 assets
                "symbols": [f"STOCK_{i}" for i in range(10)]
            }
            optimal_weights = optimizer.optimize_portfolio(market_data)
            return optimal_weights
        
        self.run_benchmark(portfolio_optimization, "financial", "portfolio_optimization", iterations)
    
    def benchmark_educational_systems(self, iterations: int = 10):
        """Benchmark educational AI systems."""
        logger.info("Benchmarking educational systems...")
        
        # Learner Profiling
        def learner_profiling():
            profiler = LearnerProfiler({
                "profiling_algorithm": "collaborative_filtering",
                "feature_extraction": True,
                "personality_modeling": True
            })
            learner_data = {
                "learning_history": [
                    {"course_id": f"course_{i}", "completion_rate": np.random.uniform(0.7, 1.0), 
                     "score": np.random.uniform(70, 100)} for i in range(20)
                ],
                "preferences": {"pace": "fast", "style": "visual"},
                "demographics": {"age": 25, "education": "bachelor"}
            }
            profile = profiler.create_learner_profile(learner_data)
            return profile
        
        self.run_benchmark(learner_profiling, "educational", "learner_profiling", iterations)
        
        # Content Recommendation
        def content_recommendation():
            recommender = ContentRecommendationEngine({
                "algorithm": "hybrid",
                "max_recommendations": 10,
                "diversity_factor": 0.3
            })
            learner_profile = {
                "interests": ["machine_learning", "data_science"],
                "skill_level": "intermediate",
                "learning_style": "hands_on"
            }
            content_catalog = [
                {"id": f"content_{i}", "topic": "ml", "difficulty": "intermediate", 
                 "type": "video", "rating": np.random.uniform(3.5, 5.0)} for i in range(100)
            ]
            recommendations = recommender.recommend_content(learner_profile, content_catalog)
            return recommendations
        
        self.run_benchmark(content_recommendation, "educational", "content_recommendation", iterations)
    
    def benchmark_multimodal_processing(self, iterations: int = 10):
        """Benchmark multimodal AI processing."""
        logger.info("Benchmarking multimodal processing...")
        
        # Multimodal Fusion
        def multimodal_fusion():
            fusion_engine = MultimodalFusionEngine({
                "fusion_method": "attention",
                "modalities": ["text", "image", "audio"],
                "output_dim": 512
            })
            
            # Simulate multimodal features
            text_features = torch.randn(1, 768)  # BERT-like features
            image_features = torch.randn(1, 2048)  # ResNet-like features
            audio_features = torch.randn(1, 512)  # Audio features
            
            multimodal_input = {
                "text": text_features,
                "image": image_features,
                "audio": audio_features
            }
            
            with torch.no_grad():
                fused_features = fusion_engine.fuse_modalities(multimodal_input)
            return fused_features
        
        self.run_benchmark(multimodal_fusion, "multimodal", "modality_fusion", iterations)
        
        # Cross-Modal Attention
        def cross_modal_attention():
            # Simulate cross-modal attention computation
            text_seq = torch.randn(1, 50, 768)  # 50 tokens, 768 dim
            image_patches = torch.randn(1, 196, 768)  # 14x14 patches, 768 dim
            
            # Simple cross-attention
            attention_weights = torch.softmax(
                torch.matmul(text_seq, image_patches.transpose(-2, -1)) / np.sqrt(768), 
                dim=-1
            )
            attended_features = torch.matmul(attention_weights, image_patches)
            return attended_features
        
        self.run_benchmark(cross_modal_attention, "multimodal", "cross_modal_attention", iterations)
    
    def benchmark_pipeline_performance(self, iterations: int = 10):
        """Benchmark pipeline orchestration performance."""
        logger.info("Benchmarking pipeline performance...")
        
        # Sequential Pipeline
        def sequential_pipeline():
            pipeline = Pipeline("benchmark_pipeline")
            
            # Add mock stages
            class MockStage:
                def __init__(self, delay=0.001):
                    self.delay = delay
                
                def process(self, data):
                    time.sleep(self.delay)
                    return {"processed": True, "data": data}
            
            pipeline.add_stage("stage1", MockStage(0.001))
            pipeline.add_stage("stage2", MockStage(0.002))
            pipeline.add_stage("stage3", MockStage(0.001))
            
            result = pipeline.process({"input": "test_data"})
            return result
        
        self.run_benchmark(sequential_pipeline, "pipeline", "sequential_processing", iterations)
        
        # Parallel Pipeline
        def parallel_pipeline():
            # Simulate parallel processing
            import concurrent.futures
            
            def process_chunk(chunk_id):
                time.sleep(0.001)  # Simulate processing
                return f"processed_chunk_{chunk_id}"
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_chunk, i) for i in range(8)]
                results = [future.result() for future in futures]
            
            return results
        
        self.run_benchmark(parallel_pipeline, "pipeline", "parallel_processing", iterations)
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        results_data = {
            "timestamp": time.time(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            "results": [asdict(result) for result in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def generate_report(self, save_plots: bool = True):
        """Generate comprehensive benchmark report."""
        if not self.results:
            logger.warning("No benchmark results to report")
            return
        
        # Create summary statistics
        summary = {
            "total_tests": len(self.results),
            "modules_tested": len(set(r.module_name for r in self.results)),
            "total_runtime": sum(r.mean_time * r.iterations for r in self.results),
            "avg_throughput": statistics.mean([r.throughput for r in self.results]),
            "avg_memory_usage": statistics.mean([r.memory_usage_mb for r in self.results])
        }
        
        # Print summary
        print("\n" + "="*60)
        print("AIBF PERFORMANCE BENCHMARK REPORT")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Modules Tested: {summary['modules_tested']}")
        print(f"Total Runtime: {summary['total_runtime']:.2f}s")
        print(f"Average Throughput: {summary['avg_throughput']:.2f} ops/s")
        print(f"Average Memory Usage: {summary['avg_memory_usage']:.2f} MB")
        print("\n")
        
        # Print detailed results
        print("DETAILED RESULTS:")
        print("-" * 60)
        for result in sorted(self.results, key=lambda x: (x.module_name, x.test_name)):
            print(f"{result.module_name}.{result.test_name}:")
            print(f"  Time: {result.mean_time:.4f}s ± {result.std_time:.4f}s")
            print(f"  Throughput: {result.throughput:.2f} ops/s")
            print(f"  Memory: {result.memory_usage_mb:.2f} MB")
            if result.error_rate > 0:
                print(f"  Error Rate: {result.error_rate:.2%}")
            print()
        
        if save_plots:
            self._generate_plots()
    
    def _generate_plots(self):
        """Generate performance visualization plots."""
        if not self.results:
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AIBF Performance Benchmark Results', fontsize=16)
        
        # Plot 1: Execution time by module
        modules = [r.module_name for r in self.results]
        times = [r.mean_time for r in self.results]
        
        axes[0, 0].bar(range(len(modules)), times)
        axes[0, 0].set_xlabel('Test')
        axes[0, 0].set_ylabel('Execution Time (s)')
        axes[0, 0].set_title('Execution Time by Test')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Throughput comparison
        throughputs = [r.throughput for r in self.results]
        axes[0, 1].bar(range(len(modules)), throughputs)
        axes[0, 1].set_xlabel('Test')
        axes[0, 1].set_ylabel('Throughput (ops/s)')
        axes[0, 1].set_title('Throughput by Test')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Memory usage
        memory_usage = [r.memory_usage_mb for r in self.results]
        axes[1, 0].bar(range(len(modules)), memory_usage)
        axes[1, 0].set_xlabel('Test')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage by Test')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance vs Memory scatter
        axes[1, 1].scatter(memory_usage, throughputs, alpha=0.7)
        axes[1, 1].set_xlabel('Memory Usage (MB)')
        axes[1, 1].set_ylabel('Throughput (ops/s)')
        axes[1, 1].set_title('Performance vs Memory Usage')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "benchmark_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {plot_path}")
        plt.close()

def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="AIBF Performance Benchmark Suite")
    parser.add_argument("--module", choices=["all", "core", "vision", "healthcare", 
                                           "financial", "educational", "multimodal", "pipeline"],
                       default="all", help="Module to benchmark")
    parser.add_argument("--iterations", type=int, default=10, 
                       help="Number of iterations per test")
    parser.add_argument("--output-dir", default="benchmark_results", 
                       help="Output directory for results")
    parser.add_argument("--profile", action="store_true", 
                       help="Enable detailed profiling")
    parser.add_argument("--save-plots", action="store_true", default=True,
                       help="Generate and save performance plots")
    
    args = parser.parse_args()
    
    # Initialize benchmark suite
    suite = BenchmarkSuite(args.output_dir)
    
    try:
        # Run selected benchmarks
        if args.module in ["all", "core"]:
            suite.benchmark_core_neural_networks(args.iterations)
        
        if args.module in ["all", "vision"]:
            suite.benchmark_computer_vision(args.iterations)
        
        if args.module in ["all", "healthcare"]:
            suite.benchmark_healthcare_ai(args.iterations)
        
        if args.module in ["all", "financial"]:
            suite.benchmark_financial_analysis(args.iterations)
        
        if args.module in ["all", "educational"]:
            suite.benchmark_educational_systems(args.iterations)
        
        if args.module in ["all", "multimodal"]:
            suite.benchmark_multimodal_processing(args.iterations)
        
        if args.module in ["all", "pipeline"]:
            suite.benchmark_pipeline_performance(args.iterations)
        
        # Save results and generate report
        suite.save_results()
        suite.generate_report(args.save_plots)
        
        logger.info("Benchmark suite completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        raise

if __name__ == "__main__":
    main()