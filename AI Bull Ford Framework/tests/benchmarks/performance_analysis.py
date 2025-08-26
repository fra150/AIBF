#!/usr/bin/env python3
"""
AIBF Performance Analysis and Regression Detection

Analyzes benchmark results, detects performance regressions,
and generates comparative reports across different versions.

Usage:
    python performance_analysis.py --baseline baseline.json --current current.json
    python performance_analysis.py --history-dir results/ --detect-regressions
    python performance_analysis.py --generate-trends --output trends_report.html
"""

import argparse
import json
import logging
import statistics
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceRegression:
    """Represents a detected performance regression."""
    test_name: str
    module_name: str
    metric: str
    baseline_value: float
    current_value: float
    change_percent: float
    severity: str  # 'minor', 'major', 'critical'
    timestamp: datetime
    
    @property
    def is_regression(self) -> bool:
        """Check if this represents a performance regression."""
        if self.metric in ['mean_time', 'memory_usage_mb']:
            return self.change_percent > 0  # Increase is bad
        elif self.metric in ['throughput']:
            return self.change_percent < 0  # Decrease is bad
        return False

class PerformanceDatabase:
    """SQLite database for storing benchmark history."""
    
    def __init__(self, db_path: str = "benchmark_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    git_commit TEXT,
                    branch TEXT,
                    environment TEXT,
                    system_info TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    module_name TEXT,
                    test_name TEXT,
                    iterations INTEGER,
                    mean_time REAL,
                    std_time REAL,
                    min_time REAL,
                    max_time REAL,
                    throughput REAL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    gpu_usage_percent REAL,
                    error_rate REAL,
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_regressions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    test_name TEXT,
                    module_name TEXT,
                    metric TEXT,
                    baseline_value REAL,
                    current_value REAL,
                    change_percent REAL,
                    severity TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
    
    def store_benchmark_run(self, results_data: Dict, metadata: Dict = None) -> int:
        """Store a complete benchmark run."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert run metadata
            cursor = conn.execute("""
                INSERT INTO benchmark_runs (git_commit, branch, environment, system_info)
                VALUES (?, ?, ?, ?)
            """, (
                metadata.get('git_commit') if metadata else None,
                metadata.get('branch') if metadata else None,
                metadata.get('environment') if metadata else None,
                json.dumps(results_data.get('system_info', {}))
            ))
            
            run_id = cursor.lastrowid
            
            # Insert individual results
            for result in results_data.get('results', []):
                conn.execute("""
                    INSERT INTO benchmark_results (
                        run_id, module_name, test_name, iterations,
                        mean_time, std_time, min_time, max_time,
                        throughput, memory_usage_mb, cpu_usage_percent,
                        gpu_usage_percent, error_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    result['module_name'],
                    result['test_name'],
                    result['iterations'],
                    result['mean_time'],
                    result['std_time'],
                    result['min_time'],
                    result['max_time'],
                    result['throughput'],
                    result['memory_usage_mb'],
                    result['cpu_usage_percent'],
                    result.get('gpu_usage_percent'),
                    result['error_rate']
                ))
            
            return run_id
    
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical benchmark data."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    br.timestamp,
                    br.git_commit,
                    br.branch,
                    br.environment,
                    res.module_name,
                    res.test_name,
                    res.mean_time,
                    res.throughput,
                    res.memory_usage_mb,
                    res.error_rate
                FROM benchmark_runs br
                JOIN benchmark_results res ON br.id = res.run_id
                WHERE br.timestamp >= datetime('now', '-{} days')
                ORDER BY br.timestamp DESC
            """.format(days)
            
            return pd.read_sql_query(query, conn, parse_dates=['timestamp'])
    
    def store_regression(self, regression: PerformanceRegression):
        """Store a detected performance regression."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_regressions (
                    test_name, module_name, metric, baseline_value,
                    current_value, change_percent, severity
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                regression.test_name,
                regression.module_name,
                regression.metric,
                regression.baseline_value,
                regression.current_value,
                regression.change_percent,
                regression.severity
            ))

class PerformanceAnalyzer:
    """Analyzes benchmark results and detects regressions."""
    
    def __init__(self, config_path: str = "benchmark_config.yaml"):
        self.config = self._load_config(config_path)
        self.db = PerformanceDatabase()
        
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
            'thresholds': {},
            'reporting': {
                'regression_detection': {
                    'threshold_percent': 10,
                    'consecutive_failures': 3
                }
            }
        }
    
    def compare_results(self, baseline_file: str, current_file: str) -> List[PerformanceRegression]:
        """Compare two benchmark result files."""
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        
        with open(current_file, 'r') as f:
            current_data = json.load(f)
        
        return self._detect_regressions(baseline_data, current_data)
    
    def _detect_regressions(self, baseline_data: Dict, current_data: Dict) -> List[PerformanceRegression]:
        """Detect performance regressions between two datasets."""
        regressions = []
        
        # Create lookup dictionaries
        baseline_lookup = {
            (r['module_name'], r['test_name']): r 
            for r in baseline_data.get('results', [])
        }
        
        current_lookup = {
            (r['module_name'], r['test_name']): r 
            for r in current_data.get('results', [])
        }
        
        # Compare common tests
        common_tests = set(baseline_lookup.keys()) & set(current_lookup.keys())
        
        for test_key in common_tests:
            baseline_result = baseline_lookup[test_key]
            current_result = current_lookup[test_key]
            
            # Check different metrics
            metrics_to_check = ['mean_time', 'throughput', 'memory_usage_mb']
            
            for metric in metrics_to_check:
                if metric in baseline_result and metric in current_result:
                    baseline_value = baseline_result[metric]
                    current_value = current_result[metric]
                    
                    if baseline_value > 0:  # Avoid division by zero
                        change_percent = ((current_value - baseline_value) / baseline_value) * 100
                        
                        # Check if this is a significant regression
                        threshold = self._get_threshold(test_key[0], test_key[1], metric)
                        
                        if abs(change_percent) > threshold:
                            severity = self._classify_severity(change_percent, threshold)
                            
                            regression = PerformanceRegression(
                                test_name=test_key[1],
                                module_name=test_key[0],
                                metric=metric,
                                baseline_value=baseline_value,
                                current_value=current_value,
                                change_percent=change_percent,
                                severity=severity,
                                timestamp=datetime.now()
                            )
                            
                            if regression.is_regression:
                                regressions.append(regression)
                                self.db.store_regression(regression)
        
        return regressions
    
    def _get_threshold(self, module: str, test: str, metric: str) -> float:
        """Get regression threshold for a specific test and metric."""
        thresholds = self.config.get('thresholds', {})
        module_thresholds = thresholds.get(module, {})
        test_thresholds = module_thresholds.get(test, {})
        
        # Default threshold from config
        default_threshold = self.config.get('reporting', {}).get(
            'regression_detection', {}).get('threshold_percent', 10)
        
        # Metric-specific thresholds
        metric_thresholds = {
            'mean_time': test_thresholds.get('max_time_ms', default_threshold),
            'throughput': test_thresholds.get('min_throughput', default_threshold),
            'memory_usage_mb': test_thresholds.get('max_memory_mb', default_threshold)
        }
        
        return metric_thresholds.get(metric, default_threshold)
    
    def _classify_severity(self, change_percent: float, threshold: float) -> str:
        """Classify the severity of a performance regression."""
        abs_change = abs(change_percent)
        
        if abs_change >= threshold * 3:
            return 'critical'
        elif abs_change >= threshold * 2:
            return 'major'
        else:
            return 'minor'
    
    def generate_trend_analysis(self, days: int = 30) -> Dict:
        """Generate performance trend analysis."""
        df = self.db.get_historical_data(days)
        
        if df.empty:
            logger.warning("No historical data available for trend analysis")
            return {}
        
        trends = {}
        
        # Group by test and analyze trends
        for (module, test), group in df.groupby(['module_name', 'test_name']):
            if len(group) < 3:  # Need at least 3 data points
                continue
            
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Calculate trends for different metrics
            test_trends = {}
            
            for metric in ['mean_time', 'throughput', 'memory_usage_mb']:
                if metric in group.columns:
                    values = group[metric].values
                    timestamps = pd.to_datetime(group['timestamp']).values
                    
                    # Calculate linear trend
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    # Determine trend direction
                    if p_value < 0.05:  # Statistically significant
                        if slope > 0:
                            direction = 'increasing' if metric != 'throughput' else 'decreasing'
                        else:
                            direction = 'decreasing' if metric != 'throughput' else 'increasing'
                    else:
                        direction = 'stable'
                    
                    test_trends[metric] = {
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'direction': direction,
                        'recent_value': values[-1],
                        'change_from_start': ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
                    }
            
            trends[f"{module}.{test}"] = test_trends
        
        return trends
    
    def generate_performance_report(self, output_file: str, include_trends: bool = True):
        """Generate comprehensive performance report."""
        logger.info(f"Generating performance report: {output_file}")
        
        # Get recent data
        df = self.db.get_historical_data(30)
        
        if df.empty:
            logger.warning("No data available for report generation")
            return
        
        # Generate trends if requested
        trends = self.generate_trend_analysis() if include_trends else {}
        
        # Create HTML report
        html_content = self._generate_html_report(df, trends)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report saved to {output_file}")
    
    def _generate_html_report(self, df: pd.DataFrame, trends: Dict) -> str:
        """Generate HTML performance report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIBF Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .regression { background-color: #ffebee; }
                .improvement { background-color: #e8f5e8; }
                .stable { background-color: #f5f5f5; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AIBF Performance Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Data period: Last 30 days</p>
            </div>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Summary statistics
        html += """
            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="metric">
                    <strong>Total Tests:</strong> {total_tests}
                </div>
                <div class="metric">
                    <strong>Modules Covered:</strong> {modules}
                </div>
                <div class="metric">
                    <strong>Data Points:</strong> {data_points}
                </div>
            </div>
        """.format(
            total_tests=len(df.groupby(['module_name', 'test_name'])),
            modules=len(df['module_name'].unique()),
            data_points=len(df)
        )
        
        # Recent performance table
        html += """
            <div class="section">
                <h2>Recent Performance (Last 7 Days)</h2>
                <table>
                    <tr>
                        <th>Module</th>
                        <th>Test</th>
                        <th>Mean Time (s)</th>
                        <th>Throughput (ops/s)</th>
                        <th>Memory (MB)</th>
                        <th>Error Rate</th>
                    </tr>
        """
        
        # Get recent data (last 7 days)
        recent_df = df[df['timestamp'] >= (datetime.now() - timedelta(days=7))]
        recent_summary = recent_df.groupby(['module_name', 'test_name']).agg({
            'mean_time': 'mean',
            'throughput': 'mean',
            'memory_usage_mb': 'mean',
            'error_rate': 'mean'
        }).round(4)
        
        for (module, test), row in recent_summary.iterrows():
            html += f"""
                    <tr>
                        <td>{module}</td>
                        <td>{test}</td>
                        <td>{row['mean_time']:.4f}</td>
                        <td>{row['throughput']:.2f}</td>
                        <td>{row['memory_usage_mb']:.2f}</td>
                        <td>{row['error_rate']:.2%}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # Trends section
        if trends:
            html += """
                <div class="section">
                    <h2>Performance Trends</h2>
            """
            
            for test_name, test_trends in trends.items():
                html += f"<h3>{test_name}</h3>"
                
                for metric, trend_data in test_trends.items():
                    direction = trend_data['direction']
                    css_class = 'regression' if direction == 'increasing' and metric == 'mean_time' else \
                               'improvement' if direction == 'decreasing' and metric == 'mean_time' else 'stable'
                    
                    html += f"""
                        <div class="metric {css_class}">
                            <strong>{metric}:</strong> {direction} 
                            (R² = {trend_data['r_squared']:.3f}, 
                            Change: {trend_data['change_from_start']:.1f}%)
                        </div>
                    """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def create_performance_dashboard(self, output_dir: str = "dashboard"):
        """Create interactive performance dashboard."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get data
        df = self.db.get_historical_data(90)  # Last 3 months
        
        if df.empty:
            logger.warning("No data available for dashboard creation")
            return
        
        # Create various plots
        self._create_time_series_plots(df, output_path)
        self._create_comparison_plots(df, output_path)
        self._create_distribution_plots(df, output_path)
        
        logger.info(f"Performance dashboard created in {output_path}")
    
    def _create_time_series_plots(self, df: pd.DataFrame, output_path: Path):
        """Create time series plots for performance metrics."""
        plt.style.use('seaborn-v0_8')
        
        for metric in ['mean_time', 'throughput', 'memory_usage_mb']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{metric.replace("_", " ").title()} Over Time', fontsize=16)
            
            modules = df['module_name'].unique()[:4]  # Top 4 modules
            
            for i, module in enumerate(modules):
                ax = axes[i//2, i%2]
                module_data = df[df['module_name'] == module]
                
                for test in module_data['test_name'].unique():
                    test_data = module_data[module_data['test_name'] == test]
                    ax.plot(test_data['timestamp'], test_data[metric], 
                           label=test, marker='o', markersize=3)
                
                ax.set_title(f'{module} Module')
                ax.set_xlabel('Date')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / f'{metric}_timeseries.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_comparison_plots(self, df: pd.DataFrame, output_path: Path):
        """Create comparison plots between modules and tests."""
        # Module comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['mean_time', 'throughput', 'memory_usage_mb']
        
        for i, metric in enumerate(metrics):
            module_stats = df.groupby('module_name')[metric].mean().sort_values()
            
            axes[i].bar(range(len(module_stats)), module_stats.values)
            axes[i].set_title(f'Average {metric.replace("_", " ").title()} by Module')
            axes[i].set_xlabel('Module')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_xticks(range(len(module_stats)))
            axes[i].set_xticklabels(module_stats.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'module_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_distribution_plots(self, df: pd.DataFrame, output_path: Path):
        """Create distribution plots for performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Metrics Distributions', fontsize=16)
        
        # Execution time distribution
        axes[0, 0].hist(df['mean_time'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Execution Time Distribution')
        axes[0, 0].set_xlabel('Mean Time (s)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Throughput distribution
        axes[0, 1].hist(df['throughput'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Throughput Distribution')
        axes[0, 1].set_xlabel('Throughput (ops/s)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Memory usage distribution
        axes[1, 0].hist(df['memory_usage_mb'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Memory Usage Distribution')
        axes[1, 0].set_xlabel('Memory Usage (MB)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Error rate distribution
        axes[1, 1].hist(df['error_rate'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Error Rate Distribution')
        axes[1, 1].set_xlabel('Error Rate')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path / 'distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main analysis execution function."""
    parser = argparse.ArgumentParser(description="AIBF Performance Analysis")
    parser.add_argument("--baseline", help="Baseline benchmark results file")
    parser.add_argument("--current", help="Current benchmark results file")
    parser.add_argument("--history-dir", help="Directory containing historical results")
    parser.add_argument("--detect-regressions", action="store_true",
                       help="Detect performance regressions")
    parser.add_argument("--generate-trends", action="store_true",
                       help="Generate trend analysis")
    parser.add_argument("--create-dashboard", action="store_true",
                       help="Create performance dashboard")
    parser.add_argument("--output", default="performance_report.html",
                       help="Output file for reports")
    parser.add_argument("--config", default="benchmark_config.yaml",
                       help="Configuration file")
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(args.config)
    
    try:
        if args.baseline and args.current:
            # Compare two specific result files
            regressions = analyzer.compare_results(args.baseline, args.current)
            
            if regressions:
                print(f"\nDetected {len(regressions)} performance regressions:")
                for regression in regressions:
                    print(f"  {regression.module_name}.{regression.test_name} - "
                          f"{regression.metric}: {regression.change_percent:+.1f}% "
                          f"({regression.severity})")
            else:
                print("No performance regressions detected.")
        
        if args.history_dir:
            # Process historical data
            history_path = Path(args.history_dir)
            for result_file in history_path.glob("*.json"):
                with open(result_file, 'r') as f:
                    data = json.load(f)
                analyzer.db.store_benchmark_run(data)
            
            logger.info(f"Processed {len(list(history_path.glob('*.json')))} result files")
        
        if args.generate_trends:
            trends = analyzer.generate_trend_analysis()
            
            print("\nPerformance Trends (Last 30 days):")
            for test_name, test_trends in trends.items():
                print(f"\n{test_name}:")
                for metric, trend_data in test_trends.items():
                    direction = trend_data['direction']
                    change = trend_data['change_from_start']
                    print(f"  {metric}: {direction} ({change:+.1f}%)")
        
        if args.create_dashboard:
            analyzer.create_performance_dashboard()
            print("Performance dashboard created in 'dashboard/' directory")
        
        # Generate comprehensive report
        analyzer.generate_performance_report(args.output)
        print(f"Performance report generated: {args.output}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()