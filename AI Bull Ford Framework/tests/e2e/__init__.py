"""End-to-end test configuration and utilities."""

import os
import sys
import logging
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager, asynccontextmanager
import tempfile
import shutil
import json
from datetime import datetime, timedelta

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# E2E test configuration
@dataclass
class E2EConfig:
    """Configuration for end-to-end tests."""
    # Test timeouts
    test_timeout: int = 600  # 10 minutes
    api_timeout: int = 30  # 30 seconds
    startup_timeout: int = 60  # 1 minute
    
    # Test environment
    test_data_dir: Path = Path("test_data")
    temp_dir: Path = Path("/tmp/aibf_e2e_tests")
    log_dir: Path = Path("e2e_logs")
    
    # API endpoints
    rest_api_host: str = "localhost"
    rest_api_port: int = 8080
    websocket_port: int = 8081
    grpc_port: int = 8082
    graphql_port: int = 8083
    
    # Test data sizes
    small_dataset_size: int = 100
    medium_dataset_size: int = 1000
    large_dataset_size: int = 10000
    
    # Model configurations
    test_model_path: str = "test_models"
    checkpoint_interval: int = 10  # epochs
    
    # Healthcare test data
    medical_image_size: tuple = (512, 512, 3)
    patient_count: int = 50
    
    # Financial test data
    market_symbols: List[str] = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    portfolio_count: int = 20
    
    # Educational test data
    learner_count: int = 100
    content_count: int = 500
    
    # Performance thresholds
    max_response_time: float = 5.0  # seconds
    max_startup_time: float = 30.0  # seconds
    min_throughput: float = 10.0  # requests/second
    
    # Logging
    log_level: str = "INFO"
    enable_detailed_logging: bool = True
    log_api_requests: bool = True
    log_performance_metrics: bool = True


# Global E2E configuration
E2E_CONFIG = E2EConfig()

# Ensure test directories exist
E2E_CONFIG.temp_dir.mkdir(parents=True, exist_ok=True)
E2E_CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
E2E_CONFIG.test_data_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, E2E_CONFIG.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(E2E_CONFIG.log_dir / "e2e_tests.log")
    ]
)

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate test data for E2E tests."""
    
    def __init__(self, config: E2EConfig):
        self.config = config
        self.data_dir = config.test_data_dir
    
    def generate_medical_images(self, count: int = None) -> List[Dict[str, Any]]:
        """Generate synthetic medical image data."""
        import numpy as np
        
        count = count or self.config.patient_count
        images = []
        
        for i in range(count):
            # Generate synthetic medical image
            image_data = np.random.rand(*self.config.medical_image_size)
            
            # Add some structure to make it more realistic
            # Add circular patterns (simulating organs)
            center_x, center_y = self.config.medical_image_size[0] // 2, self.config.medical_image_size[1] // 2
            y, x = np.ogrid[:self.config.medical_image_size[0], :self.config.medical_image_size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 < (self.config.medical_image_size[0] // 4)**2
            image_data[mask] += 0.3
            
            # Normalize
            image_data = np.clip(image_data, 0, 1)
            
            image_info = {
                "image_id": f"IMG_{i:04d}",
                "patient_id": f"P{i:04d}",
                "image_type": np.random.choice(["xray", "ct", "mri"]),
                "image_data": image_data.tolist(),
                "acquisition_date": (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat(),
                "metadata": {
                    "resolution": self.config.medical_image_size[:2],
                    "modality": "synthetic",
                    "body_part": np.random.choice(["chest", "head", "abdomen"])
                }
            }
            
            images.append(image_info)
        
        return images
    
    def generate_patient_data(self, count: int = None) -> List[Dict[str, Any]]:
        """Generate synthetic patient data."""
        import numpy as np
        
        count = count or self.config.patient_count
        patients = []
        
        for i in range(count):
            patient = {
                "patient_id": f"P{i:04d}",
                "age": np.random.randint(18, 90),
                "gender": np.random.choice(["M", "F"]),
                "medical_history": np.random.choice([
                    ["diabetes"], ["hypertension"], ["asthma"], 
                    ["diabetes", "hypertension"], []
                ]),
                "current_medications": np.random.choice([
                    ["metformin"], ["lisinopril"], ["albuterol"], 
                    ["metformin", "lisinopril"], []
                ]),
                "vital_signs": {
                    "heart_rate": np.random.randint(60, 100),
                    "blood_pressure_systolic": np.random.randint(110, 140),
                    "blood_pressure_diastolic": np.random.randint(70, 90),
                    "temperature": np.random.uniform(97.0, 99.5),
                    "oxygen_saturation": np.random.randint(95, 100)
                }
            }
            
            patients.append(patient)
        
        return patients
    
    def generate_financial_data(self) -> Dict[str, Any]:
        """Generate synthetic financial data."""
        import numpy as np
        
        # Generate market data
        market_data = {}
        for symbol in self.config.market_symbols:
            base_price = np.random.uniform(50, 500)
            prices = []
            
            # Generate price series
            current_price = base_price
            for _ in range(252):  # One year of trading days
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                current_price *= (1 + change)
                prices.append(current_price)
            
            market_data[symbol] = {
                "symbol": symbol,
                "prices": prices,
                "current_price": prices[-1],
                "volume": np.random.randint(1000000, 10000000),
                "market_cap": prices[-1] * np.random.randint(1000000, 10000000000),
                "sector": np.random.choice(["Technology", "Healthcare", "Finance", "Energy"])
            }
        
        # Generate portfolios
        portfolios = []
        for i in range(self.config.portfolio_count):
            # Random portfolio allocation
            positions = {}
            total_value = np.random.uniform(10000, 1000000)
            
            # Allocate to random symbols
            selected_symbols = np.random.choice(
                self.config.market_symbols, 
                size=np.random.randint(2, len(self.config.market_symbols)), 
                replace=False
            )
            
            weights = np.random.dirichlet(np.ones(len(selected_symbols)))
            
            for symbol, weight in zip(selected_symbols, weights):
                value = total_value * weight
                shares = int(value / market_data[symbol]["current_price"])
                positions[symbol] = {
                    "shares": shares,
                    "avg_cost": market_data[symbol]["current_price"] * np.random.uniform(0.8, 1.2),
                    "current_value": shares * market_data[symbol]["current_price"]
                }
            
            portfolio = {
                "portfolio_id": f"PORT{i:03d}",
                "user_id": f"USER{i:03d}",
                "total_value": sum(pos["current_value"] for pos in positions.values()),
                "cash_balance": np.random.uniform(1000, 50000),
                "positions": positions,
                "risk_profile": np.random.choice(["conservative", "moderate", "aggressive"]),
                "last_updated": datetime.now().isoformat()
            }
            
            portfolios.append(portfolio)
        
        return {
            "market_data": market_data,
            "portfolios": portfolios
        }
    
    def generate_educational_data(self) -> Dict[str, Any]:
        """Generate synthetic educational data."""
        import numpy as np
        
        # Generate learner profiles
        learners = []
        for i in range(self.config.learner_count):
            learner = {
                "learner_id": f"L{i:04d}",
                "age": np.random.randint(8, 18),
                "grade_level": np.random.randint(3, 12),
                "learning_style": np.random.choice(["visual", "auditory", "kinesthetic", "reading"]),
                "strengths": np.random.choice([
                    ["mathematics"], ["science"], ["language_arts"], ["history"],
                    ["mathematics", "science"], ["language_arts", "history"]
                ]),
                "weaknesses": np.random.choice([
                    ["writing"], ["reading_comprehension"], ["problem_solving"],
                    ["critical_thinking"], ["writing", "problem_solving"]
                ]),
                "interests": np.random.choice([
                    ["science"], ["technology"], ["arts"], ["sports"],
                    ["science", "technology"], ["arts", "sports"]
                ]),
                "performance_history": {
                    "mathematics": np.random.uniform(0.6, 1.0),
                    "science": np.random.uniform(0.6, 1.0),
                    "language_arts": np.random.uniform(0.6, 1.0),
                    "history": np.random.uniform(0.6, 1.0)
                }
            }
            
            learners.append(learner)
        
        # Generate learning content
        subjects = ["mathematics", "science", "language_arts", "history"]
        content_types = ["video", "text", "interactive", "quiz"]
        difficulty_levels = ["beginner", "intermediate", "advanced"]
        
        content = []
        for i in range(self.config.content_count):
            content_item = {
                "content_id": f"C{i:04d}",
                "title": f"Lesson {i+1}: {np.random.choice(['Introduction to', 'Advanced', 'Fundamentals of'])} {np.random.choice(subjects).title()}",
                "subject": np.random.choice(subjects),
                "content_type": np.random.choice(content_types),
                "difficulty_level": np.random.choice(difficulty_levels),
                "estimated_duration": np.random.randint(10, 60),  # minutes
                "prerequisites": np.random.choice([
                    [], [f"C{max(0, i-10):04d}"], [f"C{max(0, i-5):04d}", f"C{max(0, i-3):04d}"]
                ]),
                "learning_objectives": [
                    f"Understand {np.random.choice(['basic', 'advanced', 'key'])} concepts",
                    f"Apply {np.random.choice(['theoretical', 'practical', 'analytical'])} knowledge",
                    f"Demonstrate {np.random.choice(['problem-solving', 'critical thinking', 'creative'])} skills"
                ],
                "tags": np.random.choice([
                    ["fundamental"], ["practical"], ["theoretical"], ["hands-on"],
                    ["fundamental", "practical"], ["theoretical", "hands-on"]
                ]),
                "rating": np.random.uniform(3.5, 5.0),
                "completion_rate": np.random.uniform(0.7, 0.95)
            }
            
            content.append(content_item)
        
        return {
            "learners": learners,
            "content": content
        }
    
    def save_test_data(self):
        """Save all generated test data to files."""
        logger.info("Generating test data...")
        
        # Generate and save medical data
        medical_images = self.generate_medical_images()
        patients = self.generate_patient_data()
        
        with open(self.data_dir / "medical_images.json", "w") as f:
            json.dump(medical_images, f, indent=2)
        
        with open(self.data_dir / "patients.json", "w") as f:
            json.dump(patients, f, indent=2)
        
        # Generate and save financial data
        financial_data = self.generate_financial_data()
        
        with open(self.data_dir / "financial_data.json", "w") as f:
            json.dump(financial_data, f, indent=2)
        
        # Generate and save educational data
        educational_data = self.generate_educational_data()
        
        with open(self.data_dir / "educational_data.json", "w") as f:
            json.dump(educational_data, f, indent=2)
        
        logger.info(f"Test data saved to {self.data_dir}")
        
        return {
            "medical_images": len(medical_images),
            "patients": len(patients),
            "portfolios": len(financial_data["portfolios"]),
            "market_symbols": len(financial_data["market_data"]),
            "learners": len(educational_data["learners"]),
            "content": len(educational_data["content"])
        }


class APITestClient:
    """Test client for API endpoints."""
    
    def __init__(self, config: E2EConfig):
        self.config = config
        self.base_url = f"http://{config.rest_api_host}:{config.rest_api_port}"
        self.websocket_url = f"ws://{config.rest_api_host}:{config.websocket_port}"
    
    async def test_rest_endpoint(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Test REST API endpoint."""
        import aiohttp
        
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.api_timeout)) as session:
            start_time = time.time()
            
            if method.upper() == "GET":
                async with session.get(url) as response:
                    response_data = await response.json()
                    status_code = response.status
            elif method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    response_data = await response.json()
                    status_code = response.status
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "status_code": status_code,
                "response_data": response_data,
                "response_time": response_time,
                "success": 200 <= status_code < 300
            }
    
    async def test_websocket_connection(self, endpoint: str, messages: List[Dict]) -> List[Dict[str, Any]]:
        """Test WebSocket connection."""
        import websockets
        import json
        
        url = f"{self.websocket_url}{endpoint}"
        responses = []
        
        try:
            async with websockets.connect(url) as websocket:
                # Send messages and collect responses
                for message in messages:
                    start_time = time.time()
                    
                    await websocket.send(json.dumps(message))
                    response = await asyncio.wait_for(
                        websocket.recv(), 
                        timeout=self.config.api_timeout
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    responses.append({
                        "message_sent": message,
                        "response_received": json.loads(response),
                        "response_time": response_time,
                        "success": True
                    })
        
        except Exception as e:
            responses.append({
                "error": str(e),
                "success": False
            })
        
        return responses


class E2ETestRunner:
    """End-to-end test runner."""
    
    def __init__(self, config: E2EConfig):
        self.config = config
        self.data_generator = TestDataGenerator(config)
        self.api_client = APITestClient(config)
        self.test_results = {}
    
    @contextmanager
    def test_environment(self, test_name: str):
        """Context manager for E2E test environment."""
        logger.info(f"Setting up E2E test environment for: {test_name}")
        
        # Create temporary directory for this test
        test_temp_dir = self.config.temp_dir / test_name
        test_temp_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        try:
            yield test_temp_dir
        finally:
            end_time = time.time()
            test_duration = end_time - start_time
            
            logger.info(f"E2E test {test_name} completed in {test_duration:.2f}s")
            
            # Clean up temporary files
            if test_temp_dir.exists():
                shutil.rmtree(test_temp_dir)
    
    def check_performance_requirements(self, metrics: Dict[str, Any]) -> List[str]:
        """Check if performance metrics meet requirements."""
        violations = []
        
        # Check response time
        if "response_time" in metrics:
            if metrics["response_time"] > self.config.max_response_time:
                violations.append(
                    f"Response time {metrics['response_time']:.2f}s exceeds threshold {self.config.max_response_time}s"
                )
        
        # Check startup time
        if "startup_time" in metrics:
            if metrics["startup_time"] > self.config.max_startup_time:
                violations.append(
                    f"Startup time {metrics['startup_time']:.2f}s exceeds threshold {self.config.max_startup_time}s"
                )
        
        # Check throughput
        if "throughput" in metrics:
            if metrics["throughput"] < self.config.min_throughput:
                violations.append(
                    f"Throughput {metrics['throughput']:.2f} req/s below threshold {self.config.min_throughput} req/s"
                )
        
        return violations
    
    def save_test_results(self, test_name: str, results: Dict[str, Any]):
        """Save test results to file."""
        timestamp = datetime.now().isoformat()
        results_file = self.config.log_dir / f"{test_name}_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump({
                "test_name": test_name,
                "timestamp": timestamp,
                "results": results,
                "config": {
                    "max_response_time": self.config.max_response_time,
                    "max_startup_time": self.config.max_startup_time,
                    "min_throughput": self.config.min_throughput
                }
            }, f, indent=2)
        
        logger.info(f"Test results saved to {results_file}")


# Global instances
data_generator = TestDataGenerator(E2E_CONFIG)
api_client = APITestClient(E2E_CONFIG)
test_runner = E2ETestRunner(E2E_CONFIG)

logger.info("E2E test environment initialized")
logger.info(f"Configuration: {E2E_CONFIG}")