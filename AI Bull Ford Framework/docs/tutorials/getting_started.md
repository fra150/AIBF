# Getting Started with AI Bull Ford Framework

## Overview

AI Bull Ford Framework (AIBF) is a comprehensive, production-ready AI framework designed for building scalable, modular, and secure AI applications. This tutorial will guide you through the basic setup and your first AI pipeline.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Basic understanding of Python and AI concepts
- 8GB+ RAM recommended

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/aibf.git
cd aibf
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv aibf_env

# Activate on Windows
aibf_env\Scripts\activate

# Activate on Linux/Mac
source aibf_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Run basic tests
pytest tests/unit/test_core.py -v

# Check framework status
python main.py --mode test
```

## Quick Start: Your First AI Pipeline

### 1. Basic Neural Network

Let's create a simple neural network for classification:

```python
from src.core.architectures.neural_networks import NeuralNetwork
from src.assembly_line.pipeline import Pipeline
from src.config.manager import ConfigManager
import numpy as np

# Initialize configuration
config = ConfigManager()
config.load_default()

# Create a simple neural network
nn = NeuralNetwork(
    input_size=784,  # 28x28 MNIST images
    hidden_sizes=[128, 64],
    output_size=10,  # 10 classes
    activation='relu'
)

# Generate sample data
X_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, 1000)

# Train the model
nn.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
X_test = np.random.randn(100, 784)
predictions = nn.predict(X_test)

print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:5]}")
```

### 2. Using Assembly Line Pipeline

AIBF's assembly line system allows you to create complex workflows:

```python
from src.assembly_line.pipeline import Pipeline
from src.assembly_line.workflow import WorkflowDefinition
from src.core.architectures.transformers import TransformerModel
from src.enhancement.rag import RAGSystem

# Define a workflow
workflow = WorkflowDefinition(
    name="text_processing_pipeline",
    description="Process text with transformer and RAG"
)

# Add steps to workflow
workflow.add_step(
    name="preprocess",
    module="text_preprocessing",
    config={"max_length": 512, "tokenizer": "bert-base-uncased"}
)

workflow.add_step(
    name="transform",
    module="transformer",
    config={"model_name": "bert-base-uncased", "num_layers": 6}
)

workflow.add_step(
    name="rag_enhance",
    module="rag",
    config={"vector_store": "faiss", "top_k": 5}
)

# Create and run pipeline
pipeline = Pipeline(workflow)
result = pipeline.run({
    "text": "What is artificial intelligence?",
    "context": "AI research documents"
})

print(f"Pipeline result: {result}")
```

### 3. Multi-Modal Processing

Process both text and images together:

```python
from src.multimodal.vision import VisionProcessor
from src.multimodal.cross_modal import CrossModalProcessor
from src.multimodal.fusion import ModalityFusion
import torch

# Initialize processors
vision_processor = VisionProcessor(model_name="resnet50")
cross_modal = CrossModalProcessor(model_name="clip")
fusion = ModalityFusion(fusion_type="attention")

# Process image and text
image_tensor = torch.randn(1, 3, 224, 224)  # Sample image
text = "A beautiful sunset over the mountains"

# Extract features
image_features = vision_processor.extract_features(image_tensor)
text_features = cross_modal.encode_text(text)

# Fuse modalities
fused_features = fusion.fuse([image_features, text_features])

print(f"Fused features shape: {fused_features.shape}")
```

## Configuration Management

AIBF uses a centralized configuration system:

### 1. Default Configuration

```python
from src.config.manager import ConfigManager

# Load default configuration
config = ConfigManager()
config.load_default()

# Access configuration values
print(f"Model config: {config.get('models')}")
print(f"API config: {config.get('api')}")
```

### 2. Custom Configuration

Create a custom configuration file:

```yaml
# config/custom.yaml
models:
  default_model: "gpt-3.5-turbo"
  max_tokens: 2048
  temperature: 0.7

api:
  host: "localhost"
  port: 8000
  enable_cors: true

database:
  url: "postgresql://user:pass@localhost/aibf"
  pool_size: 10
```

```python
# Load custom configuration
config.load_from_file("config/custom.yaml")
```

### 3. Environment Variables

```python
# Set environment variables
import os
os.environ['AIBF_API_KEY'] = 'your-api-key'
os.environ['AIBF_DEBUG'] = 'true'

# Load environment configuration
config.load_environment()
```

## API Usage

AIBF provides multiple API interfaces:

### 1. REST API

```python
from src.api.rest import RESTServer

# Start REST server
server = RESTServer(host="localhost", port=8000)
server.start()

# Make requests
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "model": "neural_network",
        "data": [[1, 2, 3, 4]],
        "config": {"batch_size": 1}
    }
)

print(response.json())
```

### 2. WebSocket API

```python
from src.api.websocket import WebSocketServer
import asyncio
import websockets

# Start WebSocket server
ws_server = WebSocketServer(host="localhost", port=8001)
ws_server.start()

# Client example
async def client():
    uri = "ws://localhost:8001"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "action": "predict",
            "data": {"text": "Hello, AI!"}
        }))
        response = await websocket.recv()
        print(f"Response: {response}")

# Run client
asyncio.run(client())
```

## Security Features

AIBF includes comprehensive security features:

### 1. Authentication

```python
from src.security.authentication import AuthenticationManager

# Initialize authentication
auth = AuthenticationManager()

# Create user
user = auth.create_user(
    username="john_doe",
    password="secure_password",
    roles=["user", "analyst"]
)

# Authenticate
token = auth.authenticate("john_doe", "secure_password")
print(f"Auth token: {token}")

# Verify token
is_valid = auth.verify_token(token)
print(f"Token valid: {is_valid}")
```

### 2. Data Validation

```python
from src.security.validation import DataValidator

# Initialize validator
validator = DataValidator()

# Validate input data
data = {
    "text": "Hello world",
    "numbers": [1, 2, 3],
    "config": {"temperature": 0.7}
}

schema = {
    "text": {"type": "string", "max_length": 1000},
    "numbers": {"type": "array", "items": {"type": "number"}},
    "config": {"type": "object"}
}

is_valid, errors = validator.validate(data, schema)
print(f"Data valid: {is_valid}")
if errors:
    print(f"Validation errors: {errors}")
```

## Monitoring and Analytics

Monitor your AI applications in real-time:

```python
from src.monitoring.analytics import AnalyticsCollector
from src.monitoring.performance import PerformanceMonitor

# Initialize monitoring
analytics = AnalyticsCollector()
perf_monitor = PerformanceMonitor()

# Track events
analytics.track_event("model_prediction", {
    "model_name": "neural_network",
    "input_size": 784,
    "prediction_time": 0.05
})

# Monitor performance
with perf_monitor.measure("inference_time"):
    # Your AI inference code here
    result = model.predict(data)

# Get metrics
metrics = perf_monitor.get_metrics()
print(f"Performance metrics: {metrics}")
```

## Interactive CLI

AIBF includes an interactive command-line interface:

```bash
# Start interactive mode
python main.py --mode interactive

# Available commands:
# - help: Show available commands
# - status: Show framework status
# - config: Manage configuration
# - models: List and manage models
# - pipeline: Create and run pipelines
# - test: Run tests
# - monitor: View monitoring data
```

## Next Steps

Now that you've completed the getting started tutorial, explore these advanced topics:

1. **[Core Neural Networks Tutorial](core_tutorial.md)** - Deep dive into neural architectures
2. **[Healthcare AI Example](../examples/healthcare_ai.py)** - Real-world healthcare application
3. **[Multimodal Analysis Example](../examples/multimodal_analysis.py)** - Process multiple data types
4. **[API Reference](../api/)** - Complete API documentation
5. **[Best Practices Guide](../guides/best_practices.md)** - Production deployment tips

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory and virtual environment
   cd aibf
   source aibf_env/bin/activate  # or aibf_env\Scripts\activate on Windows
   pip install -e .
   ```

2. **Memory Issues**
   ```python
   # Reduce batch size or model size
   config.set('training.batch_size', 16)  # Instead of 32
   config.set('models.hidden_size', 256)  # Instead of 512
   ```

3. **GPU Issues**
   ```python
   # Check GPU availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   ```

### Getting Help

- **Documentation**: Check the [docs](../index.md) directory
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join our Discord server
- **Support**: Email support@aibf.ai

## Summary

You've successfully:
- ✅ Installed AIBF framework
- ✅ Created your first neural network
- ✅ Built a processing pipeline
- ✅ Used multi-modal capabilities
- ✅ Configured the framework
- ✅ Explored API interfaces
- ✅ Set up security and monitoring

AIBF provides a solid foundation for building production-ready AI applications. Continue with the advanced tutorials to unlock the full potential of the framework!