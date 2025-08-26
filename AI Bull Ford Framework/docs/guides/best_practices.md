# Best Practices AIBF

## Panoramica

Questa guida raccoglie le migliori pratiche per lo sviluppo, deployment e manutenzione di applicazioni basate sul framework AIBF.

## Sviluppo

### 1. Struttura del Codice

#### Organizzazione dei File
```
project/
├── src/
│   ├── domain/          # Business logic
│   ├── infrastructure/  # External dependencies
│   ├── application/     # Use cases
│   └── interfaces/      # API controllers
├── tests/
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── e2e/           # End-to-end tests
├── docs/
└── scripts/
```

#### Naming Conventions
```python
# Classes: PascalCase
class ModelTrainer:
    pass

# Functions/Variables: snake_case
def train_model():
    model_config = {}

# Constants: UPPER_SNAKE_CASE
MAX_TRAINING_EPOCHS = 100

# Private methods: _underscore_prefix
def _validate_input(self, data):
    pass
```

#### Type Hints
```python
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    learning_rate: float
    epochs: int
    layers: List[int]
    metadata: Optional[Dict[str, str]] = None

def train_model(
    config: ModelConfig,
    data: List[Dict[str, Union[str, float]]]
) -> Optional[str]:
    """Train a model with given configuration.
    
    Args:
        config: Model configuration
        data: Training data
        
    Returns:
        Model ID if successful, None otherwise
    """
    pass
```

### 2. Error Handling

#### Exception Hierarchy
```python
class AIBFError(Exception):
    """Base exception for AIBF framework."""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.utcnow()

class ValidationError(AIBFError):
    """Data validation errors."""
    pass

class ModelError(AIBFError):
    """Model-related errors."""
    pass

class TrainingError(ModelError):
    """Training-specific errors."""
    pass
```

#### Error Handling Patterns
```python
# Result pattern
from typing import Union, Generic, TypeVar

T = TypeVar('T')
E = TypeVar('E')

class Result(Generic[T, E]):
    def __init__(self, value: T = None, error: E = None):
        self._value = value
        self._error = error
    
    @property
    def is_success(self) -> bool:
        return self._error is None
    
    @property
    def value(self) -> T:
        if self._error:
            raise self._error
        return self._value
    
    @property
    def error(self) -> E:
        return self._error

# Usage
def train_model(config: ModelConfig) -> Result[str, TrainingError]:
    try:
        model_id = _perform_training(config)
        return Result(value=model_id)
    except Exception as e:
        return Result(error=TrainingError(str(e)))

# Client code
result = train_model(config)
if result.is_success:
    print(f"Model trained: {result.value}")
else:
    logger.error(f"Training failed: {result.error}")
```

### 3. Logging

#### Structured Logging
```python
import structlog
from structlog.stdlib import LoggerFactory

# Configuration
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Usage
logger = structlog.get_logger()

def train_model(config: ModelConfig):
    logger.info(
        "Starting model training",
        model_name=config.name,
        learning_rate=config.learning_rate,
        epochs=config.epochs
    )
    
    try:
        # Training logic
        pass
    except Exception as e:
        logger.error(
            "Model training failed",
            model_name=config.name,
            error=str(e),
            exc_info=True
        )
        raise
    
    logger.info(
        "Model training completed",
        model_name=config.name,
        duration_seconds=training_duration
    )
```

#### Log Levels
```python
# DEBUG: Detailed information for debugging
logger.debug("Processing batch", batch_size=32, batch_id=123)

# INFO: General information about program execution
logger.info("Model training started", model_id="model_123")

# WARNING: Something unexpected happened, but the program continues
logger.warning("Low memory warning", available_memory="512MB")

# ERROR: Serious problem that prevented a function from executing
logger.error("Failed to load model", model_path="/path/to/model")

# CRITICAL: Very serious error that may cause the program to abort
logger.critical("Database connection lost", database_url="postgresql://...")
```

### 4. Testing

#### Test Structure
```python
# test_model_service.py
import pytest
from unittest.mock import Mock, patch
from src.services.model_service import ModelService
from src.domain.model import ModelConfig

class TestModelService:
    """Test suite for ModelService."""
    
    @pytest.fixture
    def model_service(self):
        """Create ModelService instance for testing."""
        return ModelService()
    
    @pytest.fixture
    def sample_config(self):
        """Create sample model configuration."""
        return ModelConfig(
            name="test_model",
            learning_rate=0.001,
            epochs=10,
            layers=[128, 64, 32]
        )
    
    def test_create_model_success(self, model_service, sample_config):
        """Test successful model creation."""
        # Arrange
        expected_model_id = "model_123"
        
        # Act
        result = model_service.create_model(sample_config)
        
        # Assert
        assert result is not None
        assert isinstance(result, str)
    
    def test_create_model_invalid_config(self, model_service):
        """Test model creation with invalid configuration."""
        # Arrange
        invalid_config = ModelConfig(
            name="",  # Invalid empty name
            learning_rate=-0.1,  # Invalid negative learning rate
            epochs=0,  # Invalid zero epochs
            layers=[]
        )
        
        # Act & Assert
        with pytest.raises(ValidationError):
            model_service.create_model(invalid_config)
    
    @patch('src.services.model_service.external_api')
    def test_create_model_external_api_failure(self, mock_api, model_service, sample_config):
        """Test model creation when external API fails."""
        # Arrange
        mock_api.create_model.side_effect = Exception("API Error")
        
        # Act & Assert
        with pytest.raises(ModelError):
            model_service.create_model(sample_config)
```

#### Test Categories
```python
# Unit Tests
@pytest.mark.unit
def test_model_validation():
    """Test model validation logic."""
    pass

# Integration Tests
@pytest.mark.integration
def test_database_integration():
    """Test database integration."""
    pass

# Slow Tests
@pytest.mark.slow
def test_full_training_pipeline():
    """Test complete training pipeline."""
    pass

# Run specific test categories
# pytest -m unit
# pytest -m "not slow"
# pytest -m "integration and not slow"
```

#### Mocking Best Practices
```python
# Mock external dependencies
@patch('requests.post')
def test_api_call(mock_post):
    mock_post.return_value.json.return_value = {'status': 'success'}
    mock_post.return_value.status_code = 200
    
    result = api_client.send_request(data)
    
    assert result['status'] == 'success'
    mock_post.assert_called_once_with(
        'https://api.example.com/endpoint',
        json=data,
        headers={'Content-Type': 'application/json'}
    )

# Use dependency injection for easier testing
class ModelService:
    def __init__(self, api_client=None, database=None):
        self.api_client = api_client or DefaultAPIClient()
        self.database = database or DefaultDatabase()

# In tests
def test_model_service():
    mock_api = Mock()
    mock_db = Mock()
    service = ModelService(api_client=mock_api, database=mock_db)
    # Test with mocked dependencies
```

## Performance

### 1. Async Programming

#### Async Best Practices
```python
import asyncio
from typing import List

# Use async/await for I/O operations
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Batch operations
async def fetch_multiple_data(urls: List[str]) -> List[dict]:
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)

# Use semaphores to limit concurrency
async def limited_fetch(urls: List[str], max_concurrent: int = 10) -> List[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_semaphore(url: str) -> dict:
        async with semaphore:
            return await fetch_data(url)
    
    tasks = [fetch_with_semaphore(url) for url in urls]
    return await asyncio.gather(*tasks)

# Handle exceptions in async operations
async def safe_fetch(url: str) -> Optional[dict]:
    try:
        return await fetch_data(url)
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None
```

### 2. Caching

#### Multi-level Caching
```python
from functools import wraps
from typing import Any, Callable

class CacheManager:
    def __init__(self):
        self.memory_cache = {}
        self.redis_cache = RedisClient()
    
    async def get(self, key: str) -> Any:
        # L1: Memory cache
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # L2: Redis cache
        value = await self.redis_cache.get(key)
        if value:
            self.memory_cache[key] = value
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        self.memory_cache[key] = value
        await self.redis_cache.set(key, value, ttl)

# Cache decorator
def cached(ttl: int = 3600):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

# Usage
@cached(ttl=1800)
async def expensive_computation(data: List[float]) -> float:
    # Expensive computation here
    return sum(data) / len(data)
```

### 3. Database Optimization

#### Query Optimization
```python
# Use indexes
class Model(Base):
    __tablename__ = 'models'
    
    id = Column(String, primary_key=True)
    name = Column(String, index=True)  # Index for frequent queries
    created_at = Column(DateTime, index=True)
    status = Column(String, index=True)
    
    # Composite index for common query patterns
    __table_args__ = (
        Index('idx_status_created', 'status', 'created_at'),
    )

# Use query optimization
class ModelRepository:
    async def get_active_models(self, limit: int = 100) -> List[Model]:
        # Use specific columns instead of SELECT *
        query = select(Model.id, Model.name, Model.status).where(
            Model.status == 'active'
        ).order_by(
            Model.created_at.desc()
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.fetchall()
    
    async def get_models_with_pagination(
        self, 
        page: int, 
        page_size: int
    ) -> Tuple[List[Model], int]:
        # Count query
        count_query = select(func.count(Model.id))
        total = await self.session.scalar(count_query)
        
        # Data query with offset/limit
        data_query = select(Model).offset(
            (page - 1) * page_size
        ).limit(page_size)
        
        result = await self.session.execute(data_query)
        models = result.scalars().all()
        
        return models, total
```

#### Connection Pooling
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configure connection pool
engine = create_async_engine(
    database_url,
    pool_size=20,          # Number of connections to maintain
    max_overflow=30,       # Additional connections when pool is full
    pool_pre_ping=True,    # Validate connections before use
    pool_recycle=3600,     # Recycle connections after 1 hour
    echo=False             # Set to True for SQL debugging
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Context manager for sessions
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

## Security

### 1. Input Validation

#### Pydantic Validation
```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import re

class ModelConfigRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    learning_rate: float = Field(..., gt=0, le=1)
    epochs: int = Field(..., ge=1, le=1000)
    layers: List[int] = Field(..., min_items=1, max_items=10)
    description: Optional[str] = Field(None, max_length=500)
    
    @validator('name')
    def name_must_be_alphanumeric(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Name must contain only alphanumeric characters, hyphens, and underscores')
        return v
    
    @validator('layers')
    def layers_must_be_positive(cls, v):
        if any(layer <= 0 for layer in v):
            raise ValueError('All layer sizes must be positive')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "my_model",
                "learning_rate": 0.001,
                "epochs": 100,
                "layers": [128, 64, 32],
                "description": "A sample neural network model"
            }
        }
```

#### SQL Injection Prevention
```python
# NEVER do this (vulnerable to SQL injection)
def get_user_by_name_bad(name: str):
    query = f"SELECT * FROM users WHERE name = '{name}'"
    return database.execute(query)

# DO this (parameterized queries)
def get_user_by_name_good(name: str):
    query = "SELECT * FROM users WHERE name = :name"
    return database.execute(query, {"name": name})

# With SQLAlchemy
def get_user_by_name_sqlalchemy(name: str):
    return session.query(User).filter(User.name == name).first()
```

### 2. Authentication & Authorization

#### JWT Implementation
```python
import jwt
from datetime import datetime, timedelta
from typing import Optional

class JWTManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(
        self, 
        user_id: str, 
        permissions: List[str],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

# Authorization decorator
def require_permission(permission: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            
            if not token:
                raise HTTPException(status_code=401, detail="Token required")
            
            try:
                payload = jwt_manager.verify_token(token)
                if permission not in payload.get("permissions", []):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                request.state.user_id = payload["user_id"]
                return await func(request, *args, **kwargs)
            except AuthenticationError as e:
                raise HTTPException(status_code=401, detail=str(e))
        
        return wrapper
    return decorator

# Usage
@require_permission("model:create")
async def create_model(request: Request, config: ModelConfigRequest):
    user_id = request.state.user_id
    # Create model logic
```

### 3. Data Protection

#### Encryption at Rest
```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionService:
    def __init__(self, password: str):
        self.key = self._derive_key(password)
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: str) -> str:
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher.decrypt(encrypted_bytes)
        return decrypted_data.decode()

# Usage in models
class SensitiveData(Base):
    __tablename__ = 'sensitive_data'
    
    id = Column(String, primary_key=True)
    encrypted_content = Column(Text)
    
    def set_content(self, content: str):
        self.encrypted_content = encryption_service.encrypt(content)
    
    def get_content(self) -> str:
        return encryption_service.decrypt(self.encrypted_content)
```

## Deployment

### 1. Docker Best Practices

#### Multi-stage Dockerfile
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r aibf && useradd -r -g aibf aibf

WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/aibf/.local

# Copy application code
COPY --chown=aibf:aibf . .

# Set environment variables
ENV PATH=/home/aibf/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER aibf

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "main.py"]
```

#### Docker Compose for Development
```yaml
version: '3.8'

services:
  aibf-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://aibf:password@db:5432/aibf
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=DEBUG
    volumes:
      - .:/app
      - /app/__pycache__
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped
  
  aibf-worker:
    build:
      context: .
      dockerfile: Dockerfile.dev
    command: celery worker -A aibf.celery --loglevel=info
    environment:
      - DATABASE_URL=postgresql://aibf:password@db:5432/aibf
      - REDIS_URL=redis://redis:6379
    volumes:
      - .:/app
    depends_on:
      - db
      - redis
    restart: unless-stopped
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: aibf
      POSTGRES_USER: aibf
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aibf"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - aibf-api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    name: aibf-network
```

### 2. Kubernetes Deployment

#### Deployment Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aibf-api
  labels:
    app: aibf-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aibf-api
  template:
    metadata:
      labels:
        app: aibf-api
    spec:
      containers:
      - name: aibf-api
        image: aibf:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: aibf-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: config-volume
        configMap:
          name: aibf-config
      imagePullSecrets:
      - name: registry-secret
---
apiVersion: v1
kind: Service
metadata:
  name: aibf-api-service
spec:
  selector:
    app: aibf-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aibf-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aibf-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. CI/CD Pipeline

#### GitHub Actions
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_aibf
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type check with mypy
      run: mypy src
    
    - name: Test with pytest
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_aibf
        REDIS_URL: redis://localhost:6379
      run: |
        pytest --cov=src --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
  
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: 'security-scan-results.sarif'
    
    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r src -f json -o bandit-report.json
  
  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/service.yaml
        images: |
          ghcr.io/${{ github.repository }}:${{ github.sha }}
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
```

## Monitoring e Observability

### 1. Metrics

#### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
REQUEST_COUNT = Counter(
    'aibf_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'aibf_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'aibf_active_connections',
    'Number of active connections'
)

MODEL_TRAINING_DURATION = Histogram(
    'aibf_model_training_duration_seconds',
    'Model training duration in seconds',
    ['model_type']
)

# Metrics decorator
def track_metrics(endpoint: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            method = request.method
            start_time = time.time()
            
            try:
                response = await func(request, *args, **kwargs)
                status_code = response.status_code
                return response
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code
                ).inc()
                REQUEST_DURATION.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator

# Usage
@track_metrics('/models/train')
async def train_model_endpoint(request: Request):
    # Training logic
    pass

# Start metrics server
start_http_server(8001)
```

### 2. Health Checks

#### Comprehensive Health Checks
```python
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float

class HealthChecker:
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'redis': self._check_redis,
            'external_apis': self._check_external_apis,
            'disk_space': self._check_disk_space,
            'memory': self._check_memory
        }
    
    async def check_health(self) -> Dict[str, HealthCheck]:
        results = {}
        
        for name, check_func in self.checks.items():
            start_time = time.time()
            try:
                status, message = await check_func()
                duration_ms = (time.time() - start_time) * 1000
                
                results[name] = HealthCheck(
                    name=name,
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    duration_ms=duration_ms
                )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.utcnow(),
                    duration_ms=duration_ms
                )
        
        return results
    
    async def _check_database(self) -> tuple[HealthStatus, str]:
        try:
            # Simple query to check database connectivity
            result = await database.execute("SELECT 1")
            if result:
                return HealthStatus.HEALTHY, "Database connection successful"
            else:
                return HealthStatus.UNHEALTHY, "Database query failed"
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Database error: {str(e)}"
    
    async def _check_redis(self) -> tuple[HealthStatus, str]:
        try:
            await redis_client.ping()
            return HealthStatus.HEALTHY, "Redis connection successful"
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Redis error: {str(e)}"
    
    async def _check_external_apis(self) -> tuple[HealthStatus, str]:
        # Check critical external APIs
        failed_apis = []
        
        for api_name, api_url in external_apis.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{api_url}/health", timeout=5) as response:
                        if response.status != 200:
                            failed_apis.append(api_name)
            except Exception:
                failed_apis.append(api_name)
        
        if not failed_apis:
            return HealthStatus.HEALTHY, "All external APIs are healthy"
        elif len(failed_apis) < len(external_apis) / 2:
            return HealthStatus.DEGRADED, f"Some APIs are down: {failed_apis}"
        else:
            return HealthStatus.UNHEALTHY, f"Most APIs are down: {failed_apis}"
    
    async def _check_disk_space(self) -> tuple[HealthStatus, str]:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_percent = (free / total) * 100
        
        if free_percent > 20:
            return HealthStatus.HEALTHY, f"Disk space: {free_percent:.1f}% free"
        elif free_percent > 10:
            return HealthStatus.DEGRADED, f"Low disk space: {free_percent:.1f}% free"
        else:
            return HealthStatus.UNHEALTHY, f"Critical disk space: {free_percent:.1f}% free"
    
    async def _check_memory(self) -> tuple[HealthStatus, str]:
        import psutil
        
        memory = psutil.virtual_memory()
        available_percent = memory.available / memory.total * 100
        
        if available_percent > 20:
            return HealthStatus.HEALTHY, f"Memory: {available_percent:.1f}% available"
        elif available_percent > 10:
            return HealthStatus.DEGRADED, f"Low memory: {available_percent:.1f}% available"
        else:
            return HealthStatus.UNHEALTHY, f"Critical memory: {available_percent:.1f}% available"

# Health endpoint
@app.get("/health")
async def health_endpoint():
    health_checker = HealthChecker()
    results = await health_checker.check_health()
    
    overall_status = HealthStatus.HEALTHY
    if any(check.status == HealthStatus.UNHEALTHY for check in results.values()):
        overall_status = HealthStatus.UNHEALTHY
    elif any(check.status == HealthStatus.DEGRADED for check in results.values()):
        overall_status = HealthStatus.DEGRADED
    
    return {
        "status": overall_status.value,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {name: {
            "status": check.status.value,
            "message": check.message,
            "duration_ms": check.duration_ms
        } for name, check in results.items()}
    }
```

## Conclusioni

Queste best practices forniscono una base solida per lo sviluppo di applicazioni robuste, sicure e scalabili con il framework AIBF. Ricorda di:

1. **Adattare alle esigenze specifiche**: Non tutte le pratiche sono necessarie per ogni progetto
2. **Iterare e migliorare**: Le best practices evolvono con l'esperienza e i requisiti
3. **Documentare le decisioni**: Mantieni documentazione delle scelte architetturali
4. **Monitorare e ottimizzare**: Usa metriche per guidare le ottimizzazioni
5. **Rimanere aggiornati**: Segui le evoluzioni delle tecnologie e delle pratiche

## Prossimi Passi

1. Consulta la [API Reference](../api/) per dettagli specifici
2. Prova i [Tutorial](../tutorials/) per esempi pratici
3. Esplora gli [Esempi](../examples/) per casi d'uso reali
4. Partecipa alle [Discussioni](https://github.com/your-repo/aibf/discussions) della community