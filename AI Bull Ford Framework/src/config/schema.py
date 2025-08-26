"""Configuration Schema for AI Bull Ford Framework.

Defines the structure and types for configuration data.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class PrecisionType(Enum):
    """Supported precision types."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class JWTAlgorithm(Enum):
    """JWT algorithms."""
    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"
    RS256 = "RS256"
    RS384 = "RS384"
    RS512 = "RS512"


@dataclass
class FrameworkConfig:
    """Framework general configuration."""
    name: str = "AI Bull Ford Framework"
    version: str = "1.0.0"
    description: str = "Advanced AI Framework for Bull Ford Applications"
    debug: bool = False
    environment: str = "development"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = LogLevel.INFO.value
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    structured_logging: bool = False


@dataclass
class APIServerConfig:
    """API server configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    timeout: int = 30
    max_connections: int = 1000
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class RESTAPIConfig(APIServerConfig):
    """REST API configuration."""
    port: int = 8000
    docs_enabled: bool = True
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"


@dataclass
class WebSocketConfig(APIServerConfig):
    """WebSocket configuration."""
    port: int = 8001
    ping_interval: int = 20
    ping_timeout: int = 10
    max_message_size: int = 1024 * 1024  # 1MB


@dataclass
class GRPCConfig(APIServerConfig):
    """gRPC configuration."""
    port: int = 8002
    max_message_length: int = 4 * 1024 * 1024  # 4MB
    compression: str = "gzip"


@dataclass
class GraphQLConfig(APIServerConfig):
    """GraphQL configuration."""
    port: int = 8003
    playground_enabled: bool = True
    introspection_enabled: bool = True
    max_query_depth: int = 10
    max_query_complexity: int = 1000


@dataclass
class APIConfig:
    """API configuration."""
    rest: RESTAPIConfig = field(default_factory=RESTAPIConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    grpc: GRPCConfig = field(default_factory=GRPCConfig)
    graphql: GraphQLConfig = field(default_factory=GraphQLConfig)


@dataclass
class JWTConfig:
    """JWT configuration."""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = JWTAlgorithm.HS256.value
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "aibf"
    audience: str = "aibf-users"


@dataclass
class AuthenticationConfig:
    """Authentication configuration."""
    enabled: bool = True
    jwt: JWTConfig = field(default_factory=JWTConfig)
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes


@dataclass
class AuthorizationConfig:
    """Authorization configuration."""
    enabled: bool = True
    default_role: str = "user"
    admin_role: str = "admin"
    rbac_enabled: bool = True
    permission_cache_ttl: int = 300  # 5 minutes


@dataclass
class AuthConfig:
    """Authentication and authorization configuration."""
    authentication: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    authorization: AuthorizationConfig = field(default_factory=AuthorizationConfig)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///aibf.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    migration_auto: bool = True


@dataclass
class RedisConfig:
    """Redis configuration."""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    collection_interval: int = 60  # seconds
    retention_days: int = 30
    export_prometheus: bool = False
    prometheus_port: int = 9090


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    enabled: bool = True
    endpoint: str = "/health"
    check_interval: int = 30  # seconds
    timeout: int = 10  # seconds


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)


@dataclass
class CoreConfig:
    """Core module configuration."""
    device: str = DeviceType.AUTO.value
    precision: str = PrecisionType.FLOAT32.value
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True
    gradient_clipping: float = 1.0
    mixed_precision: bool = False


@dataclass
class NeuralNetworkConfig:
    """Neural network configuration."""
    default_activation: str = "relu"
    default_initializer: str = "xavier_uniform"
    dropout_rate: float = 0.1
    batch_norm: bool = True
    layer_norm: bool = False


@dataclass
class TransformerConfig:
    """Transformer configuration."""
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 512
    positional_encoding: str = "sinusoidal"


@dataclass
class ReinforcementLearningConfig:
    """Reinforcement learning configuration."""
    algorithm: str = "ppo"
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    exploration_rate: float = 0.1
    target_update_freq: int = 1000


@dataclass
class VisionConfig:
    """Vision module configuration."""
    default_image_size: List[int] = field(default_factory=lambda: [224, 224])
    color_channels: int = 3
    normalization_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalization_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    augmentation_enabled: bool = True


@dataclass
class AudioConfig:
    """Audio module configuration."""
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    n_mels: int = 80
    max_audio_length: int = 30  # seconds
    normalization: bool = True


@dataclass
class MultimodalConfig:
    """Multimodal configuration."""
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion_strategy: str = "concatenation"
    cross_attention: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_workers: int = 4
    batch_size: int = 32
    max_memory_gb: float = 8.0
    cache_size: int = 1000
    async_processing: bool = True
    optimization_level: str = "O1"


@dataclass
class EncryptionConfig:
    """Encryption configuration."""
    enabled: bool = True
    algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    at_rest: bool = True
    in_transit: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    audit_logging: bool = True
    rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    input_validation: bool = True
    output_sanitization: bool = True


@dataclass
class DataPathsConfig:
    """Data paths configuration."""
    models_path: str = "./data/models"
    datasets_path: str = "./data/datasets"
    logs_path: str = "./logs"
    temp_path: str = "./temp"
    cache_path: str = "./cache"
    exports_path: str = "./exports"


@dataclass
class ExternalServiceConfig:
    """External service configuration."""
    name: str
    url: str
    api_key: Optional[str] = None
    timeout: int = 30
    retries: int = 3
    enabled: bool = True


@dataclass
class ExternalServicesConfig:
    """External services configuration."""
    services: Dict[str, ExternalServiceConfig] = field(default_factory=dict)
    default_timeout: int = 30
    default_retries: int = 3


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    default_batch_size: int = 32
    max_parallel_stages: int = 4
    checkpoint_interval: int = 1000
    error_handling: str = "continue"  # continue, stop, retry
    retry_attempts: int = 3
    timeout: int = 3600  # 1 hour


@dataclass
class ResourceLimitsConfig:
    """Resource limits configuration."""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_gpu_memory_percent: float = 90.0
    max_disk_usage_gb: float = 100.0
    monitoring_interval: int = 60  # seconds


@dataclass
class DevelopmentConfig:
    """Development configuration."""
    hot_reload: bool = True
    debug_mode: bool = True
    profiling: bool = False
    testing: bool = False
    mock_external_services: bool = True


@dataclass
class TestingConfig:
    """Testing configuration."""
    enabled: bool = False
    test_data_path: str = "./tests/data"
    coverage_threshold: float = 80.0
    parallel_tests: bool = True
    test_timeout: int = 300  # 5 minutes
    mock_services: bool = True


@dataclass
class ConfigSchema:
    """Complete configuration schema for AIBF framework."""
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    neural_networks: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    transformers: TransformerConfig = field(default_factory=TransformerConfig)
    reinforcement_learning: ReinforcementLearningConfig = field(default_factory=ReinforcementLearningConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    data: DataPathsConfig = field(default_factory=DataPathsConfig)
    external_services: ExternalServicesConfig = field(default_factory=ExternalServicesConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    resource_limits: ResourceLimitsConfig = field(default_factory=ResourceLimitsConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)


def get_json_schema() -> Dict[str, Any]:
    """Get JSON schema for configuration validation.
    
    Returns:
        JSON schema dictionary
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AI Bull Ford Framework Configuration",
        "type": "object",
        "properties": {
            "framework": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+"},
                    "description": {"type": "string"},
                    "debug": {"type": "boolean"},
                    "environment": {"type": "string", "enum": ["development", "staging", "production"]}
                },
                "required": ["name", "version"]
            },
            "logging": {
                "type": "object",
                "properties": {
                    "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                    "format": {"type": "string"},
                    "file_path": {"type": ["string", "null"]},
                    "max_file_size": {"type": "integer", "minimum": 1024},
                    "backup_count": {"type": "integer", "minimum": 0},
                    "console_output": {"type": "boolean"},
                    "structured_logging": {"type": "boolean"}
                }
            },
            "api": {
                "type": "object",
                "properties": {
                    "rest": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "host": {"type": "string"},
                            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                            "workers": {"type": "integer", "minimum": 1},
                            "timeout": {"type": "integer", "minimum": 1},
                            "max_connections": {"type": "integer", "minimum": 1},
                            "cors_enabled": {"type": "boolean"},
                            "cors_origins": {"type": "array", "items": {"type": "string"}},
                            "docs_enabled": {"type": "boolean"},
                            "docs_url": {"type": "string"},
                            "openapi_url": {"type": "string"}
                        }
                    },
                    "websocket": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "host": {"type": "string"},
                            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                            "ping_interval": {"type": "integer", "minimum": 1},
                            "ping_timeout": {"type": "integer", "minimum": 1},
                            "max_message_size": {"type": "integer", "minimum": 1024}
                        }
                    },
                    "grpc": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "host": {"type": "string"},
                            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                            "max_message_length": {"type": "integer", "minimum": 1024},
                            "compression": {"type": "string", "enum": ["none", "gzip", "deflate"]}
                        }
                    },
                    "graphql": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "host": {"type": "string"},
                            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                            "playground_enabled": {"type": "boolean"},
                            "introspection_enabled": {"type": "boolean"},
                            "max_query_depth": {"type": "integer", "minimum": 1},
                            "max_query_complexity": {"type": "integer", "minimum": 1}
                        }
                    }
                }
            },
            "core": {
                "type": "object",
                "properties": {
                    "device": {"type": "string", "enum": ["cpu", "cuda", "mps", "auto"]},
                    "precision": {"type": "string", "enum": ["float16", "float32", "float64"]},
                    "seed": {"type": "integer", "minimum": 0},
                    "deterministic": {"type": "boolean"},
                    "benchmark": {"type": "boolean"},
                    "gradient_clipping": {"type": "number", "minimum": 0},
                    "mixed_precision": {"type": "boolean"}
                }
            },
            "performance": {
                "type": "object",
                "properties": {
                    "max_workers": {"type": "integer", "minimum": 1},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "max_memory_gb": {"type": "number", "minimum": 0.1},
                    "cache_size": {"type": "integer", "minimum": 0},
                    "async_processing": {"type": "boolean"},
                    "optimization_level": {"type": "string", "enum": ["O0", "O1", "O2", "O3"]}
                }
            },
            "data": {
                "type": "object",
                "properties": {
                    "models_path": {"type": "string"},
                    "datasets_path": {"type": "string"},
                    "logs_path": {"type": "string"},
                    "temp_path": {"type": "string"},
                    "cache_path": {"type": "string"},
                    "exports_path": {"type": "string"}
                },
                "required": ["models_path", "datasets_path"]
            }
        },
        "required": ["framework", "logging", "api", "core", "data"]
    }


def create_default_config() -> ConfigSchema:
    """Create a default configuration instance.
    
    Returns:
        Default ConfigSchema instance
    """
    return ConfigSchema()


def config_to_dict(config: ConfigSchema) -> Dict[str, Any]:
    """Convert ConfigSchema to dictionary.
    
    Args:
        config: ConfigSchema instance
        
    Returns:
        Configuration dictionary
    """
    import dataclasses
    return dataclasses.asdict(config)


def dict_to_config(config_dict: Dict[str, Any]) -> ConfigSchema:
    """Convert dictionary to ConfigSchema.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        ConfigSchema instance
    """
    # This is a simplified implementation
    # In practice, you might want to use a more sophisticated approach
    # that handles nested dataclasses properly
    
    config = ConfigSchema()
    
    # Update top-level fields
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config