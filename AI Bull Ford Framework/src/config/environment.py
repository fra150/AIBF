"""Environment Configuration for AI Bull Ford Framework.

Handles environment-specific configuration and variable management.
"""

import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class EnvironmentVariable:
    """Environment variable definition."""
    name: str
    default: Optional[Any] = None
    required: bool = False
    description: str = ""
    type_hint: type = str
    validator: Optional[callable] = None


class EnvironmentConfig:
    """Manages environment-specific configuration."""
    
    def __init__(self, prefix: str = "AIBF_"):
        """Initialize environment configuration.
        
        Args:
            prefix: Environment variable prefix
        """
        self.prefix = prefix
        self.variables: Dict[str, EnvironmentVariable] = {}
        self.loaded_values: Dict[str, Any] = {}
        self._setup_default_variables()
    
    def _setup_default_variables(self):
        """Setup default environment variables for AIBF framework."""
        # Framework environment
        self.register_variable(
            "ENV",
            default=Environment.DEVELOPMENT.value,
            description="Application environment",
            validator=lambda x: x in [e.value for e in Environment]
        )
        
        self.register_variable(
            "DEBUG",
            default=False,
            description="Enable debug mode",
            type_hint=bool
        )
        
        self.register_variable(
            "LOG_LEVEL",
            default="INFO",
            description="Logging level",
            validator=lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
        
        # API Configuration
        self.register_variable(
            "API_HOST",
            default="localhost",
            description="API server host"
        )
        
        self.register_variable(
            "API_PORT",
            default=8000,
            description="API server port",
            type_hint=int,
            validator=lambda x: 1 <= x <= 65535
        )
        
        self.register_variable(
            "API_WORKERS",
            default=1,
            description="Number of API workers",
            type_hint=int,
            validator=lambda x: x > 0
        )
        
        # Database Configuration
        self.register_variable(
            "DATABASE_URL",
            default="sqlite:///aibf.db",
            description="Database connection URL"
        )
        
        self.register_variable(
            "DATABASE_POOL_SIZE",
            default=5,
            description="Database connection pool size",
            type_hint=int,
            validator=lambda x: x > 0
        )
        
        # Redis Configuration
        self.register_variable(
            "REDIS_URL",
            default="redis://localhost:6379/0",
            description="Redis connection URL"
        )
        
        self.register_variable(
            "REDIS_ENABLED",
            default=False,
            description="Enable Redis",
            type_hint=bool
        )
        
        # Security Configuration
        self.register_variable(
            "SECRET_KEY",
            required=True,
            description="Secret key for encryption and JWT",
            validator=lambda x: len(x) >= 32
        )
        
        self.register_variable(
            "JWT_ALGORITHM",
            default="HS256",
            description="JWT signing algorithm",
            validator=lambda x: x in ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]
        )
        
        self.register_variable(
            "JWT_EXPIRE_MINUTES",
            default=30,
            description="JWT token expiration in minutes",
            type_hint=int,
            validator=lambda x: x > 0
        )
        
        # Core Configuration
        self.register_variable(
            "DEVICE",
            default="auto",
            description="Compute device (cpu, cuda, mps, auto)",
            validator=lambda x: x in ["cpu", "cuda", "mps", "auto"]
        )
        
        self.register_variable(
            "PRECISION",
            default="float32",
            description="Compute precision (float16, float32, float64)",
            validator=lambda x: x in ["float16", "float32", "float64"]
        )
        
        self.register_variable(
            "SEED",
            default=42,
            description="Random seed",
            type_hint=int,
            validator=lambda x: x >= 0
        )
        
        # Performance Configuration
        self.register_variable(
            "MAX_WORKERS",
            default=4,
            description="Maximum number of worker processes",
            type_hint=int,
            validator=lambda x: x > 0
        )
        
        self.register_variable(
            "BATCH_SIZE",
            default=32,
            description="Default batch size",
            type_hint=int,
            validator=lambda x: x > 0
        )
        
        self.register_variable(
            "MAX_MEMORY_GB",
            default=8.0,
            description="Maximum memory usage in GB",
            type_hint=float,
            validator=lambda x: x > 0
        )
        
        # Data Paths
        self.register_variable(
            "MODELS_PATH",
            default="./data/models",
            description="Path to model files"
        )
        
        self.register_variable(
            "DATASETS_PATH",
            default="./data/datasets",
            description="Path to dataset files"
        )
        
        self.register_variable(
            "LOGS_PATH",
            default="./logs",
            description="Path to log files"
        )
        
        self.register_variable(
            "TEMP_PATH",
            default="./temp",
            description="Path to temporary files"
        )
        
        # External Services
        self.register_variable(
            "EXTERNAL_API_TIMEOUT",
            default=30,
            description="External API timeout in seconds",
            type_hint=int,
            validator=lambda x: x > 0
        )
        
        self.register_variable(
            "EXTERNAL_API_RETRIES",
            default=3,
            description="External API retry attempts",
            type_hint=int,
            validator=lambda x: x >= 0
        )
        
        # Monitoring
        self.register_variable(
            "METRICS_ENABLED",
            default=True,
            description="Enable metrics collection",
            type_hint=bool
        )
        
        self.register_variable(
            "HEALTH_CHECK_ENABLED",
            default=True,
            description="Enable health checks",
            type_hint=bool
        )
        
        # Development
        self.register_variable(
            "HOT_RELOAD",
            default=True,
            description="Enable hot reloading in development",
            type_hint=bool
        )
        
        self.register_variable(
            "MOCK_EXTERNAL_SERVICES",
            default=False,
            description="Mock external services for testing",
            type_hint=bool
        )
    
    def register_variable(self, name: str, default: Optional[Any] = None,
                         required: bool = False, description: str = "",
                         type_hint: type = str, validator: Optional[callable] = None):
        """Register an environment variable.
        
        Args:
            name: Variable name (without prefix)
            default: Default value
            required: Whether the variable is required
            description: Variable description
            type_hint: Expected type
            validator: Validation function
        """
        variable = EnvironmentVariable(
            name=name,
            default=default,
            required=required,
            description=description,
            type_hint=type_hint,
            validator=validator
        )
        
        self.variables[name] = variable
        logger.debug(f"Registered environment variable: {self.prefix}{name}")
    
    def load(self) -> Dict[str, Any]:
        """Load all environment variables.
        
        Returns:
            Dictionary of loaded environment variables
        """
        self.loaded_values.clear()
        errors = []
        
        for name, variable in self.variables.items():
            env_name = f"{self.prefix}{name}"
            
            try:
                value = self._load_variable(env_name, variable)
                self.loaded_values[name] = value
                
            except Exception as e:
                error_msg = f"Error loading {env_name}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        if errors:
            raise ValueError(f"Failed to load environment variables: {'; '.join(errors)}")
        
        logger.info(f"Loaded {len(self.loaded_values)} environment variables")
        return self.loaded_values.copy()
    
    def _load_variable(self, env_name: str, variable: EnvironmentVariable) -> Any:
        """Load a single environment variable.
        
        Args:
            env_name: Full environment variable name
            variable: Variable definition
            
        Returns:
            Loaded and converted value
        """
        raw_value = os.getenv(env_name)
        
        # Handle missing values
        if raw_value is None:
            if variable.required:
                raise ValueError(f"Required environment variable {env_name} is not set")
            return variable.default
        
        # Convert type
        try:
            converted_value = self._convert_value(raw_value, variable.type_hint)
        except Exception as e:
            raise ValueError(f"Cannot convert {env_name}='{raw_value}' to {variable.type_hint.__name__}: {e}")
        
        # Validate
        if variable.validator:
            try:
                if not variable.validator(converted_value):
                    raise ValueError(f"Validation failed for {env_name}='{raw_value}'")
            except Exception as e:
                raise ValueError(f"Validation error for {env_name}: {e}")
        
        return converted_value
    
    def _convert_value(self, value: str, type_hint: type) -> Any:
        """Convert string value to specified type.
        
        Args:
            value: String value
            type_hint: Target type
            
        Returns:
            Converted value
        """
        if type_hint == str:
            return value
        elif type_hint == bool:
            return value.lower() in ('true', 'yes', '1', 'on')
        elif type_hint == int:
            return int(value)
        elif type_hint == float:
            return float(value)
        elif type_hint == list:
            # Assume comma-separated values
            return [item.strip() for item in value.split(',') if item.strip()]
        else:
            # Try direct conversion
            return type_hint(value)
    
    def get(self, name: str, default: Optional[Any] = None) -> Any:
        """Get an environment variable value.
        
        Args:
            name: Variable name (without prefix)
            default: Default value if not found
            
        Returns:
            Variable value
        """
        return self.loaded_values.get(name, default)
    
    def set(self, name: str, value: Any):
        """Set an environment variable value.
        
        Args:
            name: Variable name (without prefix)
            value: Value to set
        """
        env_name = f"{self.prefix}{name}"
        os.environ[env_name] = str(value)
        
        # Update loaded values if variable is registered
        if name in self.variables:
            variable = self.variables[name]
            try:
                converted_value = self._convert_value(str(value), variable.type_hint)
                if variable.validator and not variable.validator(converted_value):
                    raise ValueError(f"Validation failed for {name}")
                self.loaded_values[name] = converted_value
            except Exception as e:
                logger.error(f"Error setting {name}: {e}")
                raise
    
    def get_environment(self) -> Environment:
        """Get the current environment.
        
        Returns:
            Current environment
        """
        env_value = self.get("ENV", Environment.DEVELOPMENT.value)
        
        try:
            return Environment(env_value)
        except ValueError:
            logger.warning(f"Unknown environment '{env_value}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.get_environment() == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment() == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.get_environment() == Environment.TESTING
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as a nested dictionary.
        
        Returns:
            Nested configuration dictionary
        """
        config = {}
        
        # Group variables by category
        categories = {
            'framework': ['ENV', 'DEBUG'],
            'logging': ['LOG_LEVEL'],
            'api': ['API_HOST', 'API_PORT', 'API_WORKERS'],
            'database': ['DATABASE_URL', 'DATABASE_POOL_SIZE'],
            'redis': ['REDIS_URL', 'REDIS_ENABLED'],
            'auth': ['SECRET_KEY', 'JWT_ALGORITHM', 'JWT_EXPIRE_MINUTES'],
            'core': ['DEVICE', 'PRECISION', 'SEED'],
            'performance': ['MAX_WORKERS', 'BATCH_SIZE', 'MAX_MEMORY_GB'],
            'data': ['MODELS_PATH', 'DATASETS_PATH', 'LOGS_PATH', 'TEMP_PATH'],
            'external_services': ['EXTERNAL_API_TIMEOUT', 'EXTERNAL_API_RETRIES'],
            'monitoring': ['METRICS_ENABLED', 'HEALTH_CHECK_ENABLED'],
            'development': ['HOT_RELOAD', 'MOCK_EXTERNAL_SERVICES']
        }
        
        for category, var_names in categories.items():
            category_config = {}
            
            for var_name in var_names:
                if var_name in self.loaded_values:
                    # Convert variable name to config key
                    config_key = var_name.lower()
                    if category == 'api' and config_key.startswith('api_'):
                        config_key = config_key[4:]  # Remove 'api_' prefix
                    elif category == 'database' and config_key.startswith('database_'):
                        config_key = config_key[9:]  # Remove 'database_' prefix
                    elif category == 'redis' and config_key.startswith('redis_'):
                        config_key = config_key[6:]  # Remove 'redis_' prefix
                    elif category == 'jwt' and config_key.startswith('jwt_'):
                        config_key = config_key[4:]  # Remove 'jwt_' prefix
                    
                    category_config[config_key] = self.loaded_values[var_name]
            
            if category_config:
                config[category] = category_config
        
        return config
    
    def validate_all(self) -> List[str]:
        """Validate all loaded environment variables.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        for name, variable in self.variables.items():
            if name not in self.loaded_values:
                if variable.required:
                    errors.append(f"Required variable {self.prefix}{name} is missing")
                continue
            
            value = self.loaded_values[name]
            
            # Type validation
            if not isinstance(value, variable.type_hint):
                errors.append(
                    f"Variable {self.prefix}{name} has wrong type: "
                    f"expected {variable.type_hint.__name__}, got {type(value).__name__}"
                )
            
            # Custom validation
            if variable.validator:
                try:
                    if not variable.validator(value):
                        errors.append(f"Validation failed for {self.prefix}{name}")
                except Exception as e:
                    errors.append(f"Validation error for {self.prefix}{name}: {e}")
        
        return errors
    
    def list_variables(self) -> List[Dict[str, Any]]:
        """List all registered environment variables.
        
        Returns:
            List of variable information
        """
        variables_info = []
        
        for name, variable in self.variables.items():
            env_name = f"{self.prefix}{name}"
            current_value = os.getenv(env_name)
            loaded_value = self.loaded_values.get(name)
            
            variables_info.append({
                'name': name,
                'env_name': env_name,
                'description': variable.description,
                'type': variable.type_hint.__name__,
                'required': variable.required,
                'default': variable.default,
                'current_value': current_value,
                'loaded_value': loaded_value,
                'is_set': current_value is not None
            })
        
        return variables_info
    
    def export_template(self, include_values: bool = False) -> str:
        """Export environment variables as a template file.
        
        Args:
            include_values: Include current values in template
            
        Returns:
            Environment file template
        """
        lines = [
            "# AI Bull Ford Framework Environment Configuration",
            "# Generated template - customize for your environment",
            ""
        ]
        
        # Group variables by category for better organization
        current_category = None
        
        for name, variable in sorted(self.variables.items()):
            # Determine category from variable name
            category = self._get_variable_category(name)
            
            if category != current_category:
                lines.append(f"\n# {category.title()} Configuration")
                current_category = category
            
            # Add description as comment
            if variable.description:
                lines.append(f"# {variable.description}")
            
            # Add type and validation info
            type_info = f"Type: {variable.type_hint.__name__}"
            if variable.required:
                type_info += " (required)"
            lines.append(f"# {type_info}")
            
            # Add variable line
            env_name = f"{self.prefix}{name}"
            
            if include_values:
                current_value = os.getenv(env_name)
                if current_value is not None:
                    lines.append(f"{env_name}={current_value}")
                elif variable.default is not None:
                    lines.append(f"{env_name}={variable.default}")
                else:
                    lines.append(f"# {env_name}=")
            else:
                if variable.default is not None:
                    lines.append(f"# {env_name}={variable.default}")
                else:
                    lines.append(f"# {env_name}=")
            
            lines.append("")  # Empty line for readability
        
        return "\n".join(lines)
    
    def _get_variable_category(self, name: str) -> str:
        """Get category for a variable name.
        
        Args:
            name: Variable name
            
        Returns:
            Category name
        """
        if name in ['ENV', 'DEBUG']:
            return 'framework'
        elif name.startswith('LOG_'):
            return 'logging'
        elif name.startswith('API_'):
            return 'api'
        elif name.startswith('DATABASE_'):
            return 'database'
        elif name.startswith('REDIS_'):
            return 'redis'
        elif name in ['SECRET_KEY'] or name.startswith('JWT_'):
            return 'security'
        elif name in ['DEVICE', 'PRECISION', 'SEED']:
            return 'core'
        elif name in ['MAX_WORKERS', 'BATCH_SIZE', 'MAX_MEMORY_GB']:
            return 'performance'
        elif name.endswith('_PATH'):
            return 'data'
        elif name.startswith('EXTERNAL_'):
            return 'external_services'
        elif name.endswith('_ENABLED'):
            return 'monitoring'
        else:
            return 'other'


# Global environment configuration instance
_global_env_config: Optional[EnvironmentConfig] = None


def get_global_env_config() -> EnvironmentConfig:
    """Get the global environment configuration instance.
    
    Returns:
        Global EnvironmentConfig instance
    """
    global _global_env_config
    
    if _global_env_config is None:
        _global_env_config = EnvironmentConfig()
        _global_env_config.load()
    
    return _global_env_config


def initialize_global_env_config(prefix: str = "AIBF_") -> EnvironmentConfig:
    """Initialize the global environment configuration.
    
    Args:
        prefix: Environment variable prefix
        
    Returns:
        Initialized EnvironmentConfig instance
    """
    global _global_env_config
    
    _global_env_config = EnvironmentConfig(prefix)
    _global_env_config.load()
    
    return _global_env_config


def get_env(name: str, default: Optional[Any] = None) -> Any:
    """Get environment variable value from global config.
    
    Args:
        name: Variable name (without prefix)
        default: Default value
        
    Returns:
        Environment variable value
    """
    env_config = get_global_env_config()
    return env_config.get(name, default)


def set_env(name: str, value: Any):
    """Set environment variable value in global config.
    
    Args:
        name: Variable name (without prefix)
        value: Value to set
    """
    env_config = get_global_env_config()
    env_config.set(name, value)