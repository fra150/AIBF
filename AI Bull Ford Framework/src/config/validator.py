"""Configuration Validator for AI Bull Ford Framework.

Provides validation and schema checking for configuration data.
"""

import re
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def add_message(self, level: ValidationLevel, message: str, path: str = ""):
        """Add a validation message.
        
        Args:
            level: Validation level
            message: Validation message
            path: Configuration path where issue occurred
        """
        full_message = f"{path}: {message}" if path else message
        
        if level == ValidationLevel.ERROR:
            self.errors.append(full_message)
            self.valid = False
        elif level == ValidationLevel.WARNING:
            self.warnings.append(full_message)
        elif level == ValidationLevel.INFO:
            self.info.append(full_message)
    
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0
    
    def get_summary(self) -> str:
        """Get validation summary."""
        parts = []
        
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        if self.info:
            parts.append(f"{len(self.info)} info messages")
        
        if not parts:
            return "Validation passed"
        
        return f"Validation completed with {', '.join(parts)}"


@dataclass
class ValidationRule:
    """Configuration validation rule."""
    path: str
    validator: Callable[[Any], bool]
    message: str
    level: ValidationLevel = ValidationLevel.ERROR
    required: bool = True
    
    def validate(self, config: Dict[str, Any]) -> Optional[str]:
        """Validate configuration against this rule.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Error message if validation fails, None otherwise
        """
        try:
            value = self._get_nested_value(config, self.path)
            
            if value is None and self.required:
                return f"Required configuration '{self.path}' is missing"
            
            if value is not None and not self.validator(value):
                return self.message
            
            return None
            
        except Exception as e:
            return f"Validation error for '{self.path}': {e}"
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path
            
        Returns:
            Configuration value or None
        """
        keys = path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        
        return current


class ConfigValidator:
    """Validates configuration against defined rules and schemas."""
    
    def __init__(self):
        """Initialize configuration validator."""
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules for AIBF framework."""
        # Framework general settings
        self.add_rule(
            'framework.name',
            lambda x: isinstance(x, str) and len(x) > 0,
            'Framework name must be a non-empty string'
        )
        
        self.add_rule(
            'framework.version',
            lambda x: isinstance(x, str) and re.match(r'^\d+\.\d+\.\d+', x),
            'Framework version must be in semantic version format (x.y.z)'
        )
        
        # Logging configuration
        self.add_rule(
            'logging.level',
            lambda x: x in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            'Logging level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL'
        )
        
        self.add_rule(
            'logging.format',
            lambda x: isinstance(x, str) and len(x) > 0,
            'Logging format must be a non-empty string'
        )
        
        # API server configurations
        self.add_rule(
            'api.rest.host',
            lambda x: isinstance(x, str) and len(x) > 0,
            'REST API host must be a non-empty string'
        )
        
        self.add_rule(
            'api.rest.port',
            lambda x: isinstance(x, int) and 1 <= x <= 65535,
            'REST API port must be an integer between 1 and 65535'
        )
        
        self.add_rule(
            'api.websocket.port',
            lambda x: isinstance(x, int) and 1 <= x <= 65535,
            'WebSocket port must be an integer between 1 and 65535'
        )
        
        self.add_rule(
            'api.grpc.port',
            lambda x: isinstance(x, int) and 1 <= x <= 65535,
            'gRPC port must be an integer between 1 and 65535'
        )
        
        # Authentication settings
        self.add_rule(
            'auth.jwt.secret_key',
            lambda x: isinstance(x, str) and len(x) >= 32,
            'JWT secret key must be at least 32 characters long',
            level=ValidationLevel.ERROR
        )
        
        self.add_rule(
            'auth.jwt.algorithm',
            lambda x: x in ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512'],
            'JWT algorithm must be one of: HS256, HS384, HS512, RS256, RS384, RS512'
        )
        
        # Database configuration
        self.add_rule(
            'database.url',
            lambda x: isinstance(x, str) and ('://' in x or x.startswith('sqlite:')),
            'Database URL must be a valid connection string'
        )
        
        # Core module settings
        self.add_rule(
            'core.device',
            lambda x: x in ['cpu', 'cuda', 'mps', 'auto'],
            'Core device must be one of: cpu, cuda, mps, auto'
        )
        
        self.add_rule(
            'core.precision',
            lambda x: x in ['float16', 'float32', 'float64'],
            'Core precision must be one of: float16, float32, float64'
        )
        
        # Performance settings
        self.add_rule(
            'performance.max_workers',
            lambda x: isinstance(x, int) and x > 0,
            'Max workers must be a positive integer'
        )
        
        self.add_rule(
            'performance.batch_size',
            lambda x: isinstance(x, int) and x > 0,
            'Batch size must be a positive integer'
        )
        
        # Security settings
        self.add_rule(
            'security.encryption.enabled',
            lambda x: isinstance(x, bool),
            'Encryption enabled must be a boolean value'
        )
        
        # Data paths
        self.add_rule(
            'data.models_path',
            lambda x: isinstance(x, str) and len(x) > 0,
            'Models path must be a non-empty string'
        )
        
        self.add_rule(
            'data.datasets_path',
            lambda x: isinstance(x, str) and len(x) > 0,
            'Datasets path must be a non-empty string'
        )
    
    def add_rule(self, path: str, validator: Callable[[Any], bool], 
                message: str, level: ValidationLevel = ValidationLevel.ERROR,
                required: bool = True):
        """Add a validation rule.
        
        Args:
            path: Configuration path to validate
            validator: Validation function
            message: Error message
            level: Validation level
            required: Whether the configuration is required
        """
        rule = ValidationRule(
            path=path,
            validator=validator,
            message=message,
            level=level,
            required=required
        )
        self.rules.append(rule)
        logger.debug(f"Added validation rule for {path}")
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against all rules.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult(valid=True)
        
        logger.info(f"Validating configuration with {len(self.rules)} rules")
        
        for rule in self.rules:
            error_message = rule.validate(config)
            
            if error_message:
                result.add_message(rule.level, error_message, rule.path)
        
        # Additional custom validations
        self._validate_port_conflicts(config, result)
        self._validate_path_accessibility(config, result)
        self._validate_resource_limits(config, result)
        
        logger.info(result.get_summary())
        return result
    
    def _validate_port_conflicts(self, config: Dict[str, Any], 
                               result: ValidationResult):
        """Check for port conflicts between services.
        
        Args:
            config: Configuration dictionary
            result: ValidationResult to update
        """
        ports = []
        port_services = []
        
        # Collect all configured ports
        api_config = config.get('api', {})
        
        for service in ['rest', 'websocket', 'grpc', 'graphql']:
            service_config = api_config.get(service, {})
            if service_config.get('enabled', False):
                port = service_config.get('port')
                if port:
                    ports.append(port)
                    port_services.append(f"{service} API")
        
        # Check for duplicates
        seen_ports = set()
        for port, service in zip(ports, port_services):
            if port in seen_ports:
                result.add_message(
                    ValidationLevel.ERROR,
                    f"Port {port} is used by multiple services",
                    f"api.{service.lower().replace(' ', '_')}.port"
                )
            seen_ports.add(port)
    
    def _validate_path_accessibility(self, config: Dict[str, Any], 
                                   result: ValidationResult):
        """Validate that configured paths are accessible.
        
        Args:
            config: Configuration dictionary
            result: ValidationResult to update
        """
        import os
        
        data_config = config.get('data', {})
        
        for path_key in ['models_path', 'datasets_path', 'logs_path', 'temp_path']:
            path = data_config.get(path_key)
            
            if path and isinstance(path, str):
                try:
                    # Try to create directory if it doesn't exist
                    os.makedirs(path, exist_ok=True)
                    
                    # Check if directory is writable
                    if not os.access(path, os.W_OK):
                        result.add_message(
                            ValidationLevel.WARNING,
                            f"Path '{path}' is not writable",
                            f"data.{path_key}"
                        )
                        
                except Exception as e:
                    result.add_message(
                        ValidationLevel.ERROR,
                        f"Cannot access path '{path}': {e}",
                        f"data.{path_key}"
                    )
    
    def _validate_resource_limits(self, config: Dict[str, Any], 
                                result: ValidationResult):
        """Validate resource limit configurations.
        
        Args:
            config: Configuration dictionary
            result: ValidationResult to update
        """
        performance = config.get('performance', {})
        
        # Check memory limits
        max_memory = performance.get('max_memory_gb')
        if max_memory and isinstance(max_memory, (int, float)):
            if max_memory < 1:
                result.add_message(
                    ValidationLevel.WARNING,
                    "Max memory limit is very low (< 1GB)",
                    "performance.max_memory_gb"
                )
            elif max_memory > 128:
                result.add_message(
                    ValidationLevel.INFO,
                    "Max memory limit is very high (> 128GB)",
                    "performance.max_memory_gb"
                )
        
        # Check worker limits
        max_workers = performance.get('max_workers')
        if max_workers and isinstance(max_workers, int):
            import os
            cpu_count = os.cpu_count() or 1
            
            if max_workers > cpu_count * 2:
                result.add_message(
                    ValidationLevel.WARNING,
                    f"Max workers ({max_workers}) exceeds 2x CPU count ({cpu_count})",
                    "performance.max_workers"
                )
    
    def validate_schema(self, config: Dict[str, Any], 
                       schema: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against a JSON schema.
        
        Args:
            config: Configuration to validate
            schema: JSON schema definition
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)
        
        try:
            import jsonschema
            jsonschema.validate(config, schema)
            
        except ImportError:
            result.add_message(
                ValidationLevel.WARNING,
                "jsonschema package not available, skipping schema validation"
            )
            
        except jsonschema.ValidationError as e:
            result.add_message(
                ValidationLevel.ERROR,
                f"Schema validation failed: {e.message}",
                '.'.join(str(p) for p in e.absolute_path)
            )
            
        except Exception as e:
            result.add_message(
                ValidationLevel.ERROR,
                f"Schema validation error: {e}"
            )
        
        return result
    
    def get_rule_count(self) -> int:
        """Get the number of validation rules.
        
        Returns:
            Number of rules
        """
        return len(self.rules)
    
    def clear_rules(self):
        """Clear all validation rules."""
        self.rules.clear()
        logger.debug("All validation rules cleared")
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """List all validation rules.
        
        Returns:
            List of rule information
        """
        return [
            {
                'path': rule.path,
                'message': rule.message,
                'level': rule.level.value,
                'required': rule.required
            }
            for rule in self.rules
        ]


def create_default_validator() -> ConfigValidator:
    """Create a ConfigValidator with default rules.
    
    Returns:
        Configured ConfigValidator instance
    """
    return ConfigValidator()