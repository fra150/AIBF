"""Validation module for AI Bull Ford security layer.

Provides input validation, sanitization, and data integrity checks.
"""

import re
import json
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date
import logging
import hashlib
import base64
import secrets
from pathlib import Path
import mimetypes
from decimal import Decimal, InvalidOperation
import ipaddress
from email.utils import parseaddr


class ValidationType(Enum):
    """Types of validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    DATE = "date"
    DATETIME = "datetime"
    JSON = "json"
    FILE = "file"
    REGEX = "regex"
    ENUM = "enum"
    LIST = "list"
    DICT = "dict"
    CUSTOM = "custom"


class SanitizationType(Enum):
    """Types of sanitization."""
    HTML_ESCAPE = "html_escape"
    URL_ENCODE = "url_encode"
    SQL_ESCAPE = "sql_escape"
    STRIP_WHITESPACE = "strip_whitespace"
    REMOVE_SPECIAL_CHARS = "remove_special_chars"
    NORMALIZE_UNICODE = "normalize_unicode"
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    REMOVE_HTML_TAGS = "remove_html_tags"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    validation_type: ValidationType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    error_message: Optional[str] = None
    sanitization: List[SanitizationType] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            'name': self.name,
            'validation_type': self.validation_type.value,
            'required': self.required,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'pattern': self.pattern,
            'allowed_values': self.allowed_values,
            'error_message': self.error_message,
            'sanitization': [s.value for s in self.sanitization],
            'metadata': self.metadata
        }


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_value: Any = None
    original_value: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'sanitized_value': self.sanitized_value,
            'original_value': self.original_value
        }


class Validator(ABC):
    """Abstract validator interface."""
    
    @abstractmethod
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate a value against a rule.
        
        Args:
            value: Value to validate
            rule: Validation rule
            
        Returns:
            Validation result
        """
        pass


class StringValidator(Validator):
    """String validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate string value."""
        result = ValidationResult(valid=True, original_value=value)
        
        # Convert to string if not already
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = ""
                return result
        
        str_value = str(value)
        result.sanitized_value = str_value
        
        # Length validation
        if rule.min_length is not None and len(str_value) < rule.min_length:
            result.valid = False
            result.errors.append(f"{rule.name} must be at least {rule.min_length} characters")
        
        if rule.max_length is not None and len(str_value) > rule.max_length:
            result.valid = False
            result.errors.append(f"{rule.name} must be at most {rule.max_length} characters")
        
        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, str_value):
            result.valid = False
            error_msg = rule.error_message or f"{rule.name} does not match required pattern"
            result.errors.append(error_msg)
        
        # Allowed values validation
        if rule.allowed_values and str_value not in rule.allowed_values:
            result.valid = False
            result.errors.append(f"{rule.name} must be one of: {', '.join(map(str, rule.allowed_values))}")
        
        return result


class NumericValidator(Validator):
    """Numeric validator for integers and floats."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate numeric value."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = None
                return result
        
        # Convert to appropriate numeric type
        try:
            if rule.validation_type == ValidationType.INTEGER:
                numeric_value = int(value)
            else:  # FLOAT
                numeric_value = float(value)
            
            result.sanitized_value = numeric_value
            
        except (ValueError, TypeError):
            result.valid = False
            result.errors.append(f"{rule.name} must be a valid number")
            return result
        
        # Range validation
        if rule.min_value is not None and numeric_value < rule.min_value:
            result.valid = False
            result.errors.append(f"{rule.name} must be at least {rule.min_value}")
        
        if rule.max_value is not None and numeric_value > rule.max_value:
            result.valid = False
            result.errors.append(f"{rule.name} must be at most {rule.max_value}")
        
        # Allowed values validation
        if rule.allowed_values and numeric_value not in rule.allowed_values:
            result.valid = False
            result.errors.append(f"{rule.name} must be one of: {', '.join(map(str, rule.allowed_values))}")
        
        return result


class BooleanValidator(Validator):
    """Boolean validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate boolean value."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = False
                return result
        
        # Convert to boolean
        if isinstance(value, bool):
            result.sanitized_value = value
        elif isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ['true', '1', 'yes', 'on']:
                result.sanitized_value = True
            elif lower_value in ['false', '0', 'no', 'off']:
                result.sanitized_value = False
            else:
                result.valid = False
                result.errors.append(f"{rule.name} must be a valid boolean value")
        elif isinstance(value, (int, float)):
            result.sanitized_value = bool(value)
        else:
            result.valid = False
            result.errors.append(f"{rule.name} must be a valid boolean value")
        
        return result


class EmailValidator(Validator):
    """Email validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate email address."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = ""
                return result
        
        email_str = str(value).strip().lower()
        result.sanitized_value = email_str
        
        # Basic email pattern validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email_str):
            result.valid = False
            result.errors.append(f"{rule.name} must be a valid email address")
            return result
        
        # Additional validation using parseaddr
        parsed_name, parsed_email = parseaddr(email_str)
        if not parsed_email or '@' not in parsed_email:
            result.valid = False
            result.errors.append(f"{rule.name} must be a valid email address")
        
        return result


class URLValidator(Validator):
    """URL validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate URL."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = ""
                return result
        
        url_str = str(value).strip()
        result.sanitized_value = url_str
        
        # Basic URL pattern validation
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, url_str, re.IGNORECASE):
            result.valid = False
            result.errors.append(f"{rule.name} must be a valid URL")
            return result
        
        # Additional validation using urllib.parse
        try:
            parsed = urllib.parse.urlparse(url_str)
            if not all([parsed.scheme, parsed.netloc]):
                result.valid = False
                result.errors.append(f"{rule.name} must be a valid URL")
        except Exception:
            result.valid = False
            result.errors.append(f"{rule.name} must be a valid URL")
        
        return result


class IPAddressValidator(Validator):
    """IP address validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate IP address."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = ""
                return result
        
        ip_str = str(value).strip()
        result.sanitized_value = ip_str
        
        try:
            # This will validate both IPv4 and IPv6
            ipaddress.ip_address(ip_str)
        except ValueError:
            result.valid = False
            result.errors.append(f"{rule.name} must be a valid IP address")
        
        return result


class DateTimeValidator(Validator):
    """Date and datetime validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate date or datetime."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = None
                return result
        
        # If already a date/datetime object
        if isinstance(value, (date, datetime)):
            result.sanitized_value = value
            return result
        
        # Try to parse string
        date_str = str(value).strip()
        
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M',
        ]
        
        parsed_date = None
        for fmt in date_formats:
            try:
                if rule.validation_type == ValidationType.DATE:
                    parsed_date = datetime.strptime(date_str, fmt).date()
                else:  # DATETIME
                    parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        
        if parsed_date is None:
            result.valid = False
            result.errors.append(f"{rule.name} must be a valid date/time")
        else:
            result.sanitized_value = parsed_date
        
        return result


class JSONValidator(Validator):
    """JSON validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate JSON."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = None
                return result
        
        # If already a dict/list, assume it's valid JSON data
        if isinstance(value, (dict, list)):
            result.sanitized_value = value
            return result
        
        # Try to parse JSON string
        try:
            json_data = json.loads(str(value))
            result.sanitized_value = json_data
        except (json.JSONDecodeError, TypeError):
            result.valid = False
            result.errors.append(f"{rule.name} must be valid JSON")
        
        return result


class FileValidator(Validator):
    """File validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate file."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = None
                return result
        
        # Handle different file representations
        file_path = None
        file_size = None
        file_type = None
        
        if isinstance(value, str):
            # File path
            file_path = Path(value)
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_type = mimetypes.guess_type(str(file_path))[0]
        elif isinstance(value, dict):
            # File info dict
            file_path = value.get('path')
            file_size = value.get('size')
            file_type = value.get('type')
        
        result.sanitized_value = {
            'path': str(file_path) if file_path else None,
            'size': file_size,
            'type': file_type
        }
        
        # Validate file exists
        if file_path and not Path(file_path).exists():
            result.valid = False
            result.errors.append(f"{rule.name} file does not exist")
            return result
        
        # Validate file size
        if file_size is not None:
            if rule.min_value is not None and file_size < rule.min_value:
                result.valid = False
                result.errors.append(f"{rule.name} file size must be at least {rule.min_value} bytes")
            
            if rule.max_value is not None and file_size > rule.max_value:
                result.valid = False
                result.errors.append(f"{rule.name} file size must be at most {rule.max_value} bytes")
        
        # Validate file type
        if rule.allowed_values and file_type:
            if file_type not in rule.allowed_values:
                result.valid = False
                result.errors.append(f"{rule.name} file type must be one of: {', '.join(rule.allowed_values)}")
        
        return result


class ListValidator(Validator):
    """List validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate list."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = []
                return result
        
        # Convert to list if not already
        if not isinstance(value, list):
            if isinstance(value, (tuple, set)):
                list_value = list(value)
            else:
                list_value = [value]
        else:
            list_value = value
        
        result.sanitized_value = list_value
        
        # Length validation
        if rule.min_length is not None and len(list_value) < rule.min_length:
            result.valid = False
            result.errors.append(f"{rule.name} must have at least {rule.min_length} items")
        
        if rule.max_length is not None and len(list_value) > rule.max_length:
            result.valid = False
            result.errors.append(f"{rule.name} must have at most {rule.max_length} items")
        
        return result


class DictValidator(Validator):
    """Dictionary validator."""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate dictionary."""
        result = ValidationResult(valid=True, original_value=value)
        
        if value is None:
            if rule.required:
                result.valid = False
                result.errors.append(f"{rule.name} is required")
                return result
            else:
                result.sanitized_value = {}
                return result
        
        # Must be a dictionary
        if not isinstance(value, dict):
            result.valid = False
            result.errors.append(f"{rule.name} must be a dictionary")
            return result
        
        result.sanitized_value = value
        
        # Length validation (number of keys)
        if rule.min_length is not None and len(value) < rule.min_length:
            result.valid = False
            result.errors.append(f"{rule.name} must have at least {rule.min_length} keys")
        
        if rule.max_length is not None and len(value) > rule.max_length:
            result.valid = False
            result.errors.append(f"{rule.name} must have at most {rule.max_length} keys")
        
        return result


class Sanitizer:
    """Data sanitizer."""
    
    def __init__(self):
        """Initialize sanitizer."""
        self.logger = logging.getLogger("security.sanitizer")
    
    def sanitize(self, value: Any, sanitization_types: List[SanitizationType]) -> Any:
        """Sanitize a value.
        
        Args:
            value: Value to sanitize
            sanitization_types: List of sanitization types to apply
            
        Returns:
            Sanitized value
        """
        if value is None:
            return None
        
        sanitized_value = value
        
        for sanitization_type in sanitization_types:
            sanitized_value = self._apply_sanitization(sanitized_value, sanitization_type)
        
        return sanitized_value
    
    def _apply_sanitization(self, value: Any, sanitization_type: SanitizationType) -> Any:
        """Apply a single sanitization.
        
        Args:
            value: Value to sanitize
            sanitization_type: Type of sanitization
            
        Returns:
            Sanitized value
        """
        if value is None:
            return None
        
        str_value = str(value)
        
        if sanitization_type == SanitizationType.HTML_ESCAPE:
            return html.escape(str_value)
        
        elif sanitization_type == SanitizationType.URL_ENCODE:
            return urllib.parse.quote(str_value)
        
        elif sanitization_type == SanitizationType.SQL_ESCAPE:
            # Basic SQL escaping - in production, use parameterized queries
            return str_value.replace("'", "''").replace('"', '""')
        
        elif sanitization_type == SanitizationType.STRIP_WHITESPACE:
            return str_value.strip()
        
        elif sanitization_type == SanitizationType.REMOVE_SPECIAL_CHARS:
            return re.sub(r'[^a-zA-Z0-9\s]', '', str_value)
        
        elif sanitization_type == SanitizationType.NORMALIZE_UNICODE:
            import unicodedata
            return unicodedata.normalize('NFKC', str_value)
        
        elif sanitization_type == SanitizationType.LOWERCASE:
            return str_value.lower()
        
        elif sanitization_type == SanitizationType.UPPERCASE:
            return str_value.upper()
        
        elif sanitization_type == SanitizationType.REMOVE_HTML_TAGS:
            return re.sub(r'<[^>]+>', '', str_value)
        
        else:
            self.logger.warning(f"Unknown sanitization type: {sanitization_type}")
            return value


class ValidationEngine:
    """Main validation engine."""
    
    def __init__(self):
        """Initialize validation engine."""
        self.validators = {
            ValidationType.STRING: StringValidator(),
            ValidationType.INTEGER: NumericValidator(),
            ValidationType.FLOAT: NumericValidator(),
            ValidationType.BOOLEAN: BooleanValidator(),
            ValidationType.EMAIL: EmailValidator(),
            ValidationType.URL: URLValidator(),
            ValidationType.IP_ADDRESS: IPAddressValidator(),
            ValidationType.DATE: DateTimeValidator(),
            ValidationType.DATETIME: DateTimeValidator(),
            ValidationType.JSON: JSONValidator(),
            ValidationType.FILE: FileValidator(),
            ValidationType.LIST: ListValidator(),
            ValidationType.DICT: DictValidator(),
        }
        
        self.sanitizer = Sanitizer()
        self.logger = logging.getLogger("security.validation")
    
    def validate_value(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate a single value.
        
        Args:
            value: Value to validate
            rule: Validation rule
            
        Returns:
            Validation result
        """
        try:
            # Apply sanitization first
            if rule.sanitization:
                sanitized_value = self.sanitizer.sanitize(value, rule.sanitization)
            else:
                sanitized_value = value
            
            # Get appropriate validator
            if rule.validation_type == ValidationType.CUSTOM and rule.custom_validator:
                # Custom validation
                result = ValidationResult(valid=True, original_value=value, sanitized_value=sanitized_value)
                
                try:
                    is_valid = rule.custom_validator(sanitized_value)
                    if not is_valid:
                        result.valid = False
                        error_msg = rule.error_message or f"{rule.name} failed custom validation"
                        result.errors.append(error_msg)
                except Exception as e:
                    result.valid = False
                    result.errors.append(f"{rule.name} custom validation error: {str(e)}")
                
                return result
            
            elif rule.validation_type == ValidationType.REGEX:
                # Regex validation
                result = ValidationResult(valid=True, original_value=value, sanitized_value=sanitized_value)
                
                if rule.pattern:
                    if sanitized_value is None:
                        if rule.required:
                            result.valid = False
                            result.errors.append(f"{rule.name} is required")
                    else:
                        str_value = str(sanitized_value)
                        if not re.match(rule.pattern, str_value):
                            result.valid = False
                            error_msg = rule.error_message or f"{rule.name} does not match required pattern"
                            result.errors.append(error_msg)
                
                return result
            
            elif rule.validation_type == ValidationType.ENUM:
                # Enum validation
                result = ValidationResult(valid=True, original_value=value, sanitized_value=sanitized_value)
                
                if sanitized_value is None:
                    if rule.required:
                        result.valid = False
                        result.errors.append(f"{rule.name} is required")
                elif rule.allowed_values and sanitized_value not in rule.allowed_values:
                    result.valid = False
                    result.errors.append(f"{rule.name} must be one of: {', '.join(map(str, rule.allowed_values))}")
                
                return result
            
            else:
                # Standard validation
                validator = self.validators.get(rule.validation_type)
                if not validator:
                    raise ValueError(f"Unknown validation type: {rule.validation_type}")
                
                result = validator.validate(sanitized_value, rule)
                result.original_value = value
                return result
        
        except Exception as e:
            self.logger.error(f"Validation error for {rule.name}: {e}")
            return ValidationResult(
                valid=False,
                errors=[f"Validation error: {str(e)}"],
                original_value=value
            )
    
    def validate_data(self, data: Dict[str, Any], rules: List[ValidationRule]) -> Dict[str, ValidationResult]:
        """Validate a dictionary of data.
        
        Args:
            data: Data to validate
            rules: List of validation rules
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Create rule lookup
        rule_lookup = {rule.name: rule for rule in rules}
        
        # Validate each field
        for field_name, rule in rule_lookup.items():
            value = data.get(field_name)
            results[field_name] = self.validate_value(value, rule)
        
        # Check for unexpected fields
        for field_name in data:
            if field_name not in rule_lookup:
                results[field_name] = ValidationResult(
                    valid=False,
                    errors=[f"Unexpected field: {field_name}"],
                    original_value=data[field_name]
                )
        
        return results
    
    def is_valid(self, results: Dict[str, ValidationResult]) -> bool:
        """Check if all validation results are valid.
        
        Args:
            results: Validation results
            
        Returns:
            True if all results are valid
        """
        return all(result.valid for result in results.values())
    
    def get_sanitized_data(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Get sanitized data from validation results.
        
        Args:
            results: Validation results
            
        Returns:
            Dictionary of sanitized data
        """
        sanitized_data = {}
        
        for field_name, result in results.items():
            if result.valid and result.sanitized_value is not None:
                sanitized_data[field_name] = result.sanitized_value
        
        return sanitized_data
    
    def get_all_errors(self, results: Dict[str, ValidationResult]) -> List[str]:
        """Get all validation errors.
        
        Args:
            results: Validation results
            
        Returns:
            List of all error messages
        """
        all_errors = []
        
        for result in results.values():
            all_errors.extend(result.errors)
        
        return all_errors


class SchemaValidator:
    """Schema-based validator."""
    
    def __init__(self):
        """Initialize schema validator."""
        self.validation_engine = ValidationEngine()
        self.schemas = {}
        self.logger = logging.getLogger("security.schema")
    
    def register_schema(self, schema_name: str, rules: List[ValidationRule]) -> None:
        """Register a validation schema.
        
        Args:
            schema_name: Name of the schema
            rules: List of validation rules
        """
        self.schemas[schema_name] = rules
        self.logger.info(f"Schema registered: {schema_name}")
    
    def validate_against_schema(self, data: Dict[str, Any], schema_name: str) -> Dict[str, ValidationResult]:
        """Validate data against a registered schema.
        
        Args:
            data: Data to validate
            schema_name: Name of the schema
            
        Returns:
            Dictionary of validation results
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        rules = self.schemas[schema_name]
        return self.validation_engine.validate_data(data, rules)
    
    def load_schema_from_file(self, schema_name: str, file_path: str) -> None:
        """Load schema from JSON file.
        
        Args:
            schema_name: Name of the schema
            file_path: Path to JSON schema file
        """
        try:
            with open(file_path, 'r') as f:
                schema_data = json.load(f)
            
            rules = []
            for rule_data in schema_data.get('rules', []):
                rule = ValidationRule(
                    name=rule_data['name'],
                    validation_type=ValidationType(rule_data['validation_type']),
                    required=rule_data.get('required', True),
                    min_length=rule_data.get('min_length'),
                    max_length=rule_data.get('max_length'),
                    min_value=rule_data.get('min_value'),
                    max_value=rule_data.get('max_value'),
                    pattern=rule_data.get('pattern'),
                    allowed_values=rule_data.get('allowed_values'),
                    error_message=rule_data.get('error_message'),
                    sanitization=[SanitizationType(s) for s in rule_data.get('sanitization', [])],
                    metadata=rule_data.get('metadata', {})
                )
                rules.append(rule)
            
            self.register_schema(schema_name, rules)
            
        except Exception as e:
            self.logger.error(f"Failed to load schema from {file_path}: {e}")
            raise
    
    def save_schema_to_file(self, schema_name: str, file_path: str) -> None:
        """Save schema to JSON file.
        
        Args:
            schema_name: Name of the schema
            file_path: Path to save JSON schema file
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        try:
            rules = self.schemas[schema_name]
            schema_data = {
                'name': schema_name,
                'rules': [rule.to_dict() for rule in rules]
            }
            
            with open(file_path, 'w') as f:
                json.dump(schema_data, f, indent=2, default=str)
            
            self.logger.info(f"Schema saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save schema to {file_path}: {e}")
            raise


# Convenience functions for common validations
def create_user_schema() -> List[ValidationRule]:
    """Create a common user validation schema."""
    return [
        ValidationRule(
            name="username",
            validation_type=ValidationType.STRING,
            min_length=3,
            max_length=50,
            pattern=r'^[a-zA-Z0-9_]+$',
            sanitization=[SanitizationType.STRIP_WHITESPACE, SanitizationType.LOWERCASE]
        ),
        ValidationRule(
            name="email",
            validation_type=ValidationType.EMAIL,
            sanitization=[SanitizationType.STRIP_WHITESPACE, SanitizationType.LOWERCASE]
        ),
        ValidationRule(
            name="password",
            validation_type=ValidationType.STRING,
            min_length=8,
            max_length=128,
            pattern=r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]',
            error_message="Password must contain at least one lowercase letter, one uppercase letter, one digit, and one special character"
        ),
        ValidationRule(
            name="age",
            validation_type=ValidationType.INTEGER,
            min_value=13,
            max_value=120,
            required=False
        )
    ]


def create_api_request_schema() -> List[ValidationRule]:
    """Create a common API request validation schema."""
    return [
        ValidationRule(
            name="api_key",
            validation_type=ValidationType.STRING,
            min_length=32,
            max_length=64,
            pattern=r'^[a-zA-Z0-9]+$',
            sanitization=[SanitizationType.STRIP_WHITESPACE]
        ),
        ValidationRule(
            name="timestamp",
            validation_type=ValidationType.DATETIME
        ),
        ValidationRule(
            name="data",
            validation_type=ValidationType.JSON,
            required=False
        ),
        ValidationRule(
            name="client_ip",
            validation_type=ValidationType.IP_ADDRESS,
            required=False
        )
    ]


# Global validation engine instance
_validation_engine = None
_schema_validator = None


def get_validation_engine() -> ValidationEngine:
    """Get global validation engine instance."""
    global _validation_engine
    if _validation_engine is None:
        _validation_engine = ValidationEngine()
    return _validation_engine


def get_schema_validator() -> SchemaValidator:
    """Get global schema validator instance."""
    global _schema_validator
    if _schema_validator is None:
        _schema_validator = SchemaValidator()
        
        # Register common schemas
        _schema_validator.register_schema("user", create_user_schema())
        _schema_validator.register_schema("api_request", create_api_request_schema())
    
    return _schema_validator


def validate_input(data: Dict[str, Any], schema_name: str) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Convenience function to validate input data.
    
    Args:
        data: Data to validate
        schema_name: Name of the schema
        
    Returns:
        Tuple of (is_valid, sanitized_data, errors)
    """
    schema_validator = get_schema_validator()
    validation_engine = get_validation_engine()
    
    results = schema_validator.validate_against_schema(data, schema_name)
    
    is_valid = validation_engine.is_valid(results)
    sanitized_data = validation_engine.get_sanitized_data(results)
    errors = validation_engine.get_all_errors(results)
    
    return is_valid, sanitized_data, errors