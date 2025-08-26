"""Configuration Loader for AI Bull Ford Framework.

Handles loading configuration from various sources and formats.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfigSource:
    """Configuration source definition."""
    path: str
    format: str  # yaml, json, env
    required: bool = True
    priority: int = 0  # Higher priority overrides lower


class ConfigLoader:
    """Loads configuration from multiple sources with priority handling."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            base_path: Base directory for relative config paths
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.sources: List[ConfigSource] = []
        self.loaded_config: Dict[str, Any] = {}
        self.environment_prefix = "AIBF_"
    
    def add_source(self, path: str, format: str = "auto", 
                   required: bool = True, priority: int = 0):
        """Add a configuration source.
        
        Args:
            path: Path to configuration file
            format: Configuration format (yaml, json, env, auto)
            required: Whether this source is required
            priority: Priority level (higher overrides lower)
        """
        if format == "auto":
            format = self._detect_format(path)
        
        source = ConfigSource(
            path=path,
            format=format,
            required=required,
            priority=priority
        )
        
        self.sources.append(source)
        logger.debug(f"Added config source: {path} (format: {format}, priority: {priority})")
    
    def _detect_format(self, path: str) -> str:
        """Auto-detect configuration format from file extension.
        
        Args:
            path: File path
            
        Returns:
            Detected format
        """
        ext = Path(path).suffix.lower()
        
        format_map = {
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.env': 'env'
        }
        
        return format_map.get(ext, 'yaml')
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from all sources.
        
        Returns:
            Merged configuration dictionary
        """
        # Sort sources by priority (ascending)
        sorted_sources = sorted(self.sources, key=lambda s: s.priority)
        
        config = {}
        
        # Load from each source in priority order
        for source in sorted_sources:
            try:
                source_config = self._load_source(source)
                if source_config:
                    config = self._merge_configs(config, source_config)
                    logger.debug(f"Loaded config from {source.path}")
                    
            except Exception as e:
                if source.required:
                    logger.error(f"Failed to load required config from {source.path}: {e}")
                    raise
                else:
                    logger.warning(f"Failed to load optional config from {source.path}: {e}")
        
        # Load environment variables
        env_config = self._load_environment()
        if env_config:
            config = self._merge_configs(config, env_config)
            logger.debug("Loaded environment configuration")
        
        self.loaded_config = config
        return config
    
    def _load_source(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """Load configuration from a single source.
        
        Args:
            source: Configuration source
            
        Returns:
            Configuration dictionary or None
        """
        file_path = self._resolve_path(source.path)
        
        if not file_path.exists():
            if source.required:
                raise FileNotFoundError(f"Required config file not found: {file_path}")
            return None
        
        if source.format == 'yaml':
            return self._load_yaml(file_path)
        elif source.format == 'json':
            return self._load_json(file_path)
        elif source.format == 'env':
            return self._load_env_file(file_path)
        else:
            raise ValueError(f"Unsupported config format: {source.format}")
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve configuration file path.
        
        Args:
            path: File path (absolute or relative)
            
        Returns:
            Resolved Path object
        """
        path_obj = Path(path)
        
        if path_obj.is_absolute():
            return path_obj
        
        return self.base_path / path_obj
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    def _load_env_file(self, file_path: Path) -> Dict[str, Any]:
        """Load environment file (.env format).
        
        Args:
            file_path: Path to .env file
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        # Convert to nested dict using dot notation
                        self._set_nested_value(config, key, value)
                    else:
                        logger.warning(f"Invalid line {line_num} in {file_path}: {line}")
            
            return config
            
        except Exception as e:
            raise ValueError(f"Error reading env file {file_path}: {e}")
    
    def _load_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Returns:
            Configuration dictionary from environment
        """
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.environment_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.environment_prefix):].lower()
                
                # Convert to nested dict using underscore notation
                nested_key = config_key.replace('_', '.')
                self._set_nested_value(config, nested_key, value)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: str):
        """Set nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                # Convert to dict if not already
                current[k] = {}
            current = current[k]
        
        # Set the final value with type conversion
        final_key = keys[-1]
        current[final_key] = self._convert_value(value)
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type.
        
        Args:
            value: String value
            
        Returns:
            Converted value
        """
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Handle null/none values
        if value.lower() in ('null', 'none', ''):
            return None
        
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Handle JSON-like values
        if value.startswith(('[', '{')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value
    
    def _merge_configs(self, base: Dict[str, Any], 
                      override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override the value
                result[key] = value
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """Get the loaded configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.loaded_config.copy()
    
    def reload(self) -> Dict[str, Any]:
        """Reload configuration from all sources.
        
        Returns:
            Reloaded configuration dictionary
        """
        logger.info("Reloading configuration")
        return self.load()
    
    def save_config(self, config: Dict[str, Any], path: str, 
                   format: str = "yaml") -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            path: Output file path
            format: Output format (yaml, json)
        """
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'yaml':
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        elif format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"Configuration saved to {file_path}")
    
    def set_environment_prefix(self, prefix: str):
        """Set the environment variable prefix.
        
        Args:
            prefix: Environment variable prefix
        """
        self.environment_prefix = prefix
        logger.debug(f"Environment prefix set to: {prefix}")
    
    def clear_sources(self):
        """Clear all configuration sources."""
        self.sources.clear()
        self.loaded_config.clear()
        logger.debug("Configuration sources cleared")
    
    def list_sources(self) -> List[Dict[str, Any]]:
        """List all configuration sources.
        
        Returns:
            List of source information
        """
        return [
            {
                'path': source.path,
                'format': source.format,
                'required': source.required,
                'priority': source.priority,
                'exists': self._resolve_path(source.path).exists()
            }
            for source in self.sources
        ]


def create_default_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """Create a ConfigLoader with default sources.
    
    Args:
        config_dir: Configuration directory path
        
    Returns:
        Configured ConfigLoader instance
    """
    if config_dir is None:
        config_dir = os.path.join(os.getcwd(), 'config')
    
    loader = ConfigLoader(config_dir)
    
    # Add default configuration sources in priority order
    loader.add_source('default.yaml', required=True, priority=0)
    loader.add_source('local.yaml', required=False, priority=10)
    loader.add_source('.env', required=False, priority=20)
    
    # Environment-specific configs
    env = os.getenv('AIBF_ENV', 'development')
    loader.add_source(f'{env}.yaml', required=False, priority=5)
    
    return loader