"""Configuration Manager for AI Bull Ford Framework.

Provides centralized configuration management with hot-reloading and caching.
"""

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .loader import ConfigLoader, create_default_loader
from .validator import ConfigValidator, ValidationResult, create_default_validator

logger = logging.getLogger(__name__)


@dataclass
class ConfigChangeEvent:
    """Configuration change event."""
    path: str
    old_value: Any
    new_value: Any
    timestamp: float


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration files."""
    
    def __init__(self, manager: 'ConfigManager'):
        """Initialize file handler.
        
        Args:
            manager: ConfigManager instance
        """
        self.manager = manager
        self.last_reload = 0
        self.reload_delay = 1.0  # Minimum delay between reloads
    
    def on_modified(self, event):
        """Handle file modification events.
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
        
        # Check if it's a config file
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in ['.yaml', '.yml', '.json', '.env']:
            current_time = time.time()
            
            # Debounce rapid file changes
            if current_time - self.last_reload > self.reload_delay:
                logger.info(f"Configuration file changed: {file_path}")
                self.manager._reload_config()
                self.last_reload = current_time


class ConfigManager:
    """Manages configuration loading, validation, and hot-reloading."""
    
    def __init__(self, config_dir: Optional[str] = None, 
                 auto_reload: bool = True):
        """Initialize configuration manager.
        
        Args:
            config_dir: Configuration directory path
            auto_reload: Enable automatic configuration reloading
        """
        self.config_dir = config_dir or os.path.join(os.getcwd(), 'config')
        self.auto_reload = auto_reload
        
        # Core components
        self.loader = create_default_loader(self.config_dir)
        self.validator = create_default_validator()
        
        # Configuration state
        self.config: Dict[str, Any] = {}
        self.last_loaded: Optional[float] = None
        self.validation_result: Optional[ValidationResult] = None
        
        # Change tracking
        self.change_callbacks: List[Callable[[ConfigChangeEvent], None]] = []
        self._lock = threading.RLock()
        
        # File watching
        self.observer: Optional[Observer] = None
        self.file_handler: Optional[ConfigFileHandler] = None
        
        # Initialize
        self._setup_file_watching()
        self.load_config()
    
    def _setup_file_watching(self):
        """Setup file system watching for configuration changes."""
        if not self.auto_reload:
            return
        
        try:
            self.observer = Observer()
            self.file_handler = ConfigFileHandler(self)
            
            # Watch the config directory
            if os.path.exists(self.config_dir):
                self.observer.schedule(
                    self.file_handler, 
                    self.config_dir, 
                    recursive=False
                )
                self.observer.start()
                logger.info(f"Started watching config directory: {self.config_dir}")
            else:
                logger.warning(f"Config directory does not exist: {self.config_dir}")
                
        except Exception as e:
            logger.error(f"Failed to setup file watching: {e}")
            self.auto_reload = False
    
    def load_config(self, validate: bool = True) -> Dict[str, Any]:
        """Load configuration from all sources.
        
        Args:
            validate: Whether to validate the configuration
            
        Returns:
            Loaded configuration dictionary
        """
        with self._lock:
            try:
                logger.info("Loading configuration")
                
                # Load configuration
                old_config = self.config.copy()
                self.config = self.loader.load()
                self.last_loaded = time.time()
                
                # Validate if requested
                if validate:
                    self.validation_result = self.validator.validate(self.config)
                    
                    if self.validation_result.has_errors():
                        logger.error("Configuration validation failed:")
                        for error in self.validation_result.errors:
                            logger.error(f"  - {error}")
                        raise ValueError("Configuration validation failed")
                    
                    if self.validation_result.has_warnings():
                        logger.warning("Configuration validation warnings:")
                        for warning in self.validation_result.warnings:
                            logger.warning(f"  - {warning}")
                
                # Notify change callbacks
                self._notify_changes(old_config, self.config)
                
                logger.info("Configuration loaded successfully")
                return self.config.copy()
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise
    
    def _reload_config(self):
        """Internal method to reload configuration."""
        try:
            self.load_config()
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def _notify_changes(self, old_config: Dict[str, Any], 
                       new_config: Dict[str, Any]):
        """Notify callbacks about configuration changes.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        changes = self._find_changes(old_config, new_config)
        
        for change in changes:
            for callback in self.change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Error in change callback: {e}")
    
    def _find_changes(self, old_config: Dict[str, Any], 
                     new_config: Dict[str, Any], 
                     path: str = "") -> List[ConfigChangeEvent]:
        """Find changes between two configuration dictionaries.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
            path: Current path in the configuration tree
            
        Returns:
            List of configuration changes
        """
        changes = []
        current_time = time.time()
        
        # Find all keys in both configs
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if old_value != new_value:
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    # Recursively check nested dictionaries
                    changes.extend(
                        self._find_changes(old_value, new_value, current_path)
                    )
                else:
                    # Value changed
                    changes.append(ConfigChangeEvent(
                        path=current_path,
                        old_value=old_value,
                        new_value=new_value,
                        timestamp=current_time
                    ))
        
        return changes
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            Configuration dictionary
        """
        with self._lock:
            return self.config.copy()
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by path.
        
        Args:
            path: Dot-separated configuration path
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        with self._lock:
            return self._get_nested_value(self.config, path, default)
    
    def set(self, path: str, value: Any, validate: bool = True):
        """Set a configuration value by path.
        
        Args:
            path: Dot-separated configuration path
            value: Value to set
            validate: Whether to validate after setting
        """
        with self._lock:
            old_config = self.config.copy()
            self._set_nested_value(self.config, path, value)
            
            if validate:
                validation_result = self.validator.validate(self.config)
                if validation_result.has_errors():
                    # Revert changes
                    self.config = old_config
                    raise ValueError(f"Configuration validation failed: {validation_result.errors}")
            
            # Notify changes
            self._notify_changes(old_config, self.config)
    
    def _get_nested_value(self, config: Dict[str, Any], path: str, 
                         default: Any = None) -> Any:
        """Get nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path
            default: Default value
            
        Returns:
            Configuration value
        """
        keys = path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        
        return current
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path
            value: Value to set
        """
        keys = path.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def add_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """Add a callback for configuration changes.
        
        Args:
            callback: Callback function
        """
        self.change_callbacks.append(callback)
        logger.debug(f"Added configuration change callback: {callback.__name__}")
    
    def remove_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """Remove a configuration change callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
            logger.debug(f"Removed configuration change callback: {callback.__name__}")
    
    def validate_config(self) -> ValidationResult:
        """Validate the current configuration.
        
        Returns:
            ValidationResult
        """
        with self._lock:
            self.validation_result = self.validator.validate(self.config)
            return self.validation_result
    
    def get_validation_result(self) -> Optional[ValidationResult]:
        """Get the last validation result.
        
        Returns:
            ValidationResult or None
        """
        return self.validation_result
    
    def save_config(self, path: Optional[str] = None, format: str = "yaml"):
        """Save current configuration to file.
        
        Args:
            path: Output file path (default: config/current.yaml)
            format: Output format (yaml, json)
        """
        if path is None:
            path = os.path.join(self.config_dir, f"current.{format}")
        
        with self._lock:
            self.loader.save_config(self.config, path, format)
    
    def reload(self) -> Dict[str, Any]:
        """Manually reload configuration.
        
        Returns:
            Reloaded configuration
        """
        logger.info("Manually reloading configuration")
        return self.load_config()
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration manager status.
        
        Returns:
            Status information
        """
        with self._lock:
            status = {
                'config_dir': self.config_dir,
                'auto_reload': self.auto_reload,
                'last_loaded': self.last_loaded,
                'sources': self.loader.list_sources(),
                'validation_rules': self.validator.get_rule_count(),
                'change_callbacks': len(self.change_callbacks)
            }
            
            if self.validation_result:
                status['validation'] = {
                    'valid': self.validation_result.valid,
                    'errors': len(self.validation_result.errors),
                    'warnings': len(self.validation_result.warnings),
                    'info': len(self.validation_result.info)
                }
            
            return status
    
    def export_config(self, include_defaults: bool = True, 
                     include_sensitive: bool = False) -> Dict[str, Any]:
        """Export configuration for external use.
        
        Args:
            include_defaults: Include default values
            include_sensitive: Include sensitive values (passwords, keys)
            
        Returns:
            Exported configuration
        """
        with self._lock:
            config = self.config.copy()
            
            if not include_sensitive:
                config = self._remove_sensitive_data(config)
            
            return config
    
    def _remove_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with sensitive data removed
        """
        sensitive_keys = {
            'password', 'secret', 'key', 'token', 'credential',
            'private_key', 'secret_key', 'api_key', 'auth_token'
        }
        
        def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            cleaned = {}
            for key, value in d.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    cleaned[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    cleaned[key] = clean_dict(value)
                else:
                    cleaned[key] = value
            return cleaned
        
        return clean_dict(config)
    
    def shutdown(self):
        """Shutdown the configuration manager."""
        logger.info("Shutting down configuration manager")
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        self.change_callbacks.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global configuration manager instance
_global_manager: Optional[ConfigManager] = None


def get_global_manager() -> ConfigManager:
    """Get the global configuration manager instance.
    
    Returns:
        Global ConfigManager instance
    """
    global _global_manager
    
    if _global_manager is None:
        _global_manager = ConfigManager()
    
    return _global_manager


def initialize_global_manager(config_dir: Optional[str] = None, 
                            auto_reload: bool = True) -> ConfigManager:
    """Initialize the global configuration manager.
    
    Args:
        config_dir: Configuration directory path
        auto_reload: Enable automatic configuration reloading
        
    Returns:
        Initialized ConfigManager instance
    """
    global _global_manager
    
    if _global_manager:
        _global_manager.shutdown()
    
    _global_manager = ConfigManager(config_dir, auto_reload)
    return _global_manager


def get_config(path: Optional[str] = None, default: Any = None) -> Any:
    """Get configuration value from global manager.
    
    Args:
        path: Configuration path (None for entire config)
        default: Default value
        
    Returns:
        Configuration value
    """
    manager = get_global_manager()
    
    if path is None:
        return manager.get_config()
    
    return manager.get(path, default)


def set_config(path: str, value: Any, validate: bool = True):
    """Set configuration value in global manager.
    
    Args:
        path: Configuration path
        value: Value to set
        validate: Whether to validate
    """
    manager = get_global_manager()
    manager.set(path, value, validate)