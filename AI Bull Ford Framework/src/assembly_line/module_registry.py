"""Module registry for AI component management.

Provides centralized registration, discovery, and lifecycle management
for AI modules, models, and components.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable, Type, Union, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
import importlib
import pkgutil
import logging
from pathlib import Path
import json
import threading
from datetime import datetime


class ModuleType(Enum):
    """Types of modules that can be registered."""
    MODEL = "model"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    OPTIMIZER = "optimizer"
    LOSS_FUNCTION = "loss_function"
    METRIC = "metric"
    TRANSFORMER = "transformer"
    PIPELINE_STAGE = "pipeline_stage"
    AGENT = "agent"
    TOOL = "tool"
    CUSTOM = "custom"


class ModuleStatus(Enum):
    """Status of registered modules."""
    REGISTERED = "registered"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


@dataclass
class ModuleMetadata:
    """Metadata for registered modules."""
    name: str
    module_type: ModuleType
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: ModuleStatus = ModuleStatus.REGISTERED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'module_type': self.module_type.value,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'config_schema': self.config_schema,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleMetadata':
        """Create metadata from dictionary."""
        return cls(
            name=data['name'],
            module_type=ModuleType(data['module_type']),
            version=data['version'],
            description=data['description'],
            author=data['author'],
            dependencies=data.get('dependencies', []),
            tags=data.get('tags', []),
            config_schema=data.get('config_schema'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            status=ModuleStatus(data['status'])
        )


@dataclass
class RegisteredModule:
    """Container for registered modules."""
    metadata: ModuleMetadata
    module_class: Type
    instance: Optional[Any] = None
    config: Optional[Dict[str, Any]] = None
    load_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def create_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create new instance of the module.
        
        Args:
            config: Configuration for the module
            
        Returns:
            Module instance
        """
        try:
            if config:
                instance = self.module_class(**config)
            else:
                instance = self.module_class()
            
            self.load_count += 1
            self.last_accessed = datetime.now()
            
            return instance
            
        except Exception as e:
            self.metadata.status = ModuleStatus.ERROR
            raise RuntimeError(f"Failed to create instance of {self.metadata.name}: {e}")
    
    def get_or_create_instance(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get existing instance or create new one.
        
        Args:
            config: Configuration for the module
            
        Returns:
            Module instance
        """
        if self.instance is None:
            self.instance = self.create_instance(config)
            self.config = config
        
        self.last_accessed = datetime.now()
        return self.instance


class ModuleRegistry:
    """Central registry for AI modules and components."""
    
    def __init__(self, name: str = "default"):
        """Initialize module registry.
        
        Args:
            name: Registry name
        """
        self.name = name
        self._modules: Dict[str, RegisteredModule] = {}
        self._type_index: Dict[ModuleType, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"registry.{name}")
        
        # Initialize type index
        for module_type in ModuleType:
            self._type_index[module_type] = set()
    
    def register(self, 
                module_class: Type,
                name: Optional[str] = None,
                module_type: Optional[ModuleType] = None,
                version: str = "1.0.0",
                description: str = "",
                author: str = "",
                dependencies: Optional[List[str]] = None,
                tags: Optional[List[str]] = None,
                config_schema: Optional[Dict[str, Any]] = None,
                force: bool = False) -> bool:
        """Register a module in the registry.
        
        Args:
            module_class: Class to register
            name: Module name (defaults to class name)
            module_type: Type of module
            version: Module version
            description: Module description
            author: Module author
            dependencies: List of dependencies
            tags: List of tags
            config_schema: Configuration schema
            force: Force registration even if module exists
            
        Returns:
            True if registration successful
        """
        with self._lock:
            # Default name to class name
            if name is None:
                name = module_class.__name__
            
            # Auto-detect module type if not provided
            if module_type is None:
                module_type = self._detect_module_type(module_class)
            
            # Check if module already exists
            if name in self._modules and not force:
                self.logger.warning(f"Module {name} already registered")
                return False
            
            # Create metadata
            metadata = ModuleMetadata(
                name=name,
                module_type=module_type,
                version=version,
                description=description or module_class.__doc__ or "",
                author=author,
                dependencies=dependencies or [],
                tags=tags or [],
                config_schema=config_schema
            )
            
            # Create registered module
            registered_module = RegisteredModule(
                metadata=metadata,
                module_class=module_class
            )
            
            # Store module
            self._modules[name] = registered_module
            
            # Update indices
            self._type_index[module_type].add(name)
            for tag in metadata.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(name)
            
            self.logger.info(f"Registered module: {name} ({module_type.value})")
            return True
    
    def unregister(self, name: str) -> bool:
        """Unregister a module.
        
        Args:
            name: Module name
            
        Returns:
            True if unregistration successful
        """
        with self._lock:
            if name not in self._modules:
                return False
            
            module = self._modules[name]
            
            # Remove from type index
            self._type_index[module.metadata.module_type].discard(name)
            
            # Remove from tag index
            for tag in module.metadata.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(name)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]
            
            # Remove module
            del self._modules[name]
            
            self.logger.info(f"Unregistered module: {name}")
            return True
    
    def get(self, name: str) -> Optional[RegisteredModule]:
        """Get a registered module.
        
        Args:
            name: Module name
            
        Returns:
            Registered module if found
        """
        return self._modules.get(name)
    
    def create_instance(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create instance of a registered module.
        
        Args:
            name: Module name
            config: Configuration for the module
            
        Returns:
            Module instance
            
        Raises:
            KeyError: If module not found
        """
        module = self.get(name)
        if module is None:
            raise KeyError(f"Module {name} not found in registry")
        
        return module.create_instance(config)
    
    def get_instance(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get or create singleton instance of a module.
        
        Args:
            name: Module name
            config: Configuration for the module
            
        Returns:
            Module instance
            
        Raises:
            KeyError: If module not found
        """
        module = self.get(name)
        if module is None:
            raise KeyError(f"Module {name} not found in registry")
        
        return module.get_or_create_instance(config)
    
    def list_modules(self, 
                    module_type: Optional[ModuleType] = None,
                    tags: Optional[List[str]] = None,
                    status: Optional[ModuleStatus] = None) -> List[str]:
        """List registered modules with optional filtering.
        
        Args:
            module_type: Filter by module type
            tags: Filter by tags (all must match)
            status: Filter by status
            
        Returns:
            List of module names
        """
        modules = list(self._modules.keys())
        
        # Filter by type
        if module_type is not None:
            modules = [name for name in modules 
                      if self._modules[name].metadata.module_type == module_type]
        
        # Filter by tags
        if tags:
            modules = [name for name in modules 
                      if all(tag in self._modules[name].metadata.tags for tag in tags)]
        
        # Filter by status
        if status is not None:
            modules = [name for name in modules 
                      if self._modules[name].metadata.status == status]
        
        return sorted(modules)
    
    def search(self, query: str) -> List[str]:
        """Search modules by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching module names
        """
        query_lower = query.lower()
        matches = []
        
        for name, module in self._modules.items():
            metadata = module.metadata
            
            # Search in name
            if query_lower in name.lower():
                matches.append(name)
                continue
            
            # Search in description
            if query_lower in metadata.description.lower():
                matches.append(name)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in metadata.tags):
                matches.append(name)
                continue
        
        return sorted(matches)
    
    def get_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a module.
        
        Args:
            name: Module name
            
        Returns:
            List of dependency names
        """
        module = self.get(name)
        if module is None:
            return []
        
        return module.metadata.dependencies.copy()
    
    def validate_dependencies(self, name: str) -> List[str]:
        """Validate that all dependencies are available.
        
        Args:
            name: Module name
            
        Returns:
            List of missing dependencies
        """
        dependencies = self.get_dependencies(name)
        missing = []
        
        for dep in dependencies:
            if dep not in self._modules:
                missing.append(dep)
        
        return missing
    
    def auto_discover(self, package_path: str) -> int:
        """Auto-discover and register modules from a package.
        
        Args:
            package_path: Path to package to scan
            
        Returns:
            Number of modules discovered
        """
        discovered = 0
        
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Walk through all modules in the package
            for importer, modname, ispkg in pkgutil.walk_packages(
                package.__path__, package.__name__ + "."):
                
                try:
                    module = importlib.import_module(modname)
                    
                    # Look for classes that could be registered
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if self._should_auto_register(obj):
                            if self.register(obj, force=False):
                                discovered += 1
                                
                except Exception as e:
                    self.logger.warning(f"Failed to import {modname}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to auto-discover from {package_path}: {e}")
        
        self.logger.info(f"Auto-discovered {discovered} modules from {package_path}")
        return discovered
    
    def export_registry(self, file_path: str) -> None:
        """Export registry to JSON file.
        
        Args:
            file_path: Path to export file
        """
        registry_data = {
            'name': self.name,
            'modules': {name: module.metadata.to_dict() 
                       for name, module in self._modules.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        self.logger.info(f"Exported registry to {file_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_modules': len(self._modules),
            'by_type': {},
            'by_status': {},
            'total_instances': 0,
            'total_loads': 0
        }
        
        # Count by type
        for module_type in ModuleType:
            stats['by_type'][module_type.value] = len(self._type_index[module_type])
        
        # Count by status and other metrics
        for module in self._modules.values():
            status = module.metadata.status.value
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            if module.instance is not None:
                stats['total_instances'] += 1
            
            stats['total_loads'] += module.load_count
        
        return stats
    
    def _detect_module_type(self, module_class: Type) -> ModuleType:
        """Auto-detect module type based on class characteristics.
        
        Args:
            module_class: Class to analyze
            
        Returns:
            Detected module type
        """
        class_name = module_class.__name__.lower()
        
        # Simple heuristics based on class name
        if 'model' in class_name or 'network' in class_name:
            return ModuleType.MODEL
        elif 'preprocess' in class_name:
            return ModuleType.PREPROCESSOR
        elif 'postprocess' in class_name:
            return ModuleType.POSTPROCESSOR
        elif 'optimizer' in class_name:
            return ModuleType.OPTIMIZER
        elif 'loss' in class_name:
            return ModuleType.LOSS_FUNCTION
        elif 'metric' in class_name:
            return ModuleType.METRIC
        elif 'transformer' in class_name:
            return ModuleType.TRANSFORMER
        elif 'stage' in class_name:
            return ModuleType.PIPELINE_STAGE
        elif 'agent' in class_name:
            return ModuleType.AGENT
        elif 'tool' in class_name:
            return ModuleType.TOOL
        else:
            return ModuleType.CUSTOM
    
    def _should_auto_register(self, cls: Type) -> bool:
        """Check if a class should be auto-registered.
        
        Args:
            cls: Class to check
            
        Returns:
            True if class should be registered
        """
        # Skip abstract classes
        if inspect.isabstract(cls):
            return False
        
        # Skip private classes
        if cls.__name__.startswith('_'):
            return False
        
        # Check for registration marker
        if hasattr(cls, '__auto_register__') and not cls.__auto_register__:
            return False
        
        return True
    
    def clear(self) -> None:
        """Clear all registered modules."""
        with self._lock:
            self._modules.clear()
            for module_type in ModuleType:
                self._type_index[module_type].clear()
            self._tag_index.clear()
        
        self.logger.info("Cleared all registered modules")
    
    def __len__(self) -> int:
        """Get number of registered modules."""
        return len(self._modules)
    
    def __contains__(self, name: str) -> bool:
        """Check if module is registered."""
        return name in self._modules
    
    def __iter__(self):
        """Iterate over module names."""
        return iter(self._modules.keys())


# Global registry instance
_global_registry = ModuleRegistry("global")


def register(module_class: Type, **kwargs) -> bool:
    """Register a module in the global registry.
    
    Args:
        module_class: Class to register
        **kwargs: Additional registration parameters
        
    Returns:
        True if registration successful
    """
    return _global_registry.register(module_class, **kwargs)


def get_registry() -> ModuleRegistry:
    """Get the global module registry.
    
    Returns:
        Global registry instance
    """
    return _global_registry


def create_instance(name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Create instance from global registry.
    
    Args:
        name: Module name
        config: Configuration
        
    Returns:
        Module instance
    """
    return _global_registry.create_instance(name, config)


def list_modules(**kwargs) -> List[str]:
    """List modules from global registry.
    
    Args:
        **kwargs: Filter parameters
        
    Returns:
        List of module names
    """
    return _global_registry.list_modules(**kwargs)