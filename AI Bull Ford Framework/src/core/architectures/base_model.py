"""Base model class for all AI models in the framework.

Provides common interface and functionality for all model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all AI models.
    
    Provides common interface for model lifecycle management,
    configuration handling, and standardized operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the base model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.version = config.get("version", "1.0.0")
        self.is_trained = False
        self.metadata = {}
        
    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Forward pass through the model.
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def validate_step(self, batch: Any) -> Dict[str, float]:
        """Single validation step.
        
        Args:
            batch: Validation batch
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    def save_model(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save the model
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = save_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Save metadata
        metadata_path = save_path / "metadata.json"
        metadata = {
            "name": self.name,
            "version": self.version,
            "is_trained": self.is_trained,
            **self.metadata
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save model weights (to be implemented by subclasses)
        self._save_weights(save_path)
    
    def load_model(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        load_path = Path(path)
        
        # Load configuration
        config_path = load_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
        
        # Load metadata
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.name = metadata.get("name", self.name)
                self.version = metadata.get("version", self.version)
                self.is_trained = metadata.get("is_trained", False)
                self.metadata.update(metadata)
        
        # Load model weights (to be implemented by subclasses)
        self._load_weights(load_path)
    
    @abstractmethod
    def _save_weights(self, path: Path) -> None:
        """Save model weights to disk.
        
        Args:
            path: Directory path to save weights
        """
        pass
    
    @abstractmethod
    def _load_weights(self, path: Path) -> None:
        """Load model weights from disk.
        
        Args:
            path: Directory path to load weights from
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.name,
            "version": self.version,
            "is_trained": self.is_trained,
            "config": self.config,
            "metadata": self.metadata
        }
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value
        """
        return self.metadata.get(key, default)
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name} v{self.version} (trained: {self.is_trained})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}', trained={self.is_trained})"