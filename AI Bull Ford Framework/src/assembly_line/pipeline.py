"""Pipeline system for modular AI workflows.

Provides flexible pipeline architecture with stages, data flow,
and error handling for AI model training and deployment.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future


class StageStatus(Enum):
    """Status of pipeline stages."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineContext:
    """Context object passed between pipeline stages."""
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    config: Dict[str, Any]
    stage_results: Dict[str, Any]
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.metadata is None:
            self.metadata = {}
        if self.config is None:
            self.config = {}
        if self.stage_results is None:
            self.stage_results = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get data from context."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set data in context."""
        self.data[key] = value
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update context data."""
        self.data.update(data)


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline stage.
        
        Args:
            name: Stage name
            config: Stage configuration
        """
        self.name = name
        self.config = config or {}
        self.status = StageStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error: Optional[Exception] = None
        self.logger = logging.getLogger(f"pipeline.{name}")
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the pipeline stage.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated pipeline context
        """
        pass
    
    def validate_inputs(self, context: PipelineContext) -> bool:
        """Validate stage inputs.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if inputs are valid
        """
        return True
    
    def should_skip(self, context: PipelineContext) -> bool:
        """Check if stage should be skipped.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if stage should be skipped
        """
        return False
    
    def cleanup(self, context: PipelineContext) -> None:
        """Cleanup after stage execution.
        
        Args:
            context: Pipeline context
        """
        pass
    
    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the pipeline stage with error handling.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated pipeline context
        """
        self.start_time = time.time()
        self.status = StageStatus.RUNNING
        
        try:
            self.logger.info(f"Starting stage: {self.name}")
            
            # Check if stage should be skipped
            if self.should_skip(context):
                self.status = StageStatus.SKIPPED
                self.logger.info(f"Skipping stage: {self.name}")
                return context
            
            # Validate inputs
            if not self.validate_inputs(context):
                raise ValueError(f"Input validation failed for stage: {self.name}")
            
            # Execute stage
            result_context = self.execute(context)
            
            # Store stage result
            result_context.stage_results[self.name] = {
                'status': StageStatus.COMPLETED,
                'start_time': self.start_time,
                'end_time': time.time(),
                'duration': time.time() - self.start_time
            }
            
            self.status = StageStatus.COMPLETED
            self.end_time = time.time()
            
            self.logger.info(f"Completed stage: {self.name} in {self.end_time - self.start_time:.2f}s")
            
            return result_context
            
        except Exception as e:
            self.error = e
            self.status = StageStatus.FAILED
            self.end_time = time.time()
            
            # Store error in stage results
            context.stage_results[self.name] = {
                'status': StageStatus.FAILED,
                'error': str(e),
                'start_time': self.start_time,
                'end_time': self.end_time
            }
            
            self.logger.error(f"Stage {self.name} failed: {e}")
            raise
        
        finally:
            # Always run cleanup
            try:
                self.cleanup(context)
            except Exception as cleanup_error:
                self.logger.warning(f"Cleanup failed for stage {self.name}: {cleanup_error}")
    
    @property
    def duration(self) -> Optional[float]:
        """Get stage execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class DataLoaderStage(PipelineStage):
    """Stage for loading data into the pipeline."""
    
    def __init__(self, name: str = "data_loader", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Load data based on configuration."""
        data_source = self.config.get('source')
        data_type = self.config.get('type', 'numpy')
        
        if data_source:
            if data_type == 'numpy':
                # Load numpy data
                data = np.load(data_source)
                context.set('raw_data', data)
            elif data_type == 'csv':
                # Simulate CSV loading
                context.set('raw_data', f"CSV data from {data_source}")
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
        
        context.metadata['data_loaded'] = True
        context.metadata['data_source'] = data_source
        
        return context


class PreprocessingStage(PipelineStage):
    """Stage for data preprocessing."""
    
    def __init__(self, name: str = "preprocessing", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    def validate_inputs(self, context: PipelineContext) -> bool:
        """Validate that raw data exists."""
        return 'raw_data' in context.data
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Preprocess the data."""
        raw_data = context.get('raw_data')
        
        # Apply preprocessing based on config
        normalize = self.config.get('normalize', False)
        scale = self.config.get('scale', False)
        
        if isinstance(raw_data, np.ndarray):
            processed_data = raw_data.copy()
            
            if normalize:
                processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
            
            if scale:
                processed_data = processed_data / np.max(np.abs(processed_data))
            
            context.set('processed_data', processed_data)
        else:
            # For non-numpy data, just pass through
            context.set('processed_data', raw_data)
        
        context.metadata['preprocessing_applied'] = True
        
        return context


class ModelTrainingStage(PipelineStage):
    """Stage for model training."""
    
    def __init__(self, name: str = "model_training", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    def validate_inputs(self, context: PipelineContext) -> bool:
        """Validate that processed data exists."""
        return 'processed_data' in context.data
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Train the model."""
        processed_data = context.get('processed_data')
        model_type = self.config.get('model_type', 'neural_network')
        epochs = self.config.get('epochs', 10)
        
        # Simulate model training
        self.logger.info(f"Training {model_type} for {epochs} epochs")
        
        # Simulate training progress
        for epoch in range(epochs):
            time.sleep(0.1)  # Simulate training time
            if epoch % 5 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs}")
        
        # Create mock trained model
        trained_model = {
            'type': model_type,
            'epochs': epochs,
            'data_shape': processed_data.shape if isinstance(processed_data, np.ndarray) else None,
            'trained': True
        }
        
        context.set('trained_model', trained_model)
        context.metadata['model_trained'] = True
        
        return context


class ModelEvaluationStage(PipelineStage):
    """Stage for model evaluation."""
    
    def __init__(self, name: str = "model_evaluation", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    def validate_inputs(self, context: PipelineContext) -> bool:
        """Validate that trained model exists."""
        return 'trained_model' in context.data
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Evaluate the model."""
        trained_model = context.get('trained_model')
        metrics = self.config.get('metrics', ['accuracy', 'loss'])
        
        # Simulate model evaluation
        evaluation_results = {}
        for metric in metrics:
            if metric == 'accuracy':
                evaluation_results[metric] = np.random.uniform(0.8, 0.95)
            elif metric == 'loss':
                evaluation_results[metric] = np.random.uniform(0.1, 0.3)
            else:
                evaluation_results[metric] = np.random.uniform(0, 1)
        
        context.set('evaluation_results', evaluation_results)
        context.metadata['model_evaluated'] = True
        
        self.logger.info(f"Evaluation results: {evaluation_results}")
        
        return context


class Pipeline:
    """Main pipeline class for orchestrating stages."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline.
        
        Args:
            name: Pipeline name
            config: Pipeline configuration
        """
        self.name = name
        self.config = config or {}
        self.stages: List[PipelineStage] = []
        self.context: Optional[PipelineContext] = None
        self.logger = logging.getLogger(f"pipeline.{name}")
        
        # Pipeline execution settings
        self.parallel_execution = self.config.get('parallel_execution', False)
        self.max_workers = self.config.get('max_workers', 4)
        self.stop_on_failure = self.config.get('stop_on_failure', True)
    
    def add_stage(self, stage: PipelineStage) -> 'Pipeline':
        """Add a stage to the pipeline.
        
        Args:
            stage: Pipeline stage to add
            
        Returns:
            Self for method chaining
        """
        self.stages.append(stage)
        return self
    
    def remove_stage(self, stage_name: str) -> bool:
        """Remove a stage from the pipeline.
        
        Args:
            stage_name: Name of stage to remove
            
        Returns:
            True if stage was removed
        """
        for i, stage in enumerate(self.stages):
            if stage.name == stage_name:
                del self.stages[i]
                return True
        return False
    
    def get_stage(self, stage_name: str) -> Optional[PipelineStage]:
        """Get a stage by name.
        
        Args:
            stage_name: Name of stage to get
            
        Returns:
            Pipeline stage if found
        """
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def run(self, initial_data: Optional[Dict[str, Any]] = None) -> PipelineContext:
        """Run the pipeline.
        
        Args:
            initial_data: Initial data for the pipeline
            
        Returns:
            Final pipeline context
        """
        # Initialize context
        self.context = PipelineContext(
            data=initial_data or {},
            metadata={'pipeline_name': self.name, 'start_time': time.time()},
            config=self.config,
            stage_results={}
        )
        
        self.logger.info(f"Starting pipeline: {self.name} with {len(self.stages)} stages")
        
        try:
            if self.parallel_execution:
                self._run_parallel()
            else:
                self._run_sequential()
            
            self.context.metadata['end_time'] = time.time()
            self.context.metadata['total_duration'] = (
                self.context.metadata['end_time'] - self.context.metadata['start_time']
            )
            
            self.logger.info(f"Pipeline {self.name} completed successfully")
            
        except Exception as e:
            self.context.metadata['error'] = str(e)
            self.context.metadata['end_time'] = time.time()
            self.logger.error(f"Pipeline {self.name} failed: {e}")
            raise
        
        return self.context
    
    def _run_sequential(self) -> None:
        """Run stages sequentially."""
        for stage in self.stages:
            try:
                self.context = stage.run(self.context)
            except Exception as e:
                if self.stop_on_failure:
                    raise
                else:
                    self.logger.warning(f"Stage {stage.name} failed but continuing: {e}")
    
    def _run_parallel(self) -> None:
        """Run stages in parallel (where possible)."""
        # For simplicity, this implementation runs stages sequentially
        # In a real implementation, you would analyze dependencies
        # and run independent stages in parallel
        self._run_sequential()
    
    def get_results(self) -> Dict[str, Any]:
        """Get pipeline execution results.
        
        Returns:
            Dictionary of results
        """
        if not self.context:
            return {}
        
        return {
            'data': self.context.data,
            'metadata': self.context.metadata,
            'stage_results': self.context.stage_results
        }
    
    def get_stage_status(self) -> Dict[str, StageStatus]:
        """Get status of all stages.
        
        Returns:
            Dictionary mapping stage names to their status
        """
        return {stage.name: stage.status for stage in self.stages}
    
    def reset(self) -> None:
        """Reset pipeline state."""
        for stage in self.stages:
            stage.status = StageStatus.PENDING
            stage.start_time = None
            stage.end_time = None
            stage.error = None
        
        self.context = None
    
    def validate(self) -> List[str]:
        """Validate pipeline configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        if not self.stages:
            errors.append("Pipeline has no stages")
        
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            errors.append("Duplicate stage names found")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation.
        
        Returns:
            Pipeline dictionary
        """
        return {
            'name': self.name,
            'config': self.config,
            'stages': [{
                'name': stage.name,
                'type': stage.__class__.__name__,
                'config': stage.config,
                'status': stage.status.value
            } for stage in self.stages]
        }
    
    @classmethod
    def from_dict(cls, pipeline_dict: Dict[str, Any]) -> 'Pipeline':
        """Create pipeline from dictionary representation.
        
        Args:
            pipeline_dict: Pipeline dictionary
            
        Returns:
            Pipeline instance
        """
        pipeline = cls(pipeline_dict['name'], pipeline_dict.get('config'))
        
        # Note: This is a simplified implementation
        # In practice, you would need a stage factory to create stages from config
        
        return pipeline