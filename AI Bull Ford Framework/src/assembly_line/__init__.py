"""Assembly Line System for AI Bull Ford.

Provides modular pipeline architecture for AI model training,
deployment, and orchestration with dynamic workflow management.
"""

from .pipeline import Pipeline, PipelineStage
from .module_registry import ModuleRegistry
from .workflow import WorkflowManager, WorkflowDefinition
from .orchestrator import Orchestrator, OrchestratorConfig, get_orchestrator, initialize_orchestrator

__version__ = "1.0.0"
__author__ = "AI Bull Ford Team"

__all__ = [
    'Pipeline',
    'PipelineStage',
    'ModuleRegistry',
    'WorkflowManager',
    'WorkflowDefinition',
    'Orchestrator',
    'OrchestratorConfig',
    'get_orchestrator',
    'initialize_orchestrator'
]