"""Specialized applications module for AI Bull Ford.

This module provides domain-specific AI applications including:
- Robotics applications for autonomous systems
- Industrial automation and process optimization
- Edge AI for resource-constrained environments
- Healthcare AI applications
- Financial AI systems
- Educational AI tools
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import all application modules
from .robotics import (
    RobotType, SensorType, ActuatorType, NavigationMode,
    RobotConfig, SensorData, ActuatorCommand, NavigationState, MotionPlan, TaskResult,
    SensorManager, ActuatorController, PathPlanner, NavigationSystem, TaskExecutor, RobotController,
    initialize_robotics, shutdown_robotics
)
from .industrial import (
    ProcessType, EquipmentStatus, QualityStatus, MaintenanceType,
    IndustrialConfig, ProductionData, QualityInspection, MaintenanceRecord, EquipmentHealth, ProductionOrder,
    ProcessController, QualityController, MaintenanceManager, ProductionPlanner, IndustrialAutomationSystem,
    initialize_industrial, shutdown_industrial
)
from .edge_ai import (
    DeviceType, ModelFormat, ProcessingMode, SyncStrategy,
    EdgeConfig, DeviceSpecs, ModelInfo, InferenceRequest, InferenceResult, ResourceUsage,
    ModelOptimizer, EdgeInferenceEngine, ResourceMonitor, EdgeCloudSync, EdgeAISystem,
    initialize_edge_ai, shutdown_edge_ai
)
from .healthcare import (
    MedicalImageType, DiagnosisType, PatientStatus, TreatmentType,
    HealthcareConfig, PatientInfo, MedicalImage, DiagnosisResult, TreatmentPlan, VitalSigns, LabResult,
    MedicalImageAnalyzer, ClinicalDecisionSupport, PatientMonitor, HealthcareAnalytics, HealthcareAISystem,
    initialize_healthcare, shutdown_healthcare
)
from .financial import (
    AssetType, MarketSentiment, RiskLevel, TransactionType, FraudRiskLevel,
    FinancialConfig, MarketData, Transaction, Portfolio, TradingSignal, RiskAssessment, FraudAlert,
    MarketAnalyzer, RiskManager, FraudDetector, PortfolioOptimizer, FinancialAISystem,
    initialize_financial, shutdown_financial
)
from .educational import (
    LearningStyle, DifficultyLevel, ContentType, AssessmentType, LearningObjective, EngagementLevel,
    EducationalConfig, LearnerProfile, LearningContent, LearningSession, Assessment, AssessmentResult, LearningPath,
    ContentRecommendationEngine, AdaptiveLearningEngine, IntelligentTutorSystem, LearningAnalytics, EducationalAISystem,
    initialize_educational, shutdown_educational
)


__all__ = [
    # Robotics components
    "RobotType", "SensorType", "ActuatorType", "NavigationMode",
    "RobotConfig", "SensorData", "ActuatorCommand", "NavigationState",
    "MotionPlan", "TaskResult",
    "SensorManager", "ActuatorController", "PathPlanner", "NavigationSystem", "TaskExecutor", "RobotController",
    "initialize_robotics", "shutdown_robotics",
    
    # Industrial components
    "ProcessType", "EquipmentStatus", "QualityStatus", "MaintenanceType",
    "IndustrialConfig", "ProductionData", "QualityInspection", "MaintenanceRecord", "EquipmentHealth", "ProductionOrder",
    "ProcessController", "QualityController", "MaintenanceManager", "ProductionPlanner", "IndustrialAutomationSystem",
    "initialize_industrial", "shutdown_industrial",
    
    # Edge AI components
    "DeviceType", "ModelFormat", "ProcessingMode", "SyncStrategy",
    "EdgeConfig", "DeviceSpecs", "ModelInfo", "InferenceRequest", "InferenceResult", "ResourceUsage",
    "ModelOptimizer", "EdgeInferenceEngine", "ResourceMonitor", "EdgeCloudSync", "EdgeAISystem",
    "initialize_edge_ai", "shutdown_edge_ai",
    
    # Healthcare components
    "MedicalImageType", "DiagnosisType", "PatientStatus", "TreatmentType",
    "HealthcareConfig", "PatientInfo", "MedicalImage", "DiagnosisResult", "TreatmentPlan", "VitalSigns", "LabResult",
    "MedicalImageAnalyzer", "ClinicalDecisionSupport", "PatientMonitor", "HealthcareAnalytics", "HealthcareAISystem",
    "initialize_healthcare", "shutdown_healthcare",
    
    # Financial components
    "AssetType", "MarketSentiment", "RiskLevel", "TransactionType", "FraudRiskLevel",
    "FinancialConfig", "MarketData", "Transaction", "Portfolio", "TradingSignal", "RiskAssessment", "FraudAlert",
    "MarketAnalyzer", "RiskManager", "FraudDetector", "PortfolioOptimizer", "FinancialAISystem",
    "initialize_financial", "shutdown_financial",
    
    # Educational components
    "LearningStyle", "DifficultyLevel", "ContentType", "AssessmentType", "LearningObjective", "EngagementLevel",
    "EducationalConfig", "LearnerProfile", "LearningContent", "LearningSession", "Assessment", "AssessmentResult", "LearningPath",
    "ContentRecommendationEngine", "AdaptiveLearningEngine", "IntelligentTutorSystem", "LearningAnalytics", "EducationalAISystem",
    "initialize_educational", "shutdown_educational",
    
    # Global functions
    "initialize_applications", "shutdown_applications"
]


def initialize_applications(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize all application modules.
    
    Args:
        config: Configuration dictionary for applications
        
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        app_config = config or {}
        
        # Initialize robotics
        robotics_config = app_config.get('robotics', {})
        if not initialize_robotics(robotics_config):
             logger.error("Failed to initialize robotics module")
             return False
             
         # Initialize industrial automation
         industrial_config = app_config.get('industrial', {})
         if not initialize_industrial(industrial_config):
             logger.error("Failed to initialize industrial module")
             return False
             
         # Initialize edge AI
         edge_config = app_config.get('edge_ai', {})
         if not initialize_edge_ai(edge_config):
             logger.error("Failed to initialize edge AI module")
             return False
             
         # Initialize healthcare
         healthcare_config = app_config.get('healthcare', {})
         if not initialize_healthcare(healthcare_config):
             logger.error("Failed to initialize healthcare module")
             return False
             
         # Initialize financial
         financial_config = app_config.get('financial', {})
         if not initialize_financial(financial_config):
             logger.error("Failed to initialize financial module")
             return False
             
         # Initialize educational
         educational_config = app_config.get('educational', {})
         if not initialize_educational(educational_config):
             logger.error("Failed to initialize educational module")
             return False
             
         logger.info("All application modules initialized successfully")
        return True
        
    except Exception as e:
         logger.error(f"Error initializing applications: {e}")
         return False


def shutdown_applications() -> bool:
    """
    Shutdown all application modules.
    
    Returns:
        bool: True if shutdown successful, False otherwise
    """
    success = True
    
    try:
        # Shutdown components in reverse order
        if not shutdown_educational():
            logger.error("Failed to shutdown educational module")
            success = False
            
        if not shutdown_financial():
            logger.error("Failed to shutdown financial module")
            success = False
            
        if not shutdown_healthcare():
            logger.error("Failed to shutdown healthcare module")
            success = False
            
        if not shutdown_edge_ai():
            logger.error("Failed to shutdown edge AI module")
            success = False
            
        if not shutdown_industrial():
            logger.error("Failed to shutdown industrial module")
            success = False
            
        if not shutdown_robotics():
            logger.error("Failed to shutdown robotics module")
            success = False
            
        if success:
            logger.info("All application modules shutdown successfully")
        else:
            logger.warning("Some application modules failed to shutdown properly")
            
        return success
        
    except Exception as e:
        logger.error(f"Error shutting down applications: {e}")
        return False