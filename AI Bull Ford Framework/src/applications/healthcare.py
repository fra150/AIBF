"""Healthcare AI module for AI Bull Ford.

This module provides comprehensive healthcare AI capabilities including:
- Medical image analysis and diagnostics
- Electronic health record (EHR) processing
- Drug discovery and development
- Clinical decision support
- Patient monitoring and care
- Medical research and analytics
"""

import asyncio
import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class MedicalImageType(Enum):
    """Types of medical images."""
    XRAY = "xray"
    CT_SCAN = "ct_scan"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    MAMMOGRAPHY = "mammography"
    PET_SCAN = "pet_scan"
    ENDOSCOPY = "endoscopy"
    PATHOLOGY = "pathology"
    RETINAL = "retinal"
    DERMATOLOGY = "dermatology"


class DiagnosisType(Enum):
    """Types of medical diagnoses."""
    CANCER = "cancer"
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL = "neurological"
    RESPIRATORY = "respiratory"
    INFECTIOUS = "infectious"
    AUTOIMMUNE = "autoimmune"
    METABOLIC = "metabolic"
    GENETIC = "genetic"
    PSYCHIATRIC = "psychiatric"
    ORTHOPEDIC = "orthopedic"


class PatientStatus(Enum):
    """Patient status levels."""
    STABLE = "stable"
    CRITICAL = "critical"
    IMPROVING = "improving"
    DETERIORATING = "deteriorating"
    DISCHARGED = "discharged"
    ADMITTED = "admitted"
    EMERGENCY = "emergency"


class TreatmentType(Enum):
    """Types of medical treatments."""
    MEDICATION = "medication"
    SURGERY = "surgery"
    THERAPY = "therapy"
    RADIATION = "radiation"
    CHEMOTHERAPY = "chemotherapy"
    REHABILITATION = "rehabilitation"
    PREVENTIVE = "preventive"
    PALLIATIVE = "palliative"


@dataclass
class HealthcareConfig:
    """Configuration for healthcare AI systems."""
    facility_id: str = "hospital_001"
    privacy_mode: bool = True
    hipaa_compliant: bool = True
    data_encryption: bool = True
    audit_logging: bool = True
    real_time_monitoring: bool = True
    ai_assistance_level: str = "advisory"  # advisory, semi_autonomous, autonomous
    confidence_threshold: float = 0.85
    alert_threshold: float = 0.9
    data_retention_years: int = 7
    anonymization_enabled: bool = True
    logging_enabled: bool = True


@dataclass
class PatientInfo:
    """Patient information (anonymized)."""
    patient_id: str
    age: int
    gender: str
    medical_history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    vital_signs: Dict[str, float] = field(default_factory=dict)
    admission_date: Optional[datetime] = None
    status: PatientStatus = PatientStatus.STABLE
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class MedicalImage:
    """Medical image data."""
    image_id: str
    patient_id: str
    image_type: MedicalImageType
    acquisition_date: datetime
    image_data: Any  # Image array or path
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 1.0
    anonymized: bool = True


@dataclass
class DiagnosisResult:
    """Medical diagnosis result."""
    diagnosis_id: str
    patient_id: str
    diagnosis_type: DiagnosisType
    condition: str
    confidence: float
    severity: str  # mild, moderate, severe, critical
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    reviewed_by: Optional[str] = None
    ai_generated: bool = True


@dataclass
class TreatmentPlan:
    """Medical treatment plan."""
    plan_id: str
    patient_id: str
    diagnosis_id: str
    treatment_type: TreatmentType
    description: str
    medications: List[Dict[str, Any]] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    duration_days: int = 0
    follow_up_schedule: List[datetime] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    created_by: str = "AI_System"
    approved_by: Optional[str] = None


@dataclass
class VitalSigns:
    """Patient vital signs."""
    patient_id: str
    timestamp: datetime
    heart_rate: Optional[float] = None  # bpm
    blood_pressure_systolic: Optional[float] = None  # mmHg
    blood_pressure_diastolic: Optional[float] = None  # mmHg
    temperature: Optional[float] = None  # Celsius
    respiratory_rate: Optional[float] = None  # breaths per minute
    oxygen_saturation: Optional[float] = None  # percentage
    blood_glucose: Optional[float] = None  # mg/dL
    pain_level: Optional[int] = None  # 0-10 scale


@dataclass
class LabResult:
    """Laboratory test result."""
    result_id: str
    patient_id: str
    test_name: str
    value: float
    unit: str
    reference_range: str
    status: str  # normal, abnormal, critical
    timestamp: datetime = field(default_factory=datetime.now)
    lab_technician: str = ""
    verified: bool = False


class MedicalImageAnalyzer:
    """Analyzes medical images for diagnostic purposes."""
    
    def __init__(self, config: HealthcareConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.analysis_history = []
    
    def load_diagnostic_model(self, image_type: MedicalImageType, model_path: str) -> None:
        """Load diagnostic model for specific image type."""
        try:
            # Simulate model loading (in real implementation, would load actual AI model)
            self.models[image_type] = {
                'path': model_path,
                'loaded_at': datetime.now(),
                'accuracy': 0.92,  # Example accuracy
                'specialization': image_type
            }
            self.logger.info(f"Loaded diagnostic model for {image_type}")
        except Exception as e:
            self.logger.error(f"Failed to load model for {image_type}: {e}")
            raise
    
    def analyze_image(self, medical_image: MedicalImage) -> DiagnosisResult:
        """Analyze medical image for diagnostic insights."""
        try:
            if medical_image.image_type not in self.models:
                raise ValueError(f"No model available for {medical_image.image_type}")
            
            # Simulate image analysis (in real implementation, would run AI model)
            model = self.models[medical_image.image_type]
            
            # Generate analysis based on image type
            diagnosis = self._generate_diagnosis(medical_image)
            
            # Create diagnosis result
            result = DiagnosisResult(
                diagnosis_id=f"DIAG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{medical_image.patient_id}",
                patient_id=medical_image.patient_id,
                diagnosis_type=self._map_image_to_diagnosis_type(medical_image.image_type),
                condition=diagnosis['condition'],
                confidence=diagnosis['confidence'],
                severity=diagnosis['severity'],
                evidence=diagnosis['evidence'],
                recommendations=diagnosis['recommendations']
            )
            
            self.analysis_history.append(result)
            
            # Keep only recent history
            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_years * 365)
            self.analysis_history = [r for r in self.analysis_history if r.timestamp > cutoff_date]
            
            self.logger.info(f"Analyzed image {medical_image.image_id}: {diagnosis['condition']}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to analyze image {medical_image.image_id}: {e}")
            raise
    
    def _generate_diagnosis(self, medical_image: MedicalImage) -> Dict[str, Any]:
        """Generate diagnosis based on image type."""
        # Simplified diagnosis generation (in real implementation, would use AI models)
        diagnoses = {
            MedicalImageType.XRAY: {
                'condition': 'Normal chest X-ray',
                'confidence': 0.89,
                'severity': 'normal',
                'evidence': ['Clear lung fields', 'Normal heart size'],
                'recommendations': ['Routine follow-up']
            },
            MedicalImageType.CT_SCAN: {
                'condition': 'No acute findings',
                'confidence': 0.92,
                'severity': 'normal',
                'evidence': ['Normal tissue density', 'No masses detected'],
                'recommendations': ['Continue monitoring']
            },
            MedicalImageType.MRI: {
                'condition': 'Normal brain MRI',
                'confidence': 0.94,
                'severity': 'normal',
                'evidence': ['Normal brain structure', 'No lesions'],
                'recommendations': ['No immediate action required']
            }
        }
        
        return diagnoses.get(medical_image.image_type, {
            'condition': 'Analysis pending',
            'confidence': 0.5,
            'severity': 'unknown',
            'evidence': [],
            'recommendations': ['Manual review required']
        })
    
    def _map_image_to_diagnosis_type(self, image_type: MedicalImageType) -> DiagnosisType:
        """Map image type to diagnosis type."""
        mapping = {
            MedicalImageType.XRAY: DiagnosisType.RESPIRATORY,
            MedicalImageType.CT_SCAN: DiagnosisType.CANCER,
            MedicalImageType.MRI: DiagnosisType.NEUROLOGICAL,
            MedicalImageType.MAMMOGRAPHY: DiagnosisType.CANCER,
            MedicalImageType.RETINAL: DiagnosisType.CARDIOVASCULAR,
            MedicalImageType.DERMATOLOGY: DiagnosisType.CANCER
        }
        return mapping.get(image_type, DiagnosisType.CANCER)


class ClinicalDecisionSupport:
    """Provides clinical decision support to healthcare providers."""
    
    def __init__(self, config: HealthcareConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = {}
        self.treatment_protocols = {}
        self.drug_interactions = {}
    
    def load_medical_knowledge(self, knowledge_data: Dict[str, Any]) -> None:
        """Load medical knowledge base."""
        try:
            self.knowledge_base.update(knowledge_data)
            self.logger.info("Loaded medical knowledge base")
        except Exception as e:
            self.logger.error(f"Failed to load medical knowledge: {e}")
            raise
    
    def recommend_treatment(self, patient: PatientInfo, diagnosis: DiagnosisResult) -> TreatmentPlan:
        """Recommend treatment plan based on patient and diagnosis."""
        try:
            plan_id = f"PLAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{patient.patient_id}"
            
            # Generate treatment recommendations based on diagnosis
            treatment_data = self._generate_treatment_recommendations(patient, diagnosis)
            
            plan = TreatmentPlan(
                plan_id=plan_id,
                patient_id=patient.patient_id,
                diagnosis_id=diagnosis.diagnosis_id,
                treatment_type=treatment_data['type'],
                description=treatment_data['description'],
                medications=treatment_data['medications'],
                procedures=treatment_data['procedures'],
                duration_days=treatment_data['duration'],
                follow_up_schedule=treatment_data['follow_up'],
                expected_outcomes=treatment_data['outcomes'],
                risks=treatment_data['risks']
            )
            
            # Check for drug interactions
            self._check_drug_interactions(plan, patient)
            
            self.logger.info(f"Generated treatment plan {plan_id} for {diagnosis.condition}")
            return plan
        except Exception as e:
            self.logger.error(f"Failed to recommend treatment: {e}")
            raise
    
    def _generate_treatment_recommendations(self, patient: PatientInfo, diagnosis: DiagnosisResult) -> Dict[str, Any]:
        """Generate treatment recommendations."""
        # Simplified treatment generation (in real implementation, would use medical AI)
        base_treatments = {
            DiagnosisType.RESPIRATORY: {
                'type': TreatmentType.MEDICATION,
                'description': 'Respiratory treatment protocol',
                'medications': [{'name': 'Albuterol', 'dosage': '2 puffs q4h', 'duration': 7}],
                'procedures': ['Chest physiotherapy'],
                'duration': 14,
                'outcomes': ['Improved breathing', 'Reduced symptoms'],
                'risks': ['Mild side effects']
            },
            DiagnosisType.CARDIOVASCULAR: {
                'type': TreatmentType.MEDICATION,
                'description': 'Cardiovascular management',
                'medications': [{'name': 'Lisinopril', 'dosage': '10mg daily', 'duration': 30}],
                'procedures': ['ECG monitoring'],
                'duration': 30,
                'outcomes': ['Blood pressure control', 'Reduced cardiac risk'],
                'risks': ['Hypotension', 'Kidney function changes']
            }
        }
        
        treatment = base_treatments.get(diagnosis.diagnosis_type, {
            'type': TreatmentType.THERAPY,
            'description': 'General supportive care',
            'medications': [],
            'procedures': ['Regular monitoring'],
            'duration': 7,
            'outcomes': ['Symptom improvement'],
            'risks': ['Minimal']
        })
        
        # Add follow-up schedule
        follow_up_dates = []
        for i in range(1, 4):  # 3 follow-ups
            follow_up_date = datetime.now() + timedelta(days=i * 7)
            follow_up_dates.append(follow_up_date)
        treatment['follow_up'] = follow_up_dates
        
        return treatment
    
    def _check_drug_interactions(self, plan: TreatmentPlan, patient: PatientInfo) -> None:
        """Check for drug interactions."""
        try:
            plan_medications = [med['name'] for med in plan.medications]
            all_medications = patient.current_medications + plan_medications
            
            # Simplified interaction checking
            known_interactions = {
                ('Warfarin', 'Aspirin'): 'Increased bleeding risk',
                ('Lisinopril', 'Potassium'): 'Hyperkalemia risk'
            }
            
            for med1 in all_medications:
                for med2 in all_medications:
                    if med1 != med2:
                        interaction = known_interactions.get((med1, med2)) or known_interactions.get((med2, med1))
                        if interaction:
                            plan.risks.append(f"Drug interaction: {med1} + {med2} - {interaction}")
                            self.logger.warning(f"Drug interaction detected: {interaction}")
        except Exception as e:
            self.logger.error(f"Failed to check drug interactions: {e}")


class PatientMonitor:
    """Monitors patient vital signs and health status."""
    
    def __init__(self, config: HealthcareConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.patient_data = {}
        self.vital_signs_history = {}
        self.alerts = []
        self.monitoring_active = False
    
    def register_patient(self, patient: PatientInfo) -> None:
        """Register patient for monitoring."""
        try:
            self.patient_data[patient.patient_id] = patient
            self.vital_signs_history[patient.patient_id] = []
            self.logger.info(f"Registered patient {patient.patient_id} for monitoring")
        except Exception as e:
            self.logger.error(f"Failed to register patient: {e}")
            raise
    
    def record_vital_signs(self, vital_signs: VitalSigns) -> None:
        """Record patient vital signs."""
        try:
            if vital_signs.patient_id not in self.patient_data:
                raise ValueError(f"Patient {vital_signs.patient_id} not registered")
            
            self.vital_signs_history[vital_signs.patient_id].append(vital_signs)
            
            # Keep only recent history (last 1000 readings)
            if len(self.vital_signs_history[vital_signs.patient_id]) > 1000:
                self.vital_signs_history[vital_signs.patient_id] = self.vital_signs_history[vital_signs.patient_id][-1000:]
            
            # Check for alerts
            self._check_vital_signs_alerts(vital_signs)
            
        except Exception as e:
            self.logger.error(f"Failed to record vital signs: {e}")
            raise
    
    def _check_vital_signs_alerts(self, vital_signs: VitalSigns) -> None:
        """Check vital signs for alert conditions."""
        try:
            alerts = []
            
            # Heart rate alerts
            if vital_signs.heart_rate:
                if vital_signs.heart_rate > 120 or vital_signs.heart_rate < 50:
                    alerts.append(f"Abnormal heart rate: {vital_signs.heart_rate} bpm")
            
            # Blood pressure alerts
            if vital_signs.blood_pressure_systolic and vital_signs.blood_pressure_diastolic:
                if vital_signs.blood_pressure_systolic > 180 or vital_signs.blood_pressure_diastolic > 110:
                    alerts.append(f"Hypertensive crisis: {vital_signs.blood_pressure_systolic}/{vital_signs.blood_pressure_diastolic} mmHg")
                elif vital_signs.blood_pressure_systolic < 90 or vital_signs.blood_pressure_diastolic < 60:
                    alerts.append(f"Hypotension: {vital_signs.blood_pressure_systolic}/{vital_signs.blood_pressure_diastolic} mmHg")
            
            # Temperature alerts
            if vital_signs.temperature:
                if vital_signs.temperature > 38.5:  # Fever
                    alerts.append(f"Fever: {vital_signs.temperature}°C")
                elif vital_signs.temperature < 35.0:  # Hypothermia
                    alerts.append(f"Hypothermia: {vital_signs.temperature}°C")
            
            # Oxygen saturation alerts
            if vital_signs.oxygen_saturation:
                if vital_signs.oxygen_saturation < 90:
                    alerts.append(f"Low oxygen saturation: {vital_signs.oxygen_saturation}%")
            
            # Record alerts
            for alert_msg in alerts:
                alert = {
                    'patient_id': vital_signs.patient_id,
                    'message': alert_msg,
                    'timestamp': vital_signs.timestamp,
                    'severity': 'high' if any(word in alert_msg.lower() for word in ['crisis', 'hypothermia']) else 'medium'
                }
                self.alerts.append(alert)
                self.logger.warning(f"Patient alert: {alert_msg}")
            
        except Exception as e:
            self.logger.error(f"Failed to check vital signs alerts: {e}")
    
    def get_patient_status(self, patient_id: str) -> Dict[str, Any]:
        """Get current patient status and trends."""
        try:
            if patient_id not in self.patient_data:
                raise ValueError(f"Patient {patient_id} not found")
            
            patient = self.patient_data[patient_id]
            recent_vitals = self.vital_signs_history[patient_id][-10:]  # Last 10 readings
            
            if not recent_vitals:
                return {'patient_id': patient_id, 'status': 'no_data'}
            
            latest_vitals = recent_vitals[-1]
            
            # Calculate trends
            trends = self._calculate_trends(recent_vitals)
            
            # Get recent alerts
            recent_alerts = [a for a in self.alerts 
                           if a['patient_id'] == patient_id and 
                           a['timestamp'] > datetime.now() - timedelta(hours=24)]
            
            status = {
                'patient_id': patient_id,
                'current_status': patient.status.value,
                'latest_vitals': {
                    'heart_rate': latest_vitals.heart_rate,
                    'blood_pressure': f"{latest_vitals.blood_pressure_systolic}/{latest_vitals.blood_pressure_diastolic}" if latest_vitals.blood_pressure_systolic else None,
                    'temperature': latest_vitals.temperature,
                    'oxygen_saturation': latest_vitals.oxygen_saturation,
                    'timestamp': latest_vitals.timestamp.isoformat()
                },
                'trends': trends,
                'recent_alerts': recent_alerts,
                'alert_count_24h': len(recent_alerts)
            }
            
            return status
        except Exception as e:
            self.logger.error(f"Failed to get patient status: {e}")
            return {}
    
    def _calculate_trends(self, vital_signs_list: List[VitalSigns]) -> Dict[str, str]:
        """Calculate trends in vital signs."""
        try:
            if len(vital_signs_list) < 2:
                return {}
            
            trends = {}
            
            # Heart rate trend
            hr_values = [v.heart_rate for v in vital_signs_list if v.heart_rate is not None]
            if len(hr_values) >= 2:
                if hr_values[-1] > hr_values[0]:
                    trends['heart_rate'] = 'increasing'
                elif hr_values[-1] < hr_values[0]:
                    trends['heart_rate'] = 'decreasing'
                else:
                    trends['heart_rate'] = 'stable'
            
            # Temperature trend
            temp_values = [v.temperature for v in vital_signs_list if v.temperature is not None]
            if len(temp_values) >= 2:
                if temp_values[-1] > temp_values[0]:
                    trends['temperature'] = 'increasing'
                elif temp_values[-1] < temp_values[0]:
                    trends['temperature'] = 'decreasing'
                else:
                    trends['temperature'] = 'stable'
            
            return trends
        except Exception as e:
            self.logger.error(f"Failed to calculate trends: {e}")
            return {}


class HealthcareAnalytics:
    """Provides healthcare analytics and insights."""
    
    def __init__(self, config: HealthcareConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analytics_data = []
    
    def analyze_patient_outcomes(self, patient_ids: List[str], 
                               diagnoses: List[DiagnosisResult],
                               treatments: List[TreatmentPlan]) -> Dict[str, Any]:
        """Analyze patient outcomes and treatment effectiveness."""
        try:
            # Group data by patient
            patient_data = {}
            for patient_id in patient_ids:
                patient_diagnoses = [d for d in diagnoses if d.patient_id == patient_id]
                patient_treatments = [t for t in treatments if t.patient_id == patient_id]
                patient_data[patient_id] = {
                    'diagnoses': patient_diagnoses,
                    'treatments': patient_treatments
                }
            
            # Calculate metrics
            total_patients = len(patient_ids)
            total_diagnoses = len(diagnoses)
            total_treatments = len(treatments)
            
            # Diagnosis distribution
            diagnosis_counts = {}
            for diagnosis in diagnoses:
                condition = diagnosis.condition
                diagnosis_counts[condition] = diagnosis_counts.get(condition, 0) + 1
            
            # Treatment distribution
            treatment_counts = {}
            for treatment in treatments:
                treatment_type = treatment.treatment_type.value
                treatment_counts[treatment_type] = treatment_counts.get(treatment_type, 0) + 1
            
            # Average confidence scores
            confidence_scores = [d.confidence for d in diagnoses]
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
            
            analytics = {
                'analysis_date': datetime.now().isoformat(),
                'total_patients': total_patients,
                'total_diagnoses': total_diagnoses,
                'total_treatments': total_treatments,
                'average_diagnosis_confidence': avg_confidence,
                'diagnosis_distribution': diagnosis_counts,
                'treatment_distribution': treatment_counts,
                'top_conditions': sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                'top_treatments': sorted(treatment_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
            return analytics
        except Exception as e:
            self.logger.error(f"Failed to analyze patient outcomes: {e}")
            return {}
    
    def generate_quality_metrics(self, diagnoses: List[DiagnosisResult]) -> Dict[str, Any]:
        """Generate quality metrics for diagnostic accuracy."""
        try:
            if not diagnoses:
                return {}
            
            # Calculate confidence distribution
            confidence_scores = [d.confidence for d in diagnoses]
            high_confidence = sum(1 for c in confidence_scores if c >= self.config.confidence_threshold)
            
            # Calculate by diagnosis type
            type_metrics = {}
            for diagnosis_type in DiagnosisType:
                type_diagnoses = [d for d in diagnoses if d.diagnosis_type == diagnosis_type]
                if type_diagnoses:
                    type_confidence = [d.confidence for d in type_diagnoses]
                    type_metrics[diagnosis_type.value] = {
                        'count': len(type_diagnoses),
                        'avg_confidence': statistics.mean(type_confidence),
                        'min_confidence': min(type_confidence),
                        'max_confidence': max(type_confidence)
                    }
            
            metrics = {
                'total_diagnoses': len(diagnoses),
                'high_confidence_count': high_confidence,
                'high_confidence_rate': high_confidence / len(diagnoses),
                'average_confidence': statistics.mean(confidence_scores),
                'confidence_std': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
                'by_diagnosis_type': type_metrics,
                'confidence_threshold': self.config.confidence_threshold
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to generate quality metrics: {e}")
            return {}


class HealthcareAISystem:
    """Main healthcare AI system integrating all components."""
    
    def __init__(self, config: Optional[HealthcareConfig] = None):
        self.config = config or HealthcareConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.image_analyzer = MedicalImageAnalyzer(self.config)
        self.decision_support = ClinicalDecisionSupport(self.config)
        self.patient_monitor = PatientMonitor(self.config)
        self.analytics = HealthcareAnalytics(self.config)
        
        self.running = False
        self.monitoring_task = None
    
    async def start(self) -> None:
        """Start healthcare AI system."""
        try:
            self.running = True
            if self.config.real_time_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info(f"Healthcare AI system started for facility {self.config.facility_id}")
        except Exception as e:
            self.logger.error(f"Failed to start healthcare AI system: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop healthcare AI system."""
        try:
            self.running = False
            if self.monitoring_task:
                await self.monitoring_task
            self.logger.info("Healthcare AI system stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop healthcare AI system: {e}")
            raise
    
    async def _monitoring_loop(self) -> None:
        """Real-time patient monitoring loop."""
        while self.running:
            try:
                # Check for critical alerts
                critical_alerts = [a for a in self.patient_monitor.alerts 
                                 if a['severity'] == 'high' and 
                                 a['timestamp'] > datetime.now() - timedelta(minutes=5)]
                
                if critical_alerts:
                    self.logger.warning(f"Critical alerts detected: {len(critical_alerts)}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get healthcare AI system status."""
        try:
            total_patients = len(self.patient_monitor.patient_data)
            total_alerts = len(self.patient_monitor.alerts)
            recent_alerts = len([a for a in self.patient_monitor.alerts 
                               if a['timestamp'] > datetime.now() - timedelta(hours=24)])
            
            status = {
                'facility_id': self.config.facility_id,
                'system_running': self.running,
                'privacy_mode': self.config.privacy_mode,
                'hipaa_compliant': self.config.hipaa_compliant,
                'total_patients_monitored': total_patients,
                'total_alerts': total_alerts,
                'alerts_last_24h': recent_alerts,
                'ai_assistance_level': self.config.ai_assistance_level,
                'confidence_threshold': self.config.confidence_threshold,
                'timestamp': datetime.now().isoformat()
            }
            
            return status
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {}


# Global healthcare AI system instance
_healthcare_system: Optional[HealthcareAISystem] = None


def initialize_healthcare(config: Optional[HealthcareConfig] = None) -> None:
    """Initialize healthcare AI system."""
    global _healthcare_system
    _healthcare_system = HealthcareAISystem(config)


async def shutdown_healthcare() -> None:
    """Shutdown healthcare AI system."""
    global _healthcare_system
    if _healthcare_system:
        await _healthcare_system.stop()
        _healthcare_system = None