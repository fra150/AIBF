#!/usr/bin/env python3
"""
Healthcare AI Example - AI Bull Ford Framework

This example demonstrates how to build a comprehensive healthcare AI system
using AIBF for medical image analysis, patient risk assessment, and clinical
decision support.

Features:
- Medical image classification (X-rays, CT scans, MRI)
- Patient risk stratification
- Drug interaction checking
- Clinical decision support
- HIPAA-compliant data handling
- Real-time monitoring

Author: AIBF Team
Date: 2024-01-22
Version: 1.0.0
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Add AIBF to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# AIBF imports
from src.core.architectures.neural_networks import NeuralNetwork
from src.core.architectures.cnn import ConvolutionalNetwork
from src.core.architectures.transformers import TransformerModel
from src.multimodal.vision import VisionProcessor
from src.multimodal.cross_modal import CrossModalProcessor
from src.multimodal.fusion import ModalityFusion
from src.enhancement.rag import RAGSystem
from src.enhancement.memory import MemoryManager
from src.security.validation import DataValidator
from src.security.encryption import EncryptionManager
from src.monitoring.analytics import AnalyticsCollector
from src.config.manager import ConfigManager
from src.api.rest import RESTServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PatientData:
    """Patient data structure with privacy protection"""
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    current_medications: List[str]
    vital_signs: Dict[str, float]
    lab_results: Dict[str, float]
    imaging_data: Optional[str] = None
    risk_factors: List[str] = None
    
    def __post_init__(self):
        # Validate patient data
        if self.age < 0 or self.age > 150:
            raise ValueError("Invalid age")
        if self.gender not in ['M', 'F', 'O']:
            raise ValueError("Invalid gender")

@dataclass
class DiagnosisResult:
    """Diagnosis result structure"""
    patient_id: str
    diagnosis: str
    confidence: float
    risk_score: float
    recommendations: List[str]
    imaging_findings: Optional[Dict[str, Any]] = None
    drug_interactions: List[str] = None
    follow_up_required: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MedicalImageClassifier:
    """Medical image classification system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vision_processor = VisionProcessor(
            model_name=config.get('vision_model', 'resnet50')
        )
        
        # Initialize specialized models for different imaging types
        self.xray_model = self._load_xray_model()
        self.ct_model = self._load_ct_model()
        self.mri_model = self._load_mri_model()
        
        logger.info("Medical image classifier initialized")
    
    def _load_xray_model(self) -> ConvolutionalNetwork:
        """Load chest X-ray classification model"""
        model = ConvolutionalNetwork(
            input_channels=1,  # Grayscale X-rays
            num_classes=14,    # Common chest pathologies
            architecture='densenet121',
            pretrained=True
        )
        
        # Load pre-trained weights if available
        weights_path = self.config.get('xray_weights_path')
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
            logger.info(f"Loaded X-ray model weights from {weights_path}")
        
        return model
    
    def _load_ct_model(self) -> ConvolutionalNetwork:
        """Load CT scan classification model"""
        model = ConvolutionalNetwork(
            input_channels=1,  # CT scans
            num_classes=8,     # Common CT findings
            architecture='resnet50',
            pretrained=True
        )
        
        weights_path = self.config.get('ct_weights_path')
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
            logger.info(f"Loaded CT model weights from {weights_path}")
        
        return model
    
    def _load_mri_model(self) -> ConvolutionalNetwork:
        """Load MRI classification model"""
        model = ConvolutionalNetwork(
            input_channels=1,  # MRI scans
            num_classes=6,     # Common MRI findings
            architecture='vgg16',
            pretrained=True
        )
        
        weights_path = self.config.get('mri_weights_path')
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
            logger.info(f"Loaded MRI model weights from {weights_path}")
        
        return model
    
    def classify_image(self, image_path: str, image_type: str) -> Dict[str, Any]:
        """Classify medical image"""
        try:
            # Load and preprocess image
            image = self.vision_processor.load_image(image_path)
            processed_image = self.vision_processor.preprocess(
                image, 
                resize=(224, 224),
                normalize=True
            )
            
            # Select appropriate model
            if image_type.lower() == 'xray':
                model = self.xray_model
                classes = self._get_xray_classes()
            elif image_type.lower() == 'ct':
                model = self.ct_model
                classes = self._get_ct_classes()
            elif image_type.lower() == 'mri':
                model = self.mri_model
                classes = self._get_mri_classes()
            else:
                raise ValueError(f"Unsupported image type: {image_type}")
            
            # Make prediction
            with torch.no_grad():
                model.eval()
                outputs = model(processed_image.unsqueeze(0))
                probabilities = torch.softmax(outputs, dim=1)
                
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=3)
            
            results = {
                'image_type': image_type,
                'predictions': [
                    {
                        'class': classes[idx.item()],
                        'confidence': prob.item(),
                        'severity': self._assess_severity(classes[idx.item()], prob.item())
                    }
                    for prob, idx in zip(top_probs[0], top_indices[0])
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Classified {image_type} image: {results['predictions'][0]['class']}")
            return results
            
        except Exception as e:
            logger.error(f"Error classifying image: {str(e)}")
            raise
    
    def _get_xray_classes(self) -> List[str]:
        """Get chest X-ray classification classes"""
        return [
            'Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis',
            'Lung Cancer', 'Pneumothorax', 'Pleural Effusion',
            'Cardiomegaly', 'Atelectasis', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Mass'
        ]
    
    def _get_ct_classes(self) -> List[str]:
        """Get CT scan classification classes"""
        return [
            'Normal', 'Stroke', 'Hemorrhage', 'Tumor',
            'Infection', 'Trauma', 'Vascular', 'Other'
        ]
    
    def _get_mri_classes(self) -> List[str]:
        """Get MRI classification classes"""
        return [
            'Normal', 'Tumor', 'MS Lesions', 'Stroke',
            'Inflammation', 'Degenerative'
        ]
    
    def _assess_severity(self, diagnosis: str, confidence: float) -> str:
        """Assess severity based on diagnosis and confidence"""
        critical_conditions = ['Stroke', 'Hemorrhage', 'Pneumothorax', 'COVID-19']
        serious_conditions = ['Pneumonia', 'Tumor', 'Lung Cancer', 'Tuberculosis']
        
        if diagnosis in critical_conditions and confidence > 0.8:
            return 'Critical'
        elif diagnosis in serious_conditions and confidence > 0.7:
            return 'Serious'
        elif confidence > 0.6:
            return 'Moderate'
        else:
            return 'Low'

class PatientRiskAssessment:
    """Patient risk stratification system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_model = self._initialize_risk_model()
        self.memory_manager = MemoryManager()
        
        logger.info("Patient risk assessment system initialized")
    
    def _initialize_risk_model(self) -> NeuralNetwork:
        """Initialize risk assessment neural network"""
        model = NeuralNetwork(
            input_size=50,  # Feature vector size
            hidden_sizes=[128, 64, 32],
            output_size=1,  # Risk score
            activation='relu',
            dropout_rate=0.3
        )
        
        # Load pre-trained weights if available
        weights_path = self.config.get('risk_model_weights')
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
            logger.info(f"Loaded risk model weights from {weights_path}")
        
        return model
    
    def assess_patient_risk(self, patient: PatientData) -> Dict[str, Any]:
        """Assess patient risk based on multiple factors"""
        try:
            # Extract features from patient data
            features = self._extract_features(patient)
            
            # Normalize features
            normalized_features = self._normalize_features(features)
            
            # Predict risk score
            with torch.no_grad():
                self.risk_model.eval()
                risk_tensor = torch.tensor(normalized_features, dtype=torch.float32)
                risk_score = torch.sigmoid(self.risk_model(risk_tensor.unsqueeze(0)))
            
            risk_value = risk_score.item()
            risk_category = self._categorize_risk(risk_value)
            
            # Generate risk factors and recommendations
            risk_factors = self._identify_risk_factors(patient, features)
            recommendations = self._generate_recommendations(patient, risk_value, risk_factors)
            
            # Store in memory for future reference
            self.memory_manager.store_episodic_memory(
                f"risk_assessment_{patient.patient_id}",
                {
                    'patient_id': patient.patient_id,
                    'risk_score': risk_value,
                    'risk_category': risk_category,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            result = {
                'patient_id': patient.patient_id,
                'risk_score': risk_value,
                'risk_category': risk_category,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'assessment_date': datetime.now().isoformat()
            }
            
            logger.info(f"Risk assessment for patient {patient.patient_id}: {risk_category} ({risk_value:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            raise
    
    def _extract_features(self, patient: PatientData) -> np.ndarray:
        """Extract numerical features from patient data"""
        features = np.zeros(50)
        
        # Demographic features
        features[0] = patient.age / 100.0  # Normalized age
        features[1] = 1.0 if patient.gender == 'M' else 0.0
        features[2] = 1.0 if patient.gender == 'F' else 0.0
        
        # Vital signs (normalized)
        vital_signs = patient.vital_signs
        features[3] = vital_signs.get('heart_rate', 70) / 200.0
        features[4] = vital_signs.get('blood_pressure_systolic', 120) / 300.0
        features[5] = vital_signs.get('blood_pressure_diastolic', 80) / 200.0
        features[6] = vital_signs.get('temperature', 98.6) / 110.0
        features[7] = vital_signs.get('respiratory_rate', 16) / 40.0
        features[8] = vital_signs.get('oxygen_saturation', 98) / 100.0
        
        # Lab results (normalized)
        lab_results = patient.lab_results
        features[9] = lab_results.get('glucose', 100) / 500.0
        features[10] = lab_results.get('cholesterol', 200) / 400.0
        features[11] = lab_results.get('hemoglobin', 14) / 20.0
        features[12] = lab_results.get('white_blood_cells', 7000) / 20000.0
        features[13] = lab_results.get('creatinine', 1.0) / 5.0
        
        # Medical history (binary features)
        history_conditions = [
            'diabetes', 'hypertension', 'heart_disease', 'stroke',
            'cancer', 'kidney_disease', 'liver_disease', 'copd',
            'asthma', 'depression', 'anxiety', 'obesity'
        ]
        
        for i, condition in enumerate(history_conditions):
            features[14 + i] = 1.0 if condition in patient.medical_history else 0.0
        
        # Risk factors
        if patient.risk_factors:
            risk_factors = [
                'smoking', 'alcohol', 'sedentary', 'poor_diet',
                'stress', 'family_history', 'environmental'
            ]
            
            for i, factor in enumerate(risk_factors):
                if i < 12:  # Limit to available feature slots
                    features[26 + i] = 1.0 if factor in patient.risk_factors else 0.0
        
        # Medication count and interactions
        features[38] = min(len(patient.current_medications), 20) / 20.0
        
        # Age-specific risk adjustments
        if patient.age > 65:
            features[39] = 1.0
        if patient.age > 80:
            features[40] = 1.0
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        # Features are already normalized in extraction
        return np.clip(features, 0.0, 1.0)
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into risk levels"""
        if risk_score >= 0.8:
            return 'Very High'
        elif risk_score >= 0.6:
            return 'High'
        elif risk_score >= 0.4:
            return 'Moderate'
        elif risk_score >= 0.2:
            return 'Low'
        else:
            return 'Very Low'
    
    def _identify_risk_factors(self, patient: PatientData, features: np.ndarray) -> List[str]:
        """Identify key risk factors for the patient"""
        risk_factors = []
        
        # Age-related risks
        if patient.age > 65:
            risk_factors.append('Advanced age')
        
        # Vital signs risks
        vital_signs = patient.vital_signs
        if vital_signs.get('blood_pressure_systolic', 120) > 140:
            risk_factors.append('Hypertension')
        if vital_signs.get('heart_rate', 70) > 100:
            risk_factors.append('Tachycardia')
        if vital_signs.get('oxygen_saturation', 98) < 95:
            risk_factors.append('Low oxygen saturation')
        
        # Lab results risks
        lab_results = patient.lab_results
        if lab_results.get('glucose', 100) > 126:
            risk_factors.append('Elevated glucose')
        if lab_results.get('cholesterol', 200) > 240:
            risk_factors.append('High cholesterol')
        if lab_results.get('creatinine', 1.0) > 1.5:
            risk_factors.append('Kidney dysfunction')
        
        # Medical history risks
        high_risk_conditions = ['diabetes', 'heart_disease', 'stroke', 'cancer']
        for condition in high_risk_conditions:
            if condition in patient.medical_history:
                risk_factors.append(f'History of {condition}')
        
        # Medication risks
        if len(patient.current_medications) > 10:
            risk_factors.append('Polypharmacy')
        
        return risk_factors
    
    def _generate_recommendations(self, patient: PatientData, risk_score: float, risk_factors: List[str]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.append('Immediate medical attention required')
            recommendations.append('Consider hospitalization or intensive monitoring')
        elif risk_score >= 0.6:
            recommendations.append('Schedule urgent follow-up within 24-48 hours')
            recommendations.append('Consider additional diagnostic tests')
        elif risk_score >= 0.4:
            recommendations.append('Schedule follow-up within 1-2 weeks')
            recommendations.append('Monitor symptoms closely')
        
        # Specific recommendations based on risk factors
        if 'Hypertension' in risk_factors:
            recommendations.append('Blood pressure monitoring and management')
        if 'Elevated glucose' in risk_factors:
            recommendations.append('Diabetes screening and management')
        if 'High cholesterol' in risk_factors:
            recommendations.append('Lipid management and dietary counseling')
        if 'Polypharmacy' in risk_factors:
            recommendations.append('Medication review and optimization')
        
        # General health recommendations
        if patient.age > 50:
            recommendations.append('Regular preventive screenings')
        
        recommendations.append('Lifestyle modifications as appropriate')
        recommendations.append('Patient education on risk factors')
        
        return recommendations

class DrugInteractionChecker:
    """Drug interaction and contraindication checker"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = RAGSystem(
            vector_store_type='faiss',
            embedding_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Load drug interaction database
        self._load_drug_database()
        
        logger.info("Drug interaction checker initialized")
    
    def _load_drug_database(self):
        """Load drug interaction database into RAG system"""
        # In a real implementation, this would load from a comprehensive drug database
        drug_interactions = [
            "Warfarin and Aspirin: Increased bleeding risk",
            "ACE inhibitors and Potassium supplements: Hyperkalemia risk",
            "Statins and Fibrates: Increased myopathy risk",
            "Digoxin and Diuretics: Increased digoxin toxicity",
            "MAO inhibitors and SSRIs: Serotonin syndrome risk",
            "Beta-blockers and Calcium channel blockers: Bradycardia risk",
            "Lithium and NSAIDs: Increased lithium levels",
            "Metformin and Contrast agents: Lactic acidosis risk"
        ]
        
        # Add documents to RAG system
        for interaction in drug_interactions:
            self.rag_system.add_document(interaction)
        
        logger.info(f"Loaded {len(drug_interactions)} drug interactions")
    
    def check_interactions(self, medications: List[str]) -> Dict[str, Any]:
        """Check for drug interactions"""
        try:
            interactions = []
            warnings = []
            
            # Check each medication pair
            for i, med1 in enumerate(medications):
                for med2 in medications[i+1:]:
                    query = f"{med1} and {med2} interaction"
                    
                    # Search for interactions using RAG
                    results = self.rag_system.query(
                        query=query,
                        top_k=3,
                        similarity_threshold=0.7
                    )
                    
                    if results['documents']:
                        for doc in results['documents']:
                            if med1.lower() in doc.lower() and med2.lower() in doc.lower():
                                interactions.append({
                                    'medication1': med1,
                                    'medication2': med2,
                                    'interaction': doc,
                                    'severity': self._assess_interaction_severity(doc)
                                })
            
            # Generate warnings
            for interaction in interactions:
                if interaction['severity'] in ['High', 'Critical']:
                    warnings.append(f"WARNING: {interaction['interaction']}")
            
            result = {
                'medications_checked': medications,
                'interactions_found': len(interactions),
                'interactions': interactions,
                'warnings': warnings,
                'check_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Checked {len(medications)} medications, found {len(interactions)} interactions")
            return result
            
        except Exception as e:
            logger.error(f"Error checking drug interactions: {str(e)}")
            raise
    
    def _assess_interaction_severity(self, interaction_text: str) -> str:
        """Assess severity of drug interaction"""
        critical_keywords = ['death', 'fatal', 'severe', 'critical']
        high_keywords = ['bleeding', 'toxicity', 'syndrome', 'dangerous']
        moderate_keywords = ['increased', 'decreased', 'risk', 'caution']
        
        text_lower = interaction_text.lower()
        
        if any(keyword in text_lower for keyword in critical_keywords):
            return 'Critical'
        elif any(keyword in text_lower for keyword in high_keywords):
            return 'High'
        elif any(keyword in text_lower for keyword in moderate_keywords):
            return 'Moderate'
        else:
            return 'Low'

class ClinicalDecisionSupport:
    """Clinical decision support system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_classifier = MedicalImageClassifier(config)
        self.risk_assessor = PatientRiskAssessment(config)
        self.drug_checker = DrugInteractionChecker(config)
        self.cross_modal = CrossModalProcessor(model_name='clip')
        self.fusion = ModalityFusion(fusion_type='attention')
        
        # Initialize clinical knowledge base
        self.knowledge_base = RAGSystem(
            vector_store_type='faiss',
            embedding_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        self._load_clinical_knowledge()
        
        logger.info("Clinical decision support system initialized")
    
    def _load_clinical_knowledge(self):
        """Load clinical guidelines and knowledge"""
        # In a real implementation, this would load from medical literature
        clinical_guidelines = [
            "Chest pain with ST elevation requires immediate PCI or thrombolysis",
            "Blood pressure >180/120 requires immediate treatment",
            "Oxygen saturation <90% requires supplemental oxygen",
            "Fever >101.3°F with altered mental status suggests sepsis",
            "Sudden severe headache may indicate subarachnoid hemorrhage",
            "Shortness of breath with leg swelling suggests heart failure",
            "Abdominal pain with guarding requires surgical evaluation"
        ]
        
        for guideline in clinical_guidelines:
            self.knowledge_base.add_document(guideline)
        
        logger.info(f"Loaded {len(clinical_guidelines)} clinical guidelines")
    
    def comprehensive_assessment(self, patient: PatientData, image_path: Optional[str] = None, image_type: Optional[str] = None) -> DiagnosisResult:
        """Perform comprehensive patient assessment"""
        try:
            logger.info(f"Starting comprehensive assessment for patient {patient.patient_id}")
            
            # Risk assessment
            risk_assessment = self.risk_assessor.assess_patient_risk(patient)
            
            # Drug interaction check
            drug_interactions = self.drug_checker.check_interactions(patient.current_medications)
            
            # Image analysis if provided
            imaging_findings = None
            if image_path and image_type:
                imaging_findings = self.image_classifier.classify_image(image_path, image_type)
            
            # Generate clinical recommendations
            recommendations = self._generate_clinical_recommendations(
                patient, risk_assessment, drug_interactions, imaging_findings
            )
            
            # Determine primary diagnosis and confidence
            diagnosis, confidence = self._determine_diagnosis(
                patient, risk_assessment, imaging_findings
            )
            
            # Create comprehensive result
            result = DiagnosisResult(
                patient_id=patient.patient_id,
                diagnosis=diagnosis,
                confidence=confidence,
                risk_score=risk_assessment['risk_score'],
                recommendations=recommendations,
                imaging_findings=imaging_findings,
                drug_interactions=drug_interactions['warnings'],
                follow_up_required=self._requires_follow_up(risk_assessment, imaging_findings)
            )
            
            logger.info(f"Completed assessment for patient {patient.patient_id}: {diagnosis}")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive assessment: {str(e)}")
            raise
    
    def _generate_clinical_recommendations(self, patient: PatientData, risk_assessment: Dict, drug_interactions: Dict, imaging_findings: Optional[Dict]) -> List[str]:
        """Generate clinical recommendations based on all available data"""
        recommendations = []
        
        # Add risk-based recommendations
        recommendations.extend(risk_assessment['recommendations'])
        
        # Add drug interaction warnings
        if drug_interactions['warnings']:
            recommendations.append('Review medication interactions immediately')
            recommendations.extend(drug_interactions['warnings'])
        
        # Add imaging-based recommendations
        if imaging_findings:
            for prediction in imaging_findings['predictions']:
                if prediction['severity'] in ['Critical', 'Serious']:
                    recommendations.append(f"Urgent evaluation for {prediction['class']}")
        
        # Query knowledge base for additional recommendations
        symptoms = ' '.join(patient.medical_history)
        knowledge_results = self.knowledge_base.query(
            query=symptoms,
            top_k=3,
            similarity_threshold=0.6
        )
        
        for doc in knowledge_results['documents']:
            recommendations.append(f"Clinical guideline: {doc}")
        
        # Remove duplicates and sort by priority
        unique_recommendations = list(set(recommendations))
        return self._prioritize_recommendations(unique_recommendations)
    
    def _determine_diagnosis(self, patient: PatientData, risk_assessment: Dict, imaging_findings: Optional[Dict]) -> Tuple[str, float]:
        """Determine primary diagnosis and confidence"""
        # Simple rule-based diagnosis (in practice, this would be more sophisticated)
        diagnosis = "Assessment pending"
        confidence = 0.5
        
        # Check imaging findings first
        if imaging_findings:
            top_prediction = imaging_findings['predictions'][0]
            if top_prediction['confidence'] > 0.8:
                diagnosis = top_prediction['class']
                confidence = top_prediction['confidence']
                return diagnosis, confidence
        
        # Check risk factors and symptoms
        if 'diabetes' in patient.medical_history and patient.lab_results.get('glucose', 100) > 200:
            diagnosis = "Diabetic crisis"
            confidence = 0.85
        elif patient.vital_signs.get('blood_pressure_systolic', 120) > 180:
            diagnosis = "Hypertensive emergency"
            confidence = 0.8
        elif patient.vital_signs.get('oxygen_saturation', 98) < 90:
            diagnosis = "Respiratory failure"
            confidence = 0.9
        elif risk_assessment['risk_score'] > 0.8:
            diagnosis = "High-risk patient requiring immediate attention"
            confidence = risk_assessment['risk_score']
        
        return diagnosis, confidence
    
    def _requires_follow_up(self, risk_assessment: Dict, imaging_findings: Optional[Dict]) -> bool:
        """Determine if follow-up is required"""
        if risk_assessment['risk_score'] > 0.4:
            return True
        
        if imaging_findings:
            for prediction in imaging_findings['predictions']:
                if prediction['severity'] in ['Critical', 'Serious', 'Moderate']:
                    return True
        
        return False
    
    def _prioritize_recommendations(self, recommendations: List[str]) -> List[str]:
        """Prioritize recommendations by urgency"""
        urgent_keywords = ['immediate', 'urgent', 'emergency', 'critical']
        important_keywords = ['review', 'monitor', 'follow-up']
        
        urgent = []
        important = []
        routine = []
        
        for rec in recommendations:
            rec_lower = rec.lower()
            if any(keyword in rec_lower for keyword in urgent_keywords):
                urgent.append(rec)
            elif any(keyword in rec_lower for keyword in important_keywords):
                important.append(rec)
            else:
                routine.append(rec)
        
        return urgent + important + routine

class HealthcareAISystem:
    """Main healthcare AI system orchestrator"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config_manager.load_from_file(config_path)
        else:
            self.config_manager.load_default()
        
        self.config = self.config_manager.get_config()
        
        # Initialize security components
        self.data_validator = DataValidator()
        self.encryption_manager = EncryptionManager()
        
        # Initialize analytics
        self.analytics = AnalyticsCollector()
        
        # Initialize clinical decision support
        self.clinical_support = ClinicalDecisionSupport(self.config)
        
        # Initialize API server
        self.api_server = None
        
        logger.info("Healthcare AI system initialized")
    
    def validate_patient_data(self, patient_data: Dict[str, Any]) -> bool:
        """Validate patient data for HIPAA compliance"""
        schema = {
            'patient_id': {'type': 'string', 'required': True},
            'age': {'type': 'integer', 'min': 0, 'max': 150},
            'gender': {'type': 'string', 'enum': ['M', 'F', 'O']},
            'medical_history': {'type': 'array'},
            'current_medications': {'type': 'array'},
            'vital_signs': {'type': 'object'},
            'lab_results': {'type': 'object'}
        }
        
        is_valid, errors = self.data_validator.validate(patient_data, schema)
        if not is_valid:
            logger.error(f"Patient data validation failed: {errors}")
        
        return is_valid
    
    def encrypt_patient_data(self, patient_data: Dict[str, Any]) -> str:
        """Encrypt patient data for secure storage"""
        return self.encryption_manager.encrypt_data(patient_data)
    
    def decrypt_patient_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt patient data"""
        return self.encryption_manager.decrypt_data(encrypted_data)
    
    def process_patient(self, patient_data: Dict[str, Any], image_path: Optional[str] = None, image_type: Optional[str] = None) -> Dict[str, Any]:
        """Process patient through the complete AI pipeline"""
        try:
            # Validate input data
            if not self.validate_patient_data(patient_data):
                raise ValueError("Invalid patient data")
            
            # Create patient object
            patient = PatientData(**patient_data)
            
            # Track analytics
            self.analytics.track_event('patient_processing_started', {
                'patient_id': patient.patient_id,
                'has_imaging': image_path is not None
            })
            
            # Perform comprehensive assessment
            start_time = datetime.now()
            diagnosis_result = self.clinical_support.comprehensive_assessment(
                patient, image_path, image_type
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Track completion
            self.analytics.track_event('patient_processing_completed', {
                'patient_id': patient.patient_id,
                'diagnosis': diagnosis_result.diagnosis,
                'confidence': diagnosis_result.confidence,
                'processing_time': processing_time
            })
            
            # Convert to dictionary for API response
            result = {
                'patient_id': diagnosis_result.patient_id,
                'diagnosis': diagnosis_result.diagnosis,
                'confidence': diagnosis_result.confidence,
                'risk_score': diagnosis_result.risk_score,
                'recommendations': diagnosis_result.recommendations,
                'imaging_findings': diagnosis_result.imaging_findings,
                'drug_interactions': diagnosis_result.drug_interactions,
                'follow_up_required': diagnosis_result.follow_up_required,
                'timestamp': diagnosis_result.timestamp.isoformat(),
                'processing_time_seconds': processing_time
            }
            
            logger.info(f"Successfully processed patient {patient.patient_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing patient: {str(e)}")
            self.analytics.track_event('patient_processing_error', {
                'error': str(e),
                'patient_id': patient_data.get('patient_id', 'unknown')
            })
            raise
    
    def start_api_server(self, host: str = '0.0.0.0', port: int = 8080):
        """Start the healthcare AI API server"""
        self.api_server = RESTServer(host=host, port=port)
        
        # Define API endpoints
        @self.api_server.route('/health', methods=['GET'])
        def health_check():
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        
        @self.api_server.route('/api/v1/diagnose', methods=['POST'])
        def diagnose_patient():
            try:
                data = self.api_server.get_json()
                patient_data = data.get('patient_data')
                image_path = data.get('image_path')
                image_type = data.get('image_type')
                
                result = self.process_patient(patient_data, image_path, image_type)
                return {'success': True, 'result': result}
                
            except Exception as e:
                return {'success': False, 'error': str(e)}, 400
        
        @self.api_server.route('/api/v1/risk-assessment', methods=['POST'])
        def assess_risk():
            try:
                data = self.api_server.get_json()
                patient_data = data.get('patient_data')
                
                if not self.validate_patient_data(patient_data):
                    return {'success': False, 'error': 'Invalid patient data'}, 400
                
                patient = PatientData(**patient_data)
                risk_result = self.clinical_support.risk_assessor.assess_patient_risk(patient)
                
                return {'success': True, 'result': risk_result}
                
            except Exception as e:
                return {'success': False, 'error': str(e)}, 400
        
        @self.api_server.route('/api/v1/drug-interactions', methods=['POST'])
        def check_drug_interactions():
            try:
                data = self.api_server.get_json()
                medications = data.get('medications', [])
                
                if not medications:
                    return {'success': False, 'error': 'No medications provided'}, 400
                
                interaction_result = self.clinical_support.drug_checker.check_interactions(medications)
                
                return {'success': True, 'result': interaction_result}
                
            except Exception as e:
                return {'success': False, 'error': str(e)}, 400
        
        # Start server
        logger.info(f"Starting healthcare AI API server on {host}:{port}")
        self.api_server.start()
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Get analytics report"""
        return self.analytics.get_report()

def main():
    """Main function to demonstrate the healthcare AI system"""
    # Initialize system
    healthcare_ai = HealthcareAISystem()
    
    # Sample patient data
    sample_patient = {
        'patient_id': 'P001',
        'age': 65,
        'gender': 'M',
        'medical_history': ['diabetes', 'hypertension', 'heart_disease'],
        'current_medications': ['metformin', 'lisinopril', 'atorvastatin'],
        'vital_signs': {
            'heart_rate': 85,
            'blood_pressure_systolic': 150,
            'blood_pressure_diastolic': 90,
            'temperature': 98.6,
            'respiratory_rate': 18,
            'oxygen_saturation': 96
        },
        'lab_results': {
            'glucose': 180,
            'cholesterol': 220,
            'hemoglobin': 13.5,
            'white_blood_cells': 8000,
            'creatinine': 1.2
        },
        'risk_factors': ['smoking', 'sedentary', 'family_history']
    }
    
    try:
        # Process patient
        print("Processing patient...")
        result = healthcare_ai.process_patient(sample_patient)
        
        print("\n=== HEALTHCARE AI ASSESSMENT RESULTS ===")
        print(f"Patient ID: {result['patient_id']}")
        print(f"Diagnosis: {result['diagnosis']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Risk Score: {result['risk_score']:.3f}")
        print(f"Follow-up Required: {result['follow_up_required']}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        if result['drug_interactions']:
            print("\nDrug Interaction Warnings:")
            for warning in result['drug_interactions']:
                print(f"  ⚠️  {warning}")
        
        print(f"\nProcessing Time: {result['processing_time_seconds']:.2f} seconds")
        
        # Get analytics report
        analytics_report = healthcare_ai.get_analytics_report()
        print(f"\nTotal Patients Processed: {analytics_report.get('total_events', 0)}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()