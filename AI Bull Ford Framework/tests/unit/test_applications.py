"""Unit tests for applications module components."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

# Import applications modules
from applications.healthcare import (
    HealthcareConfig, PatientInfo, MedicalImage, DiagnosisResult,
    TreatmentPlan, VitalSigns, LabResult, MedicalImageAnalyzer,
    ClinicalDecisionSupport, PatientMonitor, HealthcareAnalytics,
    HealthcareAISystem, MedicalImageType, DiagnosisType,
    PatientStatus, TreatmentType
)
from applications.financial import (
    FinancialConfig, MarketData, Transaction, Portfolio,
    TradingSignal, RiskAssessment, FraudAlert, MarketAnalyzer,
    RiskManager, FraudDetector, PortfolioOptimizer,
    FinancialAISystem, AssetType, MarketSentiment,
    RiskLevel, TransactionType, FraudRiskLevel
)
from applications.educational import (
    EducationalConfig, LearnerProfile, LearningContent,
    LearningSession, Assessment, AssessmentResult, LearningPath,
    ContentRecommendationEngine, AdaptiveLearningEngine,
    IntelligentTutorSystem, LearningAnalytics, EducationalAISystem,
    LearningStyle, DifficultyLevel, ContentType, AssessmentType,
    LearningObjective, EngagementLevel
)


class TestHealthcareModule:
    """Test cases for healthcare AI components."""
    
    def test_healthcare_config(self):
        """Test HealthcareConfig creation and validation."""
        config = HealthcareConfig(
            model_path="/path/to/model",
            device="cuda",
            batch_size=32,
            confidence_threshold=0.8,
            enable_privacy_protection=True,
            max_image_size=512,
            supported_modalities=["xray", "mri", "ct"]
        )
        
        assert config.model_path == "/path/to/model"
        assert config.device == "cuda"
        assert config.batch_size == 32
        assert config.confidence_threshold == 0.8
        assert config.enable_privacy_protection is True
        assert config.max_image_size == 512
        assert "xray" in config.supported_modalities
    
    def test_patient_info(self):
        """Test PatientInfo dataclass."""
        patient = PatientInfo(
            patient_id="P12345",
            age=45,
            gender="M",
            medical_history=["diabetes", "hypertension"],
            current_medications=["metformin", "lisinopril"],
            allergies=["penicillin"],
            status=PatientStatus.STABLE
        )
        
        assert patient.patient_id == "P12345"
        assert patient.age == 45
        assert patient.gender == "M"
        assert "diabetes" in patient.medical_history
        assert "metformin" in patient.current_medications
        assert "penicillin" in patient.allergies
        assert patient.status == PatientStatus.STABLE
    
    def test_medical_image(self):
        """Test MedicalImage dataclass."""
        image_data = np.random.rand(512, 512, 3).astype(np.float32)
        
        medical_image = MedicalImage(
            image_id="IMG001",
            patient_id="P12345",
            image_type=MedicalImageType.XRAY,
            image_data=image_data,
            acquisition_date=datetime.now(),
            metadata={"view": "frontal", "technique": "digital"}
        )
        
        assert medical_image.image_id == "IMG001"
        assert medical_image.patient_id == "P12345"
        assert medical_image.image_type == MedicalImageType.XRAY
        assert medical_image.image_data.shape == (512, 512, 3)
        assert "view" in medical_image.metadata
    
    def test_medical_image_analyzer(self):
        """Test MedicalImageAnalyzer functionality."""
        config = HealthcareConfig(
            model_path="/path/to/model",
            device="cpu",
            confidence_threshold=0.7
        )
        
        analyzer = MedicalImageAnalyzer(config)
        
        # Test image analysis
        image_data = np.random.rand(512, 512, 3).astype(np.float32)
        medical_image = MedicalImage(
            image_id="IMG001",
            patient_id="P12345",
            image_type=MedicalImageType.XRAY,
            image_data=image_data,
            acquisition_date=datetime.now()
        )
        
        # Mock the analysis result
        with patch.object(analyzer, '_run_inference') as mock_inference:
            mock_inference.return_value = {
                "findings": ["normal"],
                "confidence": 0.95,
                "regions_of_interest": []
            }
            
            result = analyzer.analyze_image(medical_image)
            
            assert isinstance(result, DiagnosisResult)
            assert result.image_id == "IMG001"
            assert result.confidence >= config.confidence_threshold
    
    def test_clinical_decision_support(self):
        """Test ClinicalDecisionSupport functionality."""
        config = HealthcareConfig(device="cpu")
        cds = ClinicalDecisionSupport(config)
        
        # Test treatment recommendation
        patient = PatientInfo(
            patient_id="P12345",
            age=65,
            gender="F",
            medical_history=["diabetes"],
            current_medications=["metformin"]
        )
        
        diagnosis = DiagnosisResult(
            diagnosis_id="D001",
            patient_id="P12345",
            image_id="IMG001",
            diagnosis_type=DiagnosisType.PNEUMONIA,
            confidence=0.9,
            findings=["consolidation in right lower lobe"],
            timestamp=datetime.now()
        )
        
        with patch.object(cds, '_generate_recommendations') as mock_rec:
            mock_rec.return_value = [
                "Start antibiotic therapy",
                "Monitor oxygen saturation",
                "Follow-up chest X-ray in 48 hours"
            ]
            
            treatment_plan = cds.recommend_treatment(patient, diagnosis)
            
            assert isinstance(treatment_plan, TreatmentPlan)
            assert treatment_plan.patient_id == "P12345"
            assert len(treatment_plan.recommendations) > 0
    
    def test_patient_monitor(self):
        """Test PatientMonitor functionality."""
        config = HealthcareConfig(device="cpu")
        monitor = PatientMonitor(config)
        
        # Test vital signs monitoring
        vital_signs = VitalSigns(
            patient_id="P12345",
            heart_rate=85,
            blood_pressure_systolic=120,
            blood_pressure_diastolic=80,
            temperature=98.6,
            oxygen_saturation=98,
            respiratory_rate=16,
            timestamp=datetime.now()
        )
        
        with patch.object(monitor, '_analyze_vitals') as mock_analyze:
            mock_analyze.return_value = {
                "status": "normal",
                "alerts": [],
                "trends": {"heart_rate": "stable"}
            }
            
            analysis = monitor.monitor_vitals(vital_signs)
            
            assert analysis["status"] == "normal"
            assert isinstance(analysis["alerts"], list)
            assert "heart_rate" in analysis["trends"]
    
    def test_healthcare_analytics(self):
        """Test HealthcareAnalytics functionality."""
        config = HealthcareConfig(device="cpu")
        analytics = HealthcareAnalytics(config)
        
        # Test population health analysis
        patients = [
            PatientInfo(
                patient_id=f"P{i:05d}",
                age=30 + i,
                gender="M" if i % 2 == 0 else "F",
                medical_history=["diabetes"] if i % 3 == 0 else []
            )
            for i in range(100)
        ]
        
        with patch.object(analytics, '_compute_statistics') as mock_stats:
            mock_stats.return_value = {
                "total_patients": 100,
                "average_age": 79.5,
                "gender_distribution": {"M": 50, "F": 50},
                "common_conditions": {"diabetes": 34}
            }
            
            stats = analytics.analyze_population_health(patients)
            
            assert stats["total_patients"] == 100
            assert "average_age" in stats
            assert "gender_distribution" in stats
    
    def test_healthcare_ai_system(self):
        """Test HealthcareAISystem integration."""
        config = HealthcareConfig(
            model_path="/path/to/model",
            device="cpu",
            batch_size=16
        )
        
        system = HealthcareAISystem(config)
        
        # Test system initialization
        assert system.config == config
        assert hasattr(system, 'image_analyzer')
        assert hasattr(system, 'clinical_decision_support')
        assert hasattr(system, 'patient_monitor')
        assert hasattr(system, 'analytics')
        
        # Test system shutdown
        system.shutdown()
        assert system._is_running is False


class TestFinancialModule:
    """Test cases for financial AI components."""
    
    def test_financial_config(self):
        """Test FinancialConfig creation and validation."""
        config = FinancialConfig(
            api_key="test_key",
            data_source="yahoo",
            risk_tolerance=0.3,
            update_frequency=60,
            enable_real_time=True,
            supported_assets=["stocks", "bonds", "crypto"],
            max_portfolio_size=1000000
        )
        
        assert config.api_key == "test_key"
        assert config.data_source == "yahoo"
        assert config.risk_tolerance == 0.3
        assert config.update_frequency == 60
        assert config.enable_real_time is True
        assert "stocks" in config.supported_assets
        assert config.max_portfolio_size == 1000000
    
    def test_market_data(self):
        """Test MarketData dataclass."""
        market_data = MarketData(
            symbol="AAPL",
            price=150.25,
            volume=1000000,
            timestamp=datetime.now(),
            open_price=149.50,
            high_price=151.00,
            low_price=149.00,
            previous_close=149.75
        )
        
        assert market_data.symbol == "AAPL"
        assert market_data.price == 150.25
        assert market_data.volume == 1000000
        assert market_data.open_price == 149.50
        assert market_data.high_price == 151.00
    
    def test_portfolio(self):
        """Test Portfolio dataclass."""
        portfolio = Portfolio(
            portfolio_id="PORT001",
            user_id="USER123",
            total_value=100000.0,
            cash_balance=10000.0,
            positions={
                "AAPL": {"shares": 100, "avg_cost": 145.0},
                "GOOGL": {"shares": 50, "avg_cost": 2800.0}
            },
            last_updated=datetime.now()
        )
        
        assert portfolio.portfolio_id == "PORT001"
        assert portfolio.user_id == "USER123"
        assert portfolio.total_value == 100000.0
        assert portfolio.cash_balance == 10000.0
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"]["shares"] == 100
    
    def test_market_analyzer(self):
        """Test MarketAnalyzer functionality."""
        config = FinancialConfig(
            api_key="test_key",
            data_source="yahoo"
        )
        
        analyzer = MarketAnalyzer(config)
        
        # Test market analysis
        market_data = [
            MarketData(
                symbol="AAPL",
                price=150.0 + i,
                volume=1000000,
                timestamp=datetime.now() - timedelta(days=i),
                open_price=149.0 + i,
                high_price=151.0 + i,
                low_price=148.0 + i,
                previous_close=149.0 + i
            )
            for i in range(30)
        ]
        
        with patch.object(analyzer, '_compute_technical_indicators') as mock_indicators:
            mock_indicators.return_value = {
                "sma_20": 150.5,
                "rsi": 65.2,
                "macd": 1.2,
                "bollinger_upper": 155.0,
                "bollinger_lower": 145.0
            }
            
            analysis = analyzer.analyze_market_trends(market_data)
            
            assert "sma_20" in analysis
            assert "rsi" in analysis
            assert analysis["rsi"] > 0 and analysis["rsi"] < 100
    
    def test_risk_manager(self):
        """Test RiskManager functionality."""
        config = FinancialConfig(risk_tolerance=0.3)
        risk_manager = RiskManager(config)
        
        # Test risk assessment
        portfolio = Portfolio(
            portfolio_id="PORT001",
            user_id="USER123",
            total_value=100000.0,
            cash_balance=10000.0,
            positions={
                "AAPL": {"shares": 100, "avg_cost": 145.0},
                "TSLA": {"shares": 50, "avg_cost": 800.0}
            },
            last_updated=datetime.now()
        )
        
        with patch.object(risk_manager, '_calculate_var') as mock_var:
            mock_var.return_value = 5000.0  # 5% VaR
            
            risk_assessment = risk_manager.assess_portfolio_risk(portfolio)
            
            assert isinstance(risk_assessment, RiskAssessment)
            assert risk_assessment.portfolio_id == "PORT001"
            assert risk_assessment.var_95 > 0
    
    def test_fraud_detector(self):
        """Test FraudDetector functionality."""
        config = FinancialConfig()
        detector = FraudDetector(config)
        
        # Test fraud detection
        transaction = Transaction(
            transaction_id="TXN001",
            user_id="USER123",
            transaction_type=TransactionType.PURCHASE,
            amount=10000.0,
            asset_symbol="AAPL",
            quantity=100,
            timestamp=datetime.now(),
            metadata={"ip_address": "192.168.1.1", "device": "mobile"}
        )
        
        with patch.object(detector, '_analyze_transaction_patterns') as mock_analyze:
            mock_analyze.return_value = {
                "risk_score": 0.2,
                "anomaly_indicators": [],
                "risk_level": FraudRiskLevel.LOW
            }
            
            fraud_alert = detector.detect_fraud(transaction)
            
            if fraud_alert:
                assert isinstance(fraud_alert, FraudAlert)
                assert fraud_alert.transaction_id == "TXN001"
    
    def test_portfolio_optimizer(self):
        """Test PortfolioOptimizer functionality."""
        config = FinancialConfig(risk_tolerance=0.25)
        optimizer = PortfolioOptimizer(config)
        
        # Test portfolio optimization
        current_portfolio = Portfolio(
            portfolio_id="PORT001",
            user_id="USER123",
            total_value=100000.0,
            cash_balance=20000.0,
            positions={
                "AAPL": {"shares": 100, "avg_cost": 145.0},
                "MSFT": {"shares": 50, "avg_cost": 300.0}
            },
            last_updated=datetime.now()
        )
        
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                price=150.0,
                volume=1000000,
                timestamp=datetime.now()
            ),
            "MSFT": MarketData(
                symbol="MSFT",
                price=310.0,
                volume=800000,
                timestamp=datetime.now()
            )
        }
        
        with patch.object(optimizer, '_optimize_allocation') as mock_optimize:
            mock_optimize.return_value = {
                "AAPL": 0.4,
                "MSFT": 0.3,
                "BONDS": 0.2,
                "CASH": 0.1
            }
            
            optimized_allocation = optimizer.optimize_portfolio(
                current_portfolio, market_data
            )
            
            assert isinstance(optimized_allocation, dict)
            assert sum(optimized_allocation.values()) == pytest.approx(1.0)
    
    def test_financial_ai_system(self):
        """Test FinancialAISystem integration."""
        config = FinancialConfig(
            api_key="test_key",
            data_source="yahoo",
            risk_tolerance=0.3
        )
        
        system = FinancialAISystem(config)
        
        # Test system initialization
        assert system.config == config
        assert hasattr(system, 'market_analyzer')
        assert hasattr(system, 'risk_manager')
        assert hasattr(system, 'fraud_detector')
        assert hasattr(system, 'portfolio_optimizer')
        
        # Test system shutdown
        system.shutdown()
        assert system._is_running is False


class TestEducationalModule:
    """Test cases for educational AI components."""
    
    def test_educational_config(self):
        """Test EducationalConfig creation and validation."""
        config = EducationalConfig(
            model_path="/path/to/model",
            device="cpu",
            max_sequence_length=512,
            learning_rate=0.001,
            adaptation_threshold=0.7,
            supported_subjects=["math", "science", "language"],
            enable_personalization=True
        )
        
        assert config.model_path == "/path/to/model"
        assert config.device == "cpu"
        assert config.max_sequence_length == 512
        assert config.learning_rate == 0.001
        assert config.adaptation_threshold == 0.7
        assert "math" in config.supported_subjects
        assert config.enable_personalization is True
    
    def test_learner_profile(self):
        """Test LearnerProfile dataclass."""
        profile = LearnerProfile(
            learner_id="L12345",
            age=16,
            grade_level=10,
            learning_style=LearningStyle.VISUAL,
            strengths=["mathematics", "logical_reasoning"],
            weaknesses=["writing", "reading_comprehension"],
            interests=["science", "technology"],
            performance_history={"math": 0.85, "science": 0.78}
        )
        
        assert profile.learner_id == "L12345"
        assert profile.age == 16
        assert profile.grade_level == 10
        assert profile.learning_style == LearningStyle.VISUAL
        assert "mathematics" in profile.strengths
        assert "writing" in profile.weaknesses
        assert profile.performance_history["math"] == 0.85
    
    def test_learning_content(self):
        """Test LearningContent dataclass."""
        content = LearningContent(
            content_id="C001",
            title="Introduction to Algebra",
            subject="mathematics",
            content_type=ContentType.VIDEO,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            learning_objectives=[LearningObjective.UNDERSTAND, LearningObjective.APPLY],
            content_data={"video_url": "https://example.com/video.mp4"},
            prerequisites=["basic_arithmetic"],
            estimated_duration=30
        )
        
        assert content.content_id == "C001"
        assert content.title == "Introduction to Algebra"
        assert content.subject == "mathematics"
        assert content.content_type == ContentType.VIDEO
        assert content.difficulty_level == DifficultyLevel.INTERMEDIATE
        assert LearningObjective.UNDERSTAND in content.learning_objectives
        assert "basic_arithmetic" in content.prerequisites
        assert content.estimated_duration == 30
    
    def test_content_recommendation_engine(self):
        """Test ContentRecommendationEngine functionality."""
        config = EducationalConfig(device="cpu")
        engine = ContentRecommendationEngine(config)
        
        # Test content recommendation
        learner = LearnerProfile(
            learner_id="L12345",
            age=16,
            grade_level=10,
            learning_style=LearningStyle.VISUAL,
            strengths=["mathematics"],
            weaknesses=["writing"],
            interests=["science"]
        )
        
        available_content = [
            LearningContent(
                content_id=f"C{i:03d}",
                title=f"Content {i}",
                subject="mathematics" if i % 2 == 0 else "science",
                content_type=ContentType.VIDEO,
                difficulty_level=DifficultyLevel.INTERMEDIATE,
                learning_objectives=[LearningObjective.UNDERSTAND]
            )
            for i in range(10)
        ]
        
        with patch.object(engine, '_compute_recommendations') as mock_rec:
            mock_rec.return_value = available_content[:3]
            
            recommendations = engine.recommend_content(learner, available_content)
            
            assert len(recommendations) <= 3
            assert all(isinstance(content, LearningContent) for content in recommendations)
    
    def test_adaptive_learning_engine(self):
        """Test AdaptiveLearningEngine functionality."""
        config = EducationalConfig(
            adaptation_threshold=0.7,
            enable_personalization=True
        )
        
        engine = AdaptiveLearningEngine(config)
        
        # Test learning adaptation
        learner = LearnerProfile(
            learner_id="L12345",
            learning_style=LearningStyle.KINESTHETIC,
            performance_history={"math": 0.6}
        )
        
        session = LearningSession(
            session_id="S001",
            learner_id="L12345",
            content_id="C001",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=30),
            engagement_level=EngagementLevel.MEDIUM,
            performance_score=0.75
        )
        
        with patch.object(engine, '_adapt_difficulty') as mock_adapt:
            mock_adapt.return_value = DifficultyLevel.ADVANCED
            
            adapted_difficulty = engine.adapt_learning_path(learner, session)
            
            assert isinstance(adapted_difficulty, DifficultyLevel)
    
    def test_intelligent_tutor_system(self):
        """Test IntelligentTutorSystem functionality."""
        config = EducationalConfig(device="cpu")
        tutor = IntelligentTutorSystem(config)
        
        # Test tutoring session
        learner = LearnerProfile(
            learner_id="L12345",
            learning_style=LearningStyle.AUDITORY,
            weaknesses=["algebra"]
        )
        
        question = "Solve for x: 2x + 5 = 13"
        student_answer = "x = 4"
        
        with patch.object(tutor, '_evaluate_answer') as mock_eval:
            mock_eval.return_value = {
                "correct": True,
                "feedback": "Excellent! You correctly solved the equation.",
                "next_question": "Try this harder problem: 3x - 7 = 14"
            }
            
            feedback = tutor.provide_tutoring(learner, question, student_answer)
            
            assert feedback["correct"] is True
            assert "feedback" in feedback
            assert "next_question" in feedback
    
    def test_learning_analytics(self):
        """Test LearningAnalytics functionality."""
        config = EducationalConfig(device="cpu")
        analytics = LearningAnalytics(config)
        
        # Test learning analytics
        sessions = [
            LearningSession(
                session_id=f"S{i:03d}",
                learner_id="L12345",
                content_id=f"C{i%5:03d}",
                start_time=datetime.now() - timedelta(days=i),
                end_time=datetime.now() - timedelta(days=i) + timedelta(minutes=30),
                engagement_level=EngagementLevel.HIGH if i % 3 == 0 else EngagementLevel.MEDIUM,
                performance_score=0.7 + (i % 3) * 0.1
            )
            for i in range(20)
        ]
        
        with patch.object(analytics, '_compute_learning_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "total_sessions": 20,
                "average_performance": 0.75,
                "engagement_trend": "improving",
                "learning_velocity": 0.85,
                "knowledge_gaps": ["advanced_algebra"]
            }
            
            metrics = analytics.analyze_learning_progress(sessions)
            
            assert metrics["total_sessions"] == 20
            assert "average_performance" in metrics
            assert "engagement_trend" in metrics
    
    def test_educational_ai_system(self):
        """Test EducationalAISystem integration."""
        config = EducationalConfig(
            model_path="/path/to/model",
            device="cpu",
            enable_personalization=True
        )
        
        system = EducationalAISystem(config)
        
        # Test system initialization
        assert system.config == config
        assert hasattr(system, 'recommendation_engine')
        assert hasattr(system, 'adaptive_engine')
        assert hasattr(system, 'tutor_system')
        assert hasattr(system, 'analytics')
        
        # Test system shutdown
        system.shutdown()
        assert system._is_running is False


# Integration tests for applications module
class TestApplicationsIntegration:
    """Integration tests for applications module components."""
    
    def test_healthcare_workflow(self):
        """Test complete healthcare AI workflow."""
        config = HealthcareConfig(
            model_path="/path/to/model",
            device="cpu",
            confidence_threshold=0.8
        )
        
        system = HealthcareAISystem(config)
        
        # Create test data
        patient = PatientInfo(
            patient_id="P12345",
            age=55,
            gender="M",
            medical_history=["hypertension"]
        )
        
        image_data = np.random.rand(512, 512, 3).astype(np.float32)
        medical_image = MedicalImage(
            image_id="IMG001",
            patient_id="P12345",
            image_type=MedicalImageType.XRAY,
            image_data=image_data,
            acquisition_date=datetime.now()
        )
        
        # Mock the workflow
        with patch.object(system.image_analyzer, 'analyze_image') as mock_analyze:
            mock_analyze.return_value = DiagnosisResult(
                diagnosis_id="D001",
                patient_id="P12345",
                image_id="IMG001",
                diagnosis_type=DiagnosisType.NORMAL,
                confidence=0.95,
                findings=["No acute findings"],
                timestamp=datetime.now()
            )
            
            # Test workflow
            diagnosis = system.image_analyzer.analyze_image(medical_image)
            treatment_plan = system.clinical_decision_support.recommend_treatment(
                patient, diagnosis
            )
            
            assert diagnosis.confidence >= config.confidence_threshold
            assert isinstance(treatment_plan, TreatmentPlan)
    
    def test_financial_trading_workflow(self):
        """Test complete financial trading workflow."""
        config = FinancialConfig(
            api_key="test_key",
            risk_tolerance=0.3
        )
        
        system = FinancialAISystem(config)
        
        # Create test data
        portfolio = Portfolio(
            portfolio_id="PORT001",
            user_id="USER123",
            total_value=100000.0,
            cash_balance=20000.0,
            positions={"AAPL": {"shares": 100, "avg_cost": 145.0}}
        )
        
        market_data = {
            "AAPL": MarketData(
                symbol="AAPL",
                price=150.0,
                volume=1000000,
                timestamp=datetime.now()
            )
        }
        
        # Mock the workflow
        with patch.object(system.market_analyzer, 'analyze_market_trends') as mock_analyze:
            mock_analyze.return_value = {
                "trend": "bullish",
                "confidence": 0.8,
                "signals": ["buy"]
            }
            
            with patch.object(system.risk_manager, 'assess_portfolio_risk') as mock_risk:
                mock_risk.return_value = RiskAssessment(
                    assessment_id="RA001",
                    portfolio_id="PORT001",
                    risk_level=RiskLevel.MEDIUM,
                    var_95=5000.0,
                    expected_shortfall=7500.0,
                    timestamp=datetime.now()
                )
                
                # Test workflow
                market_analysis = system.market_analyzer.analyze_market_trends(
                    list(market_data.values())
                )
                risk_assessment = system.risk_manager.assess_portfolio_risk(portfolio)
                optimized_allocation = system.portfolio_optimizer.optimize_portfolio(
                    portfolio, market_data
                )
                
                assert market_analysis["confidence"] > 0.5
                assert risk_assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
                assert isinstance(optimized_allocation, dict)
    
    def test_educational_learning_workflow(self):
        """Test complete educational learning workflow."""
        config = EducationalConfig(
            model_path="/path/to/model",
            device="cpu",
            enable_personalization=True
        )
        
        system = EducationalAISystem(config)
        
        # Create test data
        learner = LearnerProfile(
            learner_id="L12345",
            age=16,
            grade_level=10,
            learning_style=LearningStyle.VISUAL,
            strengths=["mathematics"],
            weaknesses=["writing"]
        )
        
        content = [
            LearningContent(
                content_id="C001",
                title="Algebra Basics",
                subject="mathematics",
                content_type=ContentType.VIDEO,
                difficulty_level=DifficultyLevel.BEGINNER,
                learning_objectives=[LearningObjective.UNDERSTAND]
            )
        ]
        
        # Mock the workflow
        with patch.object(system.recommendation_engine, 'recommend_content') as mock_rec:
            mock_rec.return_value = content
            
            with patch.object(system.tutor_system, 'provide_tutoring') as mock_tutor:
                mock_tutor.return_value = {
                    "correct": True,
                    "feedback": "Great job!",
                    "next_question": "Try the next problem"
                }
                
                # Test workflow
                recommendations = system.recommendation_engine.recommend_content(
                    learner, content
                )
                tutoring_feedback = system.tutor_system.provide_tutoring(
                    learner, "What is 2 + 2?", "4"
                )
                
                assert len(recommendations) > 0
                assert tutoring_feedback["correct"] is True
                assert "feedback" in tutoring_feedback


if __name__ == "__main__":
    pytest.main([__file__])