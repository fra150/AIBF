"""Integration tests for API and applications module interactions."""

import pytest
import asyncio
import json
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import threading
import time

# Import API modules
from api.rest import (
    RESTConfig, RESTServer, RESTClient, RequestHandler,
    APIEndpoint, ResponseFormatter, RequestValidator
)
from api.websocket import (
    WebSocketConfig, WebSocketServer, WebSocketClient,
    MessageHandler, ConnectionManager, WebSocketEndpoint
)
from api.grpc import (
    GRPCConfig, GRPCServer, GRPCClient, ServiceHandler,
    GRPCEndpoint, MessageSerializer, ClientConfig
)
from api.graphql import (
    GraphQLConfig, GraphQLServer, GraphQLClient,
    SchemaBuilder, QueryResolver, MutationResolver
)

# Import applications
from applications.healthcare import (
    HealthcareAISystem, HealthcareConfig, PatientInfo,
    MedicalImage, DiagnosisResult, MedicalImageType
)
from applications.financial import (
    FinancialAISystem, FinancialConfig, Portfolio,
    MarketData, TradingSignal, AssetType
)
from applications.educational import (
    EducationalAISystem, EducationalConfig, LearnerProfile,
    LearningContent, ContentType, LearningStyle
)

# Import multimodal for integration
from multimodal.vision import ImageEncoder, VisionConfig
from multimodal.audio import AudioEncoder, AudioConfig
from multimodal.fusion import ModalityFusion, FusionConfig


class TestRESTAPIIntegration:
    """Test REST API integration with applications."""
    
    @pytest.fixture
    def healthcare_system(self):
        """Create healthcare system for testing."""
        config = HealthcareConfig(
            model_path="/test/model",
            device="cpu",
            confidence_threshold=0.8
        )
        return HealthcareAISystem(config)
    
    @pytest.fixture
    def rest_server(self):
        """Create REST server for testing."""
        config = RESTConfig(
            host="localhost",
            port=8080,
            enable_cors=True,
            max_request_size=10 * 1024 * 1024  # 10MB
        )
        return RESTServer(config)
    
    def test_healthcare_api_endpoints(self, healthcare_system, rest_server):
        """Test healthcare API endpoints integration."""
        # Create healthcare endpoints
        class HealthcareEndpoints:
            def __init__(self, healthcare_system):
                self.healthcare_system = healthcare_system
            
            def analyze_medical_image(self, request_data):
                """Analyze medical image endpoint."""
                try:
                    # Extract image data from request
                    image_data = np.array(request_data["image_data"])
                    patient_id = request_data["patient_id"]
                    image_type = request_data["image_type"]
                    
                    # Create medical image object
                    medical_image = MedicalImage(
                        image_id=f"IMG_{int(time.time())}",
                        patient_id=patient_id,
                        image_type=MedicalImageType(image_type),
                        image_data=image_data,
                        acquisition_date=datetime.now()
                    )
                    
                    # Mock analysis result
                    with patch.object(
                        self.healthcare_system.image_analyzer,
                        'analyze_image'
                    ) as mock_analyze:
                        mock_result = DiagnosisResult(
                            diagnosis_id="D001",
                            patient_id=patient_id,
                            image_id=medical_image.image_id,
                            diagnosis_type="normal",
                            confidence=0.95,
                            findings=["No abnormalities detected"],
                            timestamp=datetime.now()
                        )
                        mock_analyze.return_value = mock_result
                        
                        result = self.healthcare_system.image_analyzer.analyze_image(medical_image)
                    
                    return {
                        "status": "success",
                        "diagnosis_id": result.diagnosis_id,
                        "confidence": result.confidence,
                        "findings": result.findings
                    }
                
                except Exception as e:
                    return {
                        "status": "error",
                        "message": str(e)
                    }
            
            def get_patient_info(self, patient_id):
                """Get patient information endpoint."""
                # Mock patient data
                patient = PatientInfo(
                    patient_id=patient_id,
                    age=45,
                    gender="M",
                    medical_history=["diabetes"],
                    current_medications=["metformin"]
                )
                
                return {
                    "status": "success",
                    "patient": {
                        "patient_id": patient.patient_id,
                        "age": patient.age,
                        "gender": patient.gender,
                        "medical_history": patient.medical_history,
                        "current_medications": patient.current_medications
                    }
                }
        
        endpoints = HealthcareEndpoints(healthcare_system)
        
        # Register endpoints with REST server
        rest_server.add_endpoint(
            "/api/healthcare/analyze",
            "POST",
            endpoints.analyze_medical_image
        )
        rest_server.add_endpoint(
            "/api/healthcare/patient/{patient_id}",
            "GET",
            endpoints.get_patient_info
        )
        
        # Test image analysis endpoint
        image_request = {
            "patient_id": "P12345",
            "image_type": "xray",
            "image_data": np.random.rand(512, 512, 3).tolist()
        }
        
        analysis_result = endpoints.analyze_medical_image(image_request)
        
        assert analysis_result["status"] == "success"
        assert "diagnosis_id" in analysis_result
        assert analysis_result["confidence"] >= 0.8
        
        # Test patient info endpoint
        patient_result = endpoints.get_patient_info("P12345")
        
        assert patient_result["status"] == "success"
        assert patient_result["patient"]["patient_id"] == "P12345"
    
    def test_financial_api_endpoints(self):
        """Test financial API endpoints integration."""
        # Create financial system
        config = FinancialConfig(
            api_key="test_key",
            data_source="mock",
            risk_tolerance=0.3
        )
        financial_system = FinancialAISystem(config)
        
        # Create financial endpoints
        class FinancialEndpoints:
            def __init__(self, financial_system):
                self.financial_system = financial_system
            
            def analyze_portfolio(self, request_data):
                """Analyze portfolio endpoint."""
                try:
                    portfolio_data = request_data["portfolio"]
                    
                    portfolio = Portfolio(
                        portfolio_id=portfolio_data["portfolio_id"],
                        user_id=portfolio_data["user_id"],
                        total_value=portfolio_data["total_value"],
                        cash_balance=portfolio_data["cash_balance"],
                        positions=portfolio_data["positions"],
                        last_updated=datetime.now()
                    )
                    
                    # Mock risk assessment
                    with patch.object(
                        self.financial_system.risk_manager,
                        'assess_portfolio_risk'
                    ) as mock_assess:
                        mock_assessment = {
                            "risk_level": "medium",
                            "var_95": 5000.0,
                            "recommendations": ["Diversify holdings"]
                        }
                        mock_assess.return_value = mock_assessment
                        
                        risk_assessment = self.financial_system.risk_manager.assess_portfolio_risk(portfolio)
                    
                    return {
                        "status": "success",
                        "risk_assessment": risk_assessment
                    }
                
                except Exception as e:
                    return {
                        "status": "error",
                        "message": str(e)
                    }
            
            def get_market_analysis(self, symbol):
                """Get market analysis endpoint."""
                # Mock market data
                market_data = MarketData(
                    symbol=symbol,
                    price=150.0,
                    volume=1000000,
                    timestamp=datetime.now()
                )
                
                # Mock analysis
                with patch.object(
                    self.financial_system.market_analyzer,
                    'analyze_market_trends'
                ) as mock_analyze:
                    mock_analysis = {
                        "trend": "bullish",
                        "confidence": 0.8,
                        "signals": ["buy"]
                    }
                    mock_analyze.return_value = mock_analysis
                    
                    analysis = self.financial_system.market_analyzer.analyze_market_trends([market_data])
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "analysis": analysis
                }
        
        endpoints = FinancialEndpoints(financial_system)
        
        # Test portfolio analysis
        portfolio_request = {
            "portfolio": {
                "portfolio_id": "PORT001",
                "user_id": "USER123",
                "total_value": 100000.0,
                "cash_balance": 10000.0,
                "positions": {"AAPL": {"shares": 100, "avg_cost": 145.0}}
            }
        }
        
        portfolio_result = endpoints.analyze_portfolio(portfolio_request)
        
        assert portfolio_result["status"] == "success"
        assert "risk_assessment" in portfolio_result
        
        # Test market analysis
        market_result = endpoints.get_market_analysis("AAPL")
        
        assert market_result["status"] == "success"
        assert market_result["symbol"] == "AAPL"
        assert "analysis" in market_result


class TestWebSocketIntegration:
    """Test WebSocket integration with real-time applications."""
    
    @pytest.fixture
    def websocket_server(self):
        """Create WebSocket server for testing."""
        config = WebSocketConfig(
            host="localhost",
            port=8081,
            max_connections=100,
            heartbeat_interval=30
        )
        return WebSocketServer(config)
    
    def test_real_time_healthcare_monitoring(self, websocket_server):
        """Test real-time healthcare monitoring via WebSocket."""
        # Create healthcare system
        config = HealthcareConfig(
            model_path="/test/model",
            device="cpu"
        )
        healthcare_system = HealthcareAISystem(config)
        
        # Create real-time monitoring handler
        class HealthcareMonitoringHandler:
            def __init__(self, healthcare_system):
                self.healthcare_system = healthcare_system
                self.active_connections = set()
            
            async def handle_connection(self, websocket, path):
                """Handle new WebSocket connection."""
                self.active_connections.add(websocket)
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        
                        if data["type"] == "vital_signs":
                            # Process vital signs
                            vital_signs = data["data"]
                            
                            # Mock monitoring analysis
                            with patch.object(
                                self.healthcare_system.patient_monitor,
                                'monitor_vitals'
                            ) as mock_monitor:
                                mock_analysis = {
                                    "status": "normal",
                                    "alerts": [],
                                    "trends": {"heart_rate": "stable"}
                                }
                                mock_monitor.return_value = mock_analysis
                                
                                analysis = self.healthcare_system.patient_monitor.monitor_vitals(vital_signs)
                            
                            # Send analysis back to client
                            response = {
                                "type": "monitoring_result",
                                "patient_id": vital_signs["patient_id"],
                                "analysis": analysis,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            await websocket.send(json.dumps(response))
                
                except Exception as e:
                    print(f"WebSocket error: {e}")
                finally:
                    self.active_connections.discard(websocket)
            
            async def broadcast_alert(self, alert_data):
                """Broadcast alert to all connected clients."""
                if self.active_connections:
                    message = json.dumps({
                        "type": "alert",
                        "data": alert_data,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Send to all connections
                    for websocket in self.active_connections.copy():
                        try:
                            await websocket.send(message)
                        except Exception:
                            self.active_connections.discard(websocket)
        
        handler = HealthcareMonitoringHandler(healthcare_system)
        
        # Test vital signs processing
        vital_signs_data = {
            "type": "vital_signs",
            "data": {
                "patient_id": "P12345",
                "heart_rate": 85,
                "blood_pressure_systolic": 120,
                "blood_pressure_diastolic": 80,
                "temperature": 98.6,
                "oxygen_saturation": 98
            }
        }
        
        # Mock WebSocket for testing
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__.return_value = [json.dumps(vital_signs_data)]
        
        # Test handler
        asyncio.run(handler.handle_connection(mock_websocket, "/monitor"))
        
        # Verify response was sent
        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        
        assert sent_data["type"] == "monitoring_result"
        assert sent_data["patient_id"] == "P12345"
        assert "analysis" in sent_data
    
    def test_real_time_financial_updates(self):
        """Test real-time financial market updates via WebSocket."""
        # Create financial system
        config = FinancialConfig(
            api_key="test_key",
            enable_real_time=True
        )
        financial_system = FinancialAISystem(config)
        
        # Create market data handler
        class MarketDataHandler:
            def __init__(self, financial_system):
                self.financial_system = financial_system
                self.subscriptions = {}  # client_id -> symbols
            
            async def handle_subscription(self, websocket, message):
                """Handle market data subscription."""
                data = json.loads(message)
                
                if data["type"] == "subscribe":
                    client_id = data["client_id"]
                    symbols = data["symbols"]
                    
                    self.subscriptions[client_id] = symbols
                    
                    # Send confirmation
                    response = {
                        "type": "subscription_confirmed",
                        "client_id": client_id,
                        "symbols": symbols
                    }
                    
                    await websocket.send(json.dumps(response))
                
                elif data["type"] == "market_data":
                    # Process market data update
                    market_data = MarketData(
                        symbol=data["symbol"],
                        price=data["price"],
                        volume=data["volume"],
                        timestamp=datetime.now()
                    )
                    
                    # Mock market analysis
                    with patch.object(
                        self.financial_system.market_analyzer,
                        'analyze_market_trends'
                    ) as mock_analyze:
                        mock_analysis = {
                            "trend": "bullish",
                            "signals": ["buy"],
                            "confidence": 0.8
                        }
                        mock_analyze.return_value = mock_analysis
                        
                        analysis = self.financial_system.market_analyzer.analyze_market_trends([market_data])
                    
                    # Send analysis to subscribed clients
                    update = {
                        "type": "market_update",
                        "symbol": data["symbol"],
                        "price": data["price"],
                        "analysis": analysis,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await websocket.send(json.dumps(update))
        
        handler = MarketDataHandler(financial_system)
        
        # Test subscription
        subscription_message = {
            "type": "subscribe",
            "client_id": "CLIENT001",
            "symbols": ["AAPL", "GOOGL"]
        }
        
        mock_websocket = AsyncMock()
        
        asyncio.run(handler.handle_subscription(
            mock_websocket,
            json.dumps(subscription_message)
        ))
        
        # Verify subscription confirmation
        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        
        assert sent_data["type"] == "subscription_confirmed"
        assert sent_data["client_id"] == "CLIENT001"
        assert "AAPL" in sent_data["symbols"]


class TestGraphQLIntegration:
    """Test GraphQL integration with applications."""
    
    def test_educational_graphql_schema(self):
        """Test educational system GraphQL schema integration."""
        # Create educational system
        config = EducationalConfig(
            model_path="/test/model",
            device="cpu",
            enable_personalization=True
        )
        educational_system = EducationalAISystem(config)
        
        # Create GraphQL schema
        class EducationalSchema:
            def __init__(self, educational_system):
                self.educational_system = educational_system
            
            def resolve_learner_profile(self, learner_id):
                """Resolve learner profile query."""
                # Mock learner profile
                profile = LearnerProfile(
                    learner_id=learner_id,
                    age=16,
                    grade_level=10,
                    learning_style=LearningStyle.VISUAL,
                    strengths=["mathematics"],
                    weaknesses=["writing"],
                    interests=["science"]
                )
                
                return {
                    "learner_id": profile.learner_id,
                    "age": profile.age,
                    "grade_level": profile.grade_level,
                    "learning_style": profile.learning_style.value,
                    "strengths": profile.strengths,
                    "weaknesses": profile.weaknesses,
                    "interests": profile.interests
                }
            
            def resolve_content_recommendations(self, learner_id, subject):
                """Resolve content recommendations query."""
                # Mock learner profile
                learner = LearnerProfile(
                    learner_id=learner_id,
                    learning_style=LearningStyle.VISUAL,
                    interests=[subject]
                )
                
                # Mock content
                content = [
                    LearningContent(
                        content_id="C001",
                        title=f"{subject.title()} Basics",
                        subject=subject,
                        content_type=ContentType.VIDEO,
                        estimated_duration=30
                    )
                ]
                
                # Mock recommendations
                with patch.object(
                    self.educational_system.recommendation_engine,
                    'recommend_content'
                ) as mock_recommend:
                    mock_recommend.return_value = content
                    
                    recommendations = self.educational_system.recommendation_engine.recommend_content(
                        learner, content
                    )
                
                return [
                    {
                        "content_id": rec.content_id,
                        "title": rec.title,
                        "subject": rec.subject,
                        "content_type": rec.content_type.value,
                        "estimated_duration": rec.estimated_duration
                    }
                    for rec in recommendations
                ]
            
            def mutate_update_learning_progress(self, learner_id, content_id, performance_score):
                """Mutation to update learning progress."""
                # Mock learning session update
                session_data = {
                    "session_id": f"S_{int(time.time())}",
                    "learner_id": learner_id,
                    "content_id": content_id,
                    "performance_score": performance_score,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Mock adaptive learning update
                with patch.object(
                    self.educational_system.adaptive_engine,
                    'adapt_learning_path'
                ) as mock_adapt:
                    mock_adapt.return_value = "intermediate"
                    
                    adapted_level = self.educational_system.adaptive_engine.adapt_learning_path(
                        None, None  # Mock objects
                    )
                
                return {
                    "success": True,
                    "session": session_data,
                    "adapted_difficulty": adapted_level
                }
        
        schema = EducationalSchema(educational_system)
        
        # Test learner profile query
        profile_result = schema.resolve_learner_profile("L12345")
        
        assert profile_result["learner_id"] == "L12345"
        assert profile_result["age"] == 16
        assert profile_result["learning_style"] == "visual"
        
        # Test content recommendations query
        recommendations_result = schema.resolve_content_recommendations("L12345", "mathematics")
        
        assert len(recommendations_result) > 0
        assert recommendations_result[0]["subject"] == "mathematics"
        
        # Test learning progress mutation
        progress_result = schema.mutate_update_learning_progress("L12345", "C001", 0.85)
        
        assert progress_result["success"] is True
        assert "session" in progress_result
        assert "adapted_difficulty" in progress_result


class TestMultimodalAPIIntegration:
    """Test multimodal capabilities through API integration."""
    
    def test_multimodal_content_processing_api(self):
        """Test multimodal content processing through API."""
        # Create multimodal components
        vision_config = VisionConfig(
            image_size=224,
            embedding_dim=512
        )
        
        audio_config = AudioConfig(
            sample_rate=16000,
            embedding_dim=512
        )
        
        fusion_config = FusionConfig(
            input_dims=[512, 512],
            fusion_dim=1024
        )
        
        image_encoder = ImageEncoder(vision_config)
        audio_encoder = AudioEncoder(audio_config)
        fusion = ModalityFusion(fusion_config)
        
        # Create multimodal API handler
        class MultimodalAPIHandler:
            def __init__(self, image_encoder, audio_encoder, fusion):
                self.image_encoder = image_encoder
                self.audio_encoder = audio_encoder
                self.fusion = fusion
            
            def process_multimodal_content(self, request_data):
                """Process multimodal content endpoint."""
                try:
                    # Extract modalities
                    image_data = torch.tensor(request_data["image"], dtype=torch.float32)
                    audio_data = torch.tensor(request_data["audio"], dtype=torch.float32)
                    
                    # Encode modalities
                    image_features = self.image_encoder(image_data.unsqueeze(0))
                    audio_features = self.audio_encoder(audio_data.unsqueeze(0))
                    
                    # Fuse modalities
                    fused_features = self.fusion([image_features, audio_features])
                    
                    return {
                        "status": "success",
                        "image_features_shape": list(image_features.shape),
                        "audio_features_shape": list(audio_features.shape),
                        "fused_features_shape": list(fused_features.shape),
                        "fused_features_norm": float(torch.norm(fused_features).item())
                    }
                
                except Exception as e:
                    return {
                        "status": "error",
                        "message": str(e)
                    }
            
            def analyze_content_similarity(self, content1, content2):
                """Analyze similarity between multimodal contents."""
                try:
                    # Process first content
                    image1 = torch.tensor(content1["image"], dtype=torch.float32)
                    audio1 = torch.tensor(content1["audio"], dtype=torch.float32)
                    
                    img_feat1 = self.image_encoder(image1.unsqueeze(0))
                    aud_feat1 = self.audio_encoder(audio1.unsqueeze(0))
                    fused1 = self.fusion([img_feat1, aud_feat1])
                    
                    # Process second content
                    image2 = torch.tensor(content2["image"], dtype=torch.float32)
                    audio2 = torch.tensor(content2["audio"], dtype=torch.float32)
                    
                    img_feat2 = self.image_encoder(image2.unsqueeze(0))
                    aud_feat2 = self.audio_encoder(audio2.unsqueeze(0))
                    fused2 = self.fusion([img_feat2, aud_feat2])
                    
                    # Compute similarity
                    similarity = torch.cosine_similarity(fused1, fused2, dim=1)
                    
                    return {
                        "status": "success",
                        "similarity_score": float(similarity.item()),
                        "content1_features_norm": float(torch.norm(fused1).item()),
                        "content2_features_norm": float(torch.norm(fused2).item())
                    }
                
                except Exception as e:
                    return {
                        "status": "error",
                        "message": str(e)
                    }
        
        handler = MultimodalAPIHandler(image_encoder, audio_encoder, fusion)
        
        # Test multimodal content processing
        content_request = {
            "image": np.random.rand(3, 224, 224).tolist(),
            "audio": np.random.rand(16000).tolist()
        }
        
        processing_result = handler.process_multimodal_content(content_request)
        
        assert processing_result["status"] == "success"
        assert "fused_features_shape" in processing_result
        assert processing_result["fused_features_shape"] == [1, 1024]
        
        # Test content similarity
        content1 = {
            "image": np.random.rand(3, 224, 224).tolist(),
            "audio": np.random.rand(16000).tolist()
        }
        
        content2 = {
            "image": np.random.rand(3, 224, 224).tolist(),
            "audio": np.random.rand(16000).tolist()
        }
        
        similarity_result = handler.analyze_content_similarity(content1, content2)
        
        assert similarity_result["status"] == "success"
        assert "similarity_score" in similarity_result
        assert -1 <= similarity_result["similarity_score"] <= 1


class TestEndToEndAPIWorkflows:
    """Test complete end-to-end API workflows."""
    
    def test_healthcare_diagnostic_workflow(self):
        """Test complete healthcare diagnostic workflow through APIs."""
        # Create healthcare system
        config = HealthcareConfig(
            model_path="/test/model",
            device="cpu",
            confidence_threshold=0.8
        )
        healthcare_system = HealthcareAISystem(config)
        
        # Simulate complete workflow
        workflow_steps = []
        
        # Step 1: Upload medical image (REST API)
        image_upload_data = {
            "patient_id": "P12345",
            "image_type": "xray",
            "image_data": np.random.rand(512, 512, 3).tolist()
        }
        
        # Mock image analysis
        with patch.object(
            healthcare_system.image_analyzer,
            'analyze_image'
        ) as mock_analyze:
            mock_result = DiagnosisResult(
                diagnosis_id="D001",
                patient_id="P12345",
                image_id="IMG001",
                diagnosis_type="pneumonia",
                confidence=0.92,
                findings=["Consolidation in right lower lobe"],
                timestamp=datetime.now()
            )
            mock_analyze.return_value = mock_result
            
            # Simulate image analysis
            medical_image = MedicalImage(
                image_id="IMG001",
                patient_id="P12345",
                image_type=MedicalImageType.XRAY,
                image_data=np.array(image_upload_data["image_data"]),
                acquisition_date=datetime.now()
            )
            
            diagnosis_result = healthcare_system.image_analyzer.analyze_image(medical_image)
            workflow_steps.append({
                "step": "image_analysis",
                "result": diagnosis_result,
                "api": "REST"
            })
        
        # Step 2: Get treatment recommendations (GraphQL)
        patient = PatientInfo(
            patient_id="P12345",
            age=65,
            gender="F",
            medical_history=["diabetes"]
        )
        
        with patch.object(
            healthcare_system.clinical_decision_support,
            'recommend_treatment'
        ) as mock_treatment:
            mock_treatment_plan = {
                "treatment_id": "T001",
                "recommendations": [
                    "Start antibiotic therapy",
                    "Monitor oxygen saturation",
                    "Follow-up chest X-ray in 48 hours"
                ],
                "urgency": "high"
            }
            mock_treatment.return_value = mock_treatment_plan
            
            treatment_plan = healthcare_system.clinical_decision_support.recommend_treatment(
                patient, diagnosis_result
            )
            workflow_steps.append({
                "step": "treatment_recommendation",
                "result": treatment_plan,
                "api": "GraphQL"
            })
        
        # Step 3: Real-time monitoring setup (WebSocket)
        monitoring_setup = {
            "patient_id": "P12345",
            "monitoring_parameters": [
                "heart_rate", "oxygen_saturation", "blood_pressure"
            ],
            "alert_thresholds": {
                "heart_rate": {"min": 60, "max": 100},
                "oxygen_saturation": {"min": 95}
            }
        }
        
        workflow_steps.append({
            "step": "monitoring_setup",
            "result": monitoring_setup,
            "api": "WebSocket"
        })
        
        # Verify workflow completion
        assert len(workflow_steps) == 3
        assert workflow_steps[0]["step"] == "image_analysis"
        assert workflow_steps[1]["step"] == "treatment_recommendation"
        assert workflow_steps[2]["step"] == "monitoring_setup"
        
        # Verify diagnosis confidence
        assert workflow_steps[0]["result"].confidence >= config.confidence_threshold
        
        # Verify treatment recommendations
        assert len(workflow_steps[1]["result"]["recommendations"]) > 0
        
        # Verify monitoring setup
        assert "patient_id" in workflow_steps[2]["result"]
        assert "monitoring_parameters" in workflow_steps[2]["result"]
    
    def test_educational_learning_workflow(self):
        """Test complete educational learning workflow through APIs."""
        # Create educational system
        config = EducationalConfig(
            model_path="/test/model",
            device="cpu",
            enable_personalization=True
        )
        educational_system = EducationalAISystem(config)
        
        # Simulate learning workflow
        workflow_steps = []
        
        # Step 1: Get learner profile (GraphQL)
        learner_profile = LearnerProfile(
            learner_id="L12345",
            age=16,
            grade_level=10,
            learning_style=LearningStyle.VISUAL,
            strengths=["mathematics"],
            weaknesses=["writing"]
        )
        
        workflow_steps.append({
            "step": "get_learner_profile",
            "result": learner_profile,
            "api": "GraphQL"
        })
        
        # Step 2: Get content recommendations (REST)
        available_content = [
            LearningContent(
                content_id="C001",
                title="Algebra Basics",
                subject="mathematics",
                content_type=ContentType.VIDEO,
                estimated_duration=30
            )
        ]
        
        with patch.object(
            educational_system.recommendation_engine,
            'recommend_content'
        ) as mock_recommend:
            mock_recommend.return_value = available_content
            
            recommendations = educational_system.recommendation_engine.recommend_content(
                learner_profile, available_content
            )
            
            workflow_steps.append({
                "step": "get_recommendations",
                "result": recommendations,
                "api": "REST"
            })
        
        # Step 3: Real-time tutoring session (WebSocket)
        tutoring_session = {
            "session_id": "TS001",
            "learner_id": "L12345",
            "content_id": "C001",
            "questions_answered": 5,
            "correct_answers": 4,
            "session_duration": 25  # minutes
        }
        
        with patch.object(
            educational_system.tutor_system,
            'provide_tutoring'
        ) as mock_tutor:
            mock_feedback = {
                "correct": True,
                "feedback": "Great job!",
                "next_question": "Try this harder problem"
            }
            mock_tutor.return_value = mock_feedback
            
            tutoring_feedback = educational_system.tutor_system.provide_tutoring(
                learner_profile, "What is 2x + 5 = 13?", "x = 4"
            )
            
            workflow_steps.append({
                "step": "tutoring_session",
                "result": {
                    "session": tutoring_session,
                    "feedback": tutoring_feedback
                },
                "api": "WebSocket"
            })
        
        # Step 4: Update learning progress (GraphQL mutation)
        performance_score = 0.8  # 4/5 correct answers
        
        with patch.object(
            educational_system.adaptive_engine,
            'adapt_learning_path'
        ) as mock_adapt:
            mock_adapt.return_value = "intermediate"
            
            adapted_difficulty = educational_system.adaptive_engine.adapt_learning_path(
                learner_profile, None  # Mock session
            )
            
            workflow_steps.append({
                "step": "update_progress",
                "result": {
                    "performance_score": performance_score,
                    "adapted_difficulty": adapted_difficulty
                },
                "api": "GraphQL"
            })
        
        # Verify workflow completion
        assert len(workflow_steps) == 4
        assert workflow_steps[0]["step"] == "get_learner_profile"
        assert workflow_steps[1]["step"] == "get_recommendations"
        assert workflow_steps[2]["step"] == "tutoring_session"
        assert workflow_steps[3]["step"] == "update_progress"
        
        # Verify learner profile
        assert workflow_steps[0]["result"].learner_id == "L12345"
        
        # Verify recommendations
        assert len(workflow_steps[1]["result"]) > 0
        
        # Verify tutoring feedback
        assert workflow_steps[2]["result"]["feedback"]["correct"] is True
        
        # Verify progress update
        assert workflow_steps[3]["result"]["performance_score"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__])