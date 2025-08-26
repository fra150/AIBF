"""Unit tests for multimodal module components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io
from typing import Dict, Any, List, Tuple

# Import multimodal modules
from multimodal.vision import (
    VisionConfig, ImageEncoder, VideoProcessor, DiffusionModel,
    ObjectDetector, ImageClassifier, SemanticSegmentation,
    VisionTransformer, ConvolutionalEncoder
)
from multimodal.audio import (
    AudioConfig, SpeechRecognition, AudioEncoder, TextToSpeech,
    AudioClassifier, SpeechSynthesis, AudioFeatureExtractor,
    WaveformProcessor, SpectrogramAnalyzer
)
from multimodal.cross_modal import (
    CrossModalConfig, ModalityAlignment, CrossAttention,
    TextImageAlignment, AudioTextAlignment, VideoTextAlignment,
    MultiModalEncoder, ContrastiveLearning
)
from multimodal.fusion import (
    FusionConfig, ModalityFusion, AttentionFusion, ConcatenationFusion,
    BilinearFusion, TensorFusion, MultiModalTransformer,
    AdaptiveFusion, HierarchicalFusion
)


class TestVisionModule:
    """Test cases for vision processing components."""
    
    def test_vision_config(self):
        """Test VisionConfig creation and validation."""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            embedding_dim=768,
            num_layers=12,
            num_heads=12,
            dropout_rate=0.1
        )
        
        assert config.image_size == 224
        assert config.patch_size == 16
        assert config.num_channels == 3
        assert config.embedding_dim == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.dropout_rate == 0.1
    
    def test_image_encoder(self):
        """Test ImageEncoder functionality."""
        config = VisionConfig(
            image_size=224,
            embedding_dim=512,
            num_channels=3
        )
        
        encoder = ImageEncoder(config)
        
        # Test forward pass
        batch_size = 8
        images = torch.randn(batch_size, 3, 224, 224)
        
        embeddings = encoder(images)
        
        assert embeddings.shape == (batch_size, 512)
        assert not torch.isnan(embeddings).any()
    
    def test_video_processor(self):
        """Test VideoProcessor functionality."""
        config = VisionConfig(
            image_size=224,
            num_frames=16,
            embedding_dim=512
        )
        
        processor = VideoProcessor(config)
        
        # Test video processing
        batch_size = 4
        video = torch.randn(batch_size, 16, 3, 224, 224)  # [B, T, C, H, W]
        
        features = processor(video)
        
        assert features.shape == (batch_size, 512)
        assert not torch.isnan(features).any()
    
    def test_diffusion_model(self):
        """Test DiffusionModel functionality."""
        config = VisionConfig(
            image_size=64,
            num_channels=3,
            embedding_dim=256,
            num_timesteps=1000
        )
        
        diffusion = DiffusionModel(config)
        
        # Test noise prediction
        batch_size = 4
        images = torch.randn(batch_size, 3, 64, 64)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        noise_pred = diffusion(images, timesteps)
        
        assert noise_pred.shape == (batch_size, 3, 64, 64)
        assert not torch.isnan(noise_pred).any()
    
    def test_object_detector(self):
        """Test ObjectDetector functionality."""
        config = VisionConfig(
            image_size=416,
            num_classes=80,
            num_anchors=3
        )
        
        detector = ObjectDetector(config)
        
        # Test object detection
        batch_size = 2
        images = torch.randn(batch_size, 3, 416, 416)
        
        detections = detector(images)
        
        assert isinstance(detections, dict)
        assert "boxes" in detections
        assert "scores" in detections
        assert "classes" in detections
    
    def test_image_classifier(self):
        """Test ImageClassifier functionality."""
        config = VisionConfig(
            image_size=224,
            num_classes=1000,
            embedding_dim=512
        )
        
        classifier = ImageClassifier(config)
        
        # Test classification
        batch_size = 8
        images = torch.randn(batch_size, 3, 224, 224)
        
        logits = classifier(images)
        
        assert logits.shape == (batch_size, 1000)
        assert not torch.isnan(logits).any()
    
    def test_semantic_segmentation(self):
        """Test SemanticSegmentation functionality."""
        config = VisionConfig(
            image_size=512,
            num_classes=21,  # PASCAL VOC classes
            embedding_dim=256
        )
        
        segmentation = SemanticSegmentation(config)
        
        # Test segmentation
        batch_size = 4
        images = torch.randn(batch_size, 3, 512, 512)
        
        masks = segmentation(images)
        
        assert masks.shape == (batch_size, 21, 512, 512)
        assert not torch.isnan(masks).any()


class TestAudioModule:
    """Test cases for audio processing components."""
    
    def test_audio_config(self):
        """Test AudioConfig creation and validation."""
        config = AudioConfig(
            sample_rate=16000,
            n_fft=512,
            hop_length=256,
            n_mels=80,
            max_length=16000,
            embedding_dim=512
        )
        
        assert config.sample_rate == 16000
        assert config.n_fft == 512
        assert config.hop_length == 256
        assert config.n_mels == 80
        assert config.max_length == 16000
        assert config.embedding_dim == 512
    
    def test_speech_recognition(self):
        """Test SpeechRecognition functionality."""
        config = AudioConfig(
            sample_rate=16000,
            embedding_dim=512,
            vocab_size=1000
        )
        
        asr = SpeechRecognition(config)
        
        # Test speech recognition
        batch_size = 4
        audio = torch.randn(batch_size, 16000)  # 1 second of audio
        
        transcription = asr(audio)
        
        assert transcription.shape == (batch_size, asr.max_seq_length, 1000)
        assert not torch.isnan(transcription).any()
    
    def test_audio_encoder(self):
        """Test AudioEncoder functionality."""
        config = AudioConfig(
            sample_rate=16000,
            n_mels=80,
            embedding_dim=512
        )
        
        encoder = AudioEncoder(config)
        
        # Test audio encoding
        batch_size = 8
        audio = torch.randn(batch_size, 16000)
        
        embeddings = encoder(audio)
        
        assert embeddings.shape == (batch_size, 512)
        assert not torch.isnan(embeddings).any()
    
    def test_text_to_speech(self):
        """Test TextToSpeech functionality."""
        config = AudioConfig(
            sample_rate=22050,
            vocab_size=1000,
            embedding_dim=256,
            max_length=22050
        )
        
        tts = TextToSpeech(config)
        
        # Test text to speech
        batch_size = 2
        text_tokens = torch.randint(0, 1000, (batch_size, 50))
        
        audio = tts(text_tokens)
        
        assert audio.shape == (batch_size, 22050)
        assert not torch.isnan(audio).any()
    
    def test_audio_classifier(self):
        """Test AudioClassifier functionality."""
        config = AudioConfig(
            sample_rate=16000,
            num_classes=10,
            embedding_dim=512
        )
        
        classifier = AudioClassifier(config)
        
        # Test audio classification
        batch_size = 6
        audio = torch.randn(batch_size, 16000)
        
        logits = classifier(audio)
        
        assert logits.shape == (batch_size, 10)
        assert not torch.isnan(logits).any()
    
    def test_audio_feature_extractor(self):
        """Test AudioFeatureExtractor functionality."""
        config = AudioConfig(
            sample_rate=16000,
            n_fft=512,
            hop_length=256,
            n_mels=80
        )
        
        extractor = AudioFeatureExtractor(config)
        
        # Test feature extraction
        audio = torch.randn(16000)  # 1 second of audio
        
        features = extractor(audio)
        
        assert features.shape[1] == 80  # n_mels
        assert not torch.isnan(features).any()
    
    def test_waveform_processor(self):
        """Test WaveformProcessor functionality."""
        config = AudioConfig(sample_rate=16000)
        processor = WaveformProcessor(config)
        
        # Test waveform processing
        audio = torch.randn(32000)  # 2 seconds of audio
        
        # Test normalization
        normalized = processor.normalize(audio)
        assert torch.abs(normalized).max() <= 1.0
        
        # Test resampling
        resampled = processor.resample(audio, target_rate=8000)
        assert resampled.shape[0] == 16000  # Half the length
    
    def test_spectrogram_analyzer(self):
        """Test SpectrogramAnalyzer functionality."""
        config = AudioConfig(
            sample_rate=16000,
            n_fft=512,
            hop_length=256
        )
        
        analyzer = SpectrogramAnalyzer(config)
        
        # Test spectrogram analysis
        audio = torch.randn(16000)
        
        spectrogram = analyzer.compute_spectrogram(audio)
        mel_spectrogram = analyzer.compute_mel_spectrogram(audio)
        
        assert spectrogram.shape[0] == 257  # n_fft // 2 + 1
        assert mel_spectrogram.shape[0] == analyzer.n_mels
        assert not torch.isnan(spectrogram).any()
        assert not torch.isnan(mel_spectrogram).any()


class TestCrossModalModule:
    """Test cases for cross-modal processing components."""
    
    def test_cross_modal_config(self):
        """Test CrossModalConfig creation and validation."""
        config = CrossModalConfig(
            vision_dim=768,
            audio_dim=512,
            text_dim=768,
            fusion_dim=1024,
            num_heads=8,
            num_layers=6
        )
        
        assert config.vision_dim == 768
        assert config.audio_dim == 512
        assert config.text_dim == 768
        assert config.fusion_dim == 1024
        assert config.num_heads == 8
        assert config.num_layers == 6
    
    def test_modality_alignment(self):
        """Test ModalityAlignment functionality."""
        config = CrossModalConfig(
            vision_dim=512,
            text_dim=768,
            fusion_dim=256
        )
        
        alignment = ModalityAlignment(config)
        
        # Test alignment
        batch_size = 8
        vision_features = torch.randn(batch_size, 512)
        text_features = torch.randn(batch_size, 768)
        
        aligned_vision, aligned_text = alignment(vision_features, text_features)
        
        assert aligned_vision.shape == (batch_size, 256)
        assert aligned_text.shape == (batch_size, 256)
        assert not torch.isnan(aligned_vision).any()
        assert not torch.isnan(aligned_text).any()
    
    def test_cross_attention(self):
        """Test CrossAttention functionality."""
        config = CrossModalConfig(
            vision_dim=512,
            text_dim=768,
            num_heads=8
        )
        
        cross_attention = CrossAttention(config)
        
        # Test cross attention
        batch_size = 4
        vision_seq = torch.randn(batch_size, 196, 512)  # 14x14 patches
        text_seq = torch.randn(batch_size, 50, 768)  # 50 tokens
        
        attended_vision = cross_attention(vision_seq, text_seq, text_seq)
        
        assert attended_vision.shape == (batch_size, 196, 512)
        assert not torch.isnan(attended_vision).any()
    
    def test_text_image_alignment(self):
        """Test TextImageAlignment functionality."""
        config = CrossModalConfig(
            vision_dim=768,
            text_dim=768,
            fusion_dim=512
        )
        
        alignment = TextImageAlignment(config)
        
        # Test text-image alignment
        batch_size = 16
        image_features = torch.randn(batch_size, 768)
        text_features = torch.randn(batch_size, 768)
        
        similarity_scores = alignment(image_features, text_features)
        
        assert similarity_scores.shape == (batch_size, batch_size)
        assert not torch.isnan(similarity_scores).any()
    
    def test_audio_text_alignment(self):
        """Test AudioTextAlignment functionality."""
        config = CrossModalConfig(
            audio_dim=512,
            text_dim=768,
            fusion_dim=256
        )
        
        alignment = AudioTextAlignment(config)
        
        # Test audio-text alignment
        batch_size = 12
        audio_features = torch.randn(batch_size, 512)
        text_features = torch.randn(batch_size, 768)
        
        aligned_audio, aligned_text = alignment(audio_features, text_features)
        
        assert aligned_audio.shape == (batch_size, 256)
        assert aligned_text.shape == (batch_size, 256)
        assert not torch.isnan(aligned_audio).any()
        assert not torch.isnan(aligned_text).any()
    
    def test_multimodal_encoder(self):
        """Test MultiModalEncoder functionality."""
        config = CrossModalConfig(
            vision_dim=768,
            audio_dim=512,
            text_dim=768,
            fusion_dim=1024,
            num_layers=4
        )
        
        encoder = MultiModalEncoder(config)
        
        # Test multimodal encoding
        batch_size = 8
        vision_features = torch.randn(batch_size, 768)
        audio_features = torch.randn(batch_size, 512)
        text_features = torch.randn(batch_size, 768)
        
        fused_features = encoder(vision_features, audio_features, text_features)
        
        assert fused_features.shape == (batch_size, 1024)
        assert not torch.isnan(fused_features).any()
    
    def test_contrastive_learning(self):
        """Test ContrastiveLearning functionality."""
        config = CrossModalConfig(
            vision_dim=512,
            text_dim=512,
            temperature=0.07
        )
        
        contrastive = ContrastiveLearning(config)
        
        # Test contrastive learning
        batch_size = 32
        vision_features = torch.randn(batch_size, 512)
        text_features = torch.randn(batch_size, 512)
        
        loss = contrastive(vision_features, text_features)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)


class TestFusionModule:
    """Test cases for modality fusion components."""
    
    def test_fusion_config(self):
        """Test FusionConfig creation and validation."""
        config = FusionConfig(
            input_dims=[768, 512, 768],  # vision, audio, text
            fusion_dim=1024,
            fusion_type="attention",
            num_heads=8,
            dropout_rate=0.1
        )
        
        assert config.input_dims == [768, 512, 768]
        assert config.fusion_dim == 1024
        assert config.fusion_type == "attention"
        assert config.num_heads == 8
        assert config.dropout_rate == 0.1
    
    def test_concatenation_fusion(self):
        """Test ConcatenationFusion functionality."""
        config = FusionConfig(
            input_dims=[256, 128, 256],
            fusion_dim=512
        )
        
        fusion = ConcatenationFusion(config)
        
        # Test concatenation fusion
        batch_size = 16
        modality1 = torch.randn(batch_size, 256)
        modality2 = torch.randn(batch_size, 128)
        modality3 = torch.randn(batch_size, 256)
        
        fused = fusion([modality1, modality2, modality3])
        
        assert fused.shape == (batch_size, 512)
        assert not torch.isnan(fused).any()
    
    def test_attention_fusion(self):
        """Test AttentionFusion functionality."""
        config = FusionConfig(
            input_dims=[512, 512, 512],
            fusion_dim=512,
            num_heads=8
        )
        
        fusion = AttentionFusion(config)
        
        # Test attention fusion
        batch_size = 8
        modalities = [
            torch.randn(batch_size, 512),
            torch.randn(batch_size, 512),
            torch.randn(batch_size, 512)
        ]
        
        fused = fusion(modalities)
        
        assert fused.shape == (batch_size, 512)
        assert not torch.isnan(fused).any()
    
    def test_bilinear_fusion(self):
        """Test BilinearFusion functionality."""
        config = FusionConfig(
            input_dims=[256, 256],
            fusion_dim=128
        )
        
        fusion = BilinearFusion(config)
        
        # Test bilinear fusion
        batch_size = 12
        modality1 = torch.randn(batch_size, 256)
        modality2 = torch.randn(batch_size, 256)
        
        fused = fusion([modality1, modality2])
        
        assert fused.shape == (batch_size, 128)
        assert not torch.isnan(fused).any()
    
    def test_tensor_fusion(self):
        """Test TensorFusion functionality."""
        config = FusionConfig(
            input_dims=[64, 64, 64],
            fusion_dim=256
        )
        
        fusion = TensorFusion(config)
        
        # Test tensor fusion
        batch_size = 8
        modalities = [
            torch.randn(batch_size, 64),
            torch.randn(batch_size, 64),
            torch.randn(batch_size, 64)
        ]
        
        fused = fusion(modalities)
        
        assert fused.shape == (batch_size, 256)
        assert not torch.isnan(fused).any()
    
    def test_multimodal_transformer(self):
        """Test MultiModalTransformer functionality."""
        config = FusionConfig(
            input_dims=[768, 512, 768],
            fusion_dim=1024,
            num_heads=8,
            num_layers=6
        )
        
        transformer = MultiModalTransformer(config)
        
        # Test multimodal transformer
        batch_size = 4
        seq_len = 20
        
        # Sequence inputs for each modality
        vision_seq = torch.randn(batch_size, seq_len, 768)
        audio_seq = torch.randn(batch_size, seq_len, 512)
        text_seq = torch.randn(batch_size, seq_len, 768)
        
        fused_seq = transformer([vision_seq, audio_seq, text_seq])
        
        assert fused_seq.shape == (batch_size, seq_len, 1024)
        assert not torch.isnan(fused_seq).any()
    
    def test_adaptive_fusion(self):
        """Test AdaptiveFusion functionality."""
        config = FusionConfig(
            input_dims=[512, 256, 512],
            fusion_dim=768
        )
        
        fusion = AdaptiveFusion(config)
        
        # Test adaptive fusion
        batch_size = 10
        modalities = [
            torch.randn(batch_size, 512),
            torch.randn(batch_size, 256),
            torch.randn(batch_size, 512)
        ]
        
        fused, weights = fusion(modalities)
        
        assert fused.shape == (batch_size, 768)
        assert weights.shape == (batch_size, 3)  # 3 modalities
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size))
        assert not torch.isnan(fused).any()
    
    def test_hierarchical_fusion(self):
        """Test HierarchicalFusion functionality."""
        config = FusionConfig(
            input_dims=[256, 256, 256, 256],
            fusion_dim=512,
            hierarchy_levels=2
        )
        
        fusion = HierarchicalFusion(config)
        
        # Test hierarchical fusion
        batch_size = 8
        modalities = [
            torch.randn(batch_size, 256),
            torch.randn(batch_size, 256),
            torch.randn(batch_size, 256),
            torch.randn(batch_size, 256)
        ]
        
        fused = fusion(modalities)
        
        assert fused.shape == (batch_size, 512)
        assert not torch.isnan(fused).any()


# Integration tests for multimodal module
class TestMultimodalIntegration:
    """Integration tests for multimodal module components."""
    
    def test_vision_audio_fusion_pipeline(self):
        """Test complete vision-audio fusion pipeline."""
        # Vision config
        vision_config = VisionConfig(
            image_size=224,
            embedding_dim=512
        )
        
        # Audio config
        audio_config = AudioConfig(
            sample_rate=16000,
            embedding_dim=512
        )
        
        # Fusion config
        fusion_config = FusionConfig(
            input_dims=[512, 512],
            fusion_dim=1024
        )
        
        # Create components
        image_encoder = ImageEncoder(vision_config)
        audio_encoder = AudioEncoder(audio_config)
        fusion = AttentionFusion(fusion_config)
        
        # Test pipeline
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        audio = torch.randn(batch_size, 16000)
        
        # Encode modalities
        image_features = image_encoder(images)
        audio_features = audio_encoder(audio)
        
        # Fuse modalities
        fused_features = fusion([image_features, audio_features])
        
        assert fused_features.shape == (batch_size, 1024)
        assert not torch.isnan(fused_features).any()
    
    def test_text_image_retrieval_pipeline(self):
        """Test text-image retrieval pipeline."""
        # Cross-modal config
        config = CrossModalConfig(
            vision_dim=768,
            text_dim=768,
            fusion_dim=512
        )
        
        # Create components
        alignment = TextImageAlignment(config)
        contrastive = ContrastiveLearning(config)
        
        # Test retrieval pipeline
        batch_size = 32
        image_features = torch.randn(batch_size, 768)
        text_features = torch.randn(batch_size, 768)
        
        # Compute similarities
        similarities = alignment(image_features, text_features)
        
        # Compute contrastive loss
        loss = contrastive(image_features, text_features)
        
        assert similarities.shape == (batch_size, batch_size)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(similarities).any()
        assert not torch.isnan(loss)
    
    def test_multimodal_transformer_pipeline(self):
        """Test complete multimodal transformer pipeline."""
        # Create configs
        vision_config = VisionConfig(image_size=224, embedding_dim=768)
        audio_config = AudioConfig(embedding_dim=512)
        fusion_config = FusionConfig(
            input_dims=[768, 512, 768],
            fusion_dim=1024,
            num_heads=8,
            num_layers=4
        )
        
        # Create components
        image_encoder = ImageEncoder(vision_config)
        audio_encoder = AudioEncoder(audio_config)
        transformer = MultiModalTransformer(fusion_config)
        
        # Test pipeline
        batch_size = 2
        seq_len = 10
        
        # Create sequence data
        image_seq = torch.randn(batch_size, seq_len, 3, 224, 224)
        audio_seq = torch.randn(batch_size, seq_len, 16000)
        text_seq = torch.randn(batch_size, seq_len, 768)  # Pre-encoded text
        
        # Encode sequences
        vision_features = []
        audio_features = []
        
        for t in range(seq_len):
            img_feat = image_encoder(image_seq[:, t])
            aud_feat = audio_encoder(audio_seq[:, t])
            vision_features.append(img_feat)
            audio_features.append(aud_feat)
        
        vision_seq_features = torch.stack(vision_features, dim=1)
        audio_seq_features = torch.stack(audio_features, dim=1)
        
        # Fuse with transformer
        fused_seq = transformer([
            vision_seq_features,
            audio_seq_features,
            text_seq
        ])
        
        assert fused_seq.shape == (batch_size, seq_len, 1024)
        assert not torch.isnan(fused_seq).any()
    
    def test_adaptive_multimodal_fusion(self):
        """Test adaptive fusion with varying modality availability."""
        config = FusionConfig(
            input_dims=[512, 512, 512],
            fusion_dim=768
        )
        
        fusion = AdaptiveFusion(config)
        
        batch_size = 8
        
        # Test with all modalities
        all_modalities = [
            torch.randn(batch_size, 512),
            torch.randn(batch_size, 512),
            torch.randn(batch_size, 512)
        ]
        
        fused_all, weights_all = fusion(all_modalities)
        
        # Test with missing modality (set to zeros)
        partial_modalities = [
            torch.randn(batch_size, 512),
            torch.zeros(batch_size, 512),  # Missing modality
            torch.randn(batch_size, 512)
        ]
        
        fused_partial, weights_partial = fusion(partial_modalities)
        
        assert fused_all.shape == (batch_size, 768)
        assert fused_partial.shape == (batch_size, 768)
        assert weights_all.shape == (batch_size, 3)
        assert weights_partial.shape == (batch_size, 3)
        
        # Weights for missing modality should be lower
        assert weights_partial[:, 1].mean() < weights_all[:, 1].mean()


if __name__ == "__main__":
    pytest.main([__file__])