"""Multimodal AI module for AI Bull Ford.

This module provides comprehensive multimodal AI capabilities including:
- Vision processing (image, video, 3D)
- Audio processing (speech, music, sound)
- Cross-modal understanding and generation
- Modality fusion and alignment
- Multimodal embeddings and representations
"""

# Vision components
from .vision import (
    # Enums
    ImageFormat,
    VideoFormat,
    VisionTask,
    
    # Data classes
    ImageData,
    VideoData,
    VisionConfig,
    DetectionResult,
    SegmentationResult,
    
    # Core classes
    ImageProcessor,
    VideoProcessor,
    ObjectDetector,
    ImageSegmentation,
    FeatureExtractor,
    VisionEncoder,
    
    # Global functions
    process_image,
    process_video,
    extract_features,
    initialize_vision,
    shutdown_vision
)

# Audio components
from .audio import (
    # Enums
    AudioFormat,
    SampleRate,
    AudioTask,
    
    # Data classes
    AudioData,
    SpectrogramData,
    AudioConfig,
    TranscriptionResult,
    
    # Core classes
    AudioProcessor,
    SpeechRecognizer,
    SpeechSynthesizer,
    AudioEncoder,
    MusicAnalyzer,
    
    # Global functions
    process_audio,
    transcribe_speech,
    synthesize_speech,
    initialize_audio,
    shutdown_audio
)

# Cross-modal components
from .cross_modal import (
    # Enums
    ModalityType,
    AlignmentMethod,
    CrossModalTask,
    
    # Data classes
    ModalityData,
    AlignmentResult,
    CrossModalConfig,
    
    # Core classes
    ModalityAligner,
    CrossModalEncoder,
    ImageTextMatcher,
    AudioVisualMatcher,
    MultiModalRetriever,
    
    # Global functions
    align_modalities,
    match_image_text,
    match_audio_visual,
    initialize_cross_modal,
    shutdown_cross_modal
)

# Import fusion components
from .fusion import (
    # Fusion strategies and methods
    FusionStrategy, FusionMethod,
    
    # Configuration and results
    FusionConfig, FusionResult,
    
    # Fusion modules
    BaseFusionModule, EarlyFusion, LateFusion,
    AttentionFusion, TransformerFusion, AdaptiveFusion,
    ModalityFusionEngine,
    
    # Global functions
    fuse_modalities, async_fuse_modalities,
    initialize_fusion, shutdown_fusion
)

# Import cross-modal components
from .cross_modal import (
    # Core types and enums
    ModalityType, AlignmentMethod, CrossModalTask,
    
    # Data structures
    ModalityData, AlignmentResult, CrossModalConfig,
    
    # Processing classes
    ModalityAligner, CrossModalEncoder, ImageTextMatcher,
    AudioVisualMatcher, MultiModalRetriever,
    
    # Global functions
    align_modalities, match_image_text, match_audio_visual,
    initialize_cross_modal, shutdown_cross_modal
)

__all__ = [
    # Vision components
    "ImageFormat",
    "VideoFormat",
    "VisionTask",
    "ImageData",
    "VideoData",
    "VisionConfig",
    "DetectionResult",
    "SegmentationResult",
    "ImageProcessor",
    "VideoProcessor",
    "ObjectDetector",
    "ImageSegmentation",
    "FeatureExtractor",
    "VisionEncoder",
    "process_image",
    "process_video",
    "extract_features",
    "initialize_vision",
    "shutdown_vision",
    
    # Audio components
    "AudioFormat",
    "SampleRate",
    "AudioTask",
    "AudioData",
    "SpectrogramData",
    "AudioConfig",
    "TranscriptionResult",
    "AudioProcessor",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "AudioEncoder",
    "MusicAnalyzer",
    "process_audio",
    "transcribe_speech",
    "synthesize_speech",
    "initialize_audio",
    "shutdown_audio",
    
    # Cross-modal components
    "ModalityType",
    "AlignmentMethod",
    "CrossModalTask",
    "ModalityData",
    "AlignmentResult",
    "CrossModalConfig",
    "ModalityAligner",
    "CrossModalEncoder",
    "ImageTextMatcher",
    "AudioVisualMatcher",
    "MultiModalRetriever",
    "align_modalities",
    "match_image_text",
    "match_audio_visual",
    "initialize_cross_modal",
    "shutdown_cross_modal",
    
    # Fusion components
    "FusionStrategy",
    "FusionMethod",
    "FusionConfig",
    "FusionResult",
    "BaseFusionModule",
    "EarlyFusion",
    "LateFusion",
    "AttentionFusion",
    "TransformerFusion",
    "AdaptiveFusion",
    "ModalityFusionEngine",
    "fuse_modalities",
    "async_fuse_modalities",
    "initialize_fusion",
    "shutdown_fusion",
    
    # Global functions
    "initialize_multimodal",
    "shutdown_multimodal"
]

# Global initialization function
def initialize_multimodal(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize all multimodal components.
    
    Args:
        config: Optional configuration dictionary with component-specific settings
    """
    try:
        # Extract component configs
        vision_config = None
        audio_config = None
        cross_modal_config = None
        fusion_config = None
        
        if config:
            vision_config = config.get('vision')
            audio_config = config.get('audio')
            cross_modal_config = config.get('cross_modal')
            fusion_config = config.get('fusion')
        
        # Initialize components
        initialize_vision(vision_config)
        initialize_audio(audio_config)
        initialize_cross_modal(cross_modal_config)
        initialize_fusion(fusion_config)
        
        logging.info("Multimodal components initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize multimodal components: {e}")
        raise


async def shutdown_multimodal() -> None:
    """Shutdown all multimodal components."""
    try:
        # Shutdown components in reverse order
        await shutdown_fusion()
        await shutdown_cross_modal()
        await shutdown_audio()
        await shutdown_vision()
        
        logging.info("Multimodal components shutdown successfully")
    except Exception as e:
        logging.error(f"Failed to shutdown multimodal components: {e}")
        raise