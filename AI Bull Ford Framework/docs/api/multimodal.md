# Multimodal API Reference

## Panoramica

Il modulo Multimodal di AIBF fornisce capacità avanzate per l'elaborazione e l'integrazione di diverse modalità di dati: visione, audio, testo e fusione cross-modale.

## Vision

### Image Processing

```python
from src.multimodal.vision.image_processing import (
    ImageProcessor, ImageTransform, ImageAugmentation
)
import torch
from PIL import Image
import numpy as np

# Processore immagini base
processor = ImageProcessor(
    target_size=(224, 224),
    normalize=True,
    mean=[0.485, 0.456, 0.406],  # ImageNet means
    std=[0.229, 0.224, 0.225]    # ImageNet stds
)

# Carica e preprocessa immagine
image = Image.open("example.jpg")
processed_tensor = processor.process(image)
print(f"Processed image shape: {processed_tensor.shape}")  # [3, 224, 224]

# Batch processing
images = [Image.open(f"image_{i}.jpg") for i in range(5)]
batch_tensor = processor.process_batch(images)
print(f"Batch shape: {batch_tensor.shape}")  # [5, 3, 224, 224]

# Trasformazioni personalizzate
class CustomTransform(ImageTransform):
    def __init__(self, brightness_factor=1.2):
        self.brightness_factor = brightness_factor
    
    def apply(self, image):
        # Aumenta luminosità
        if isinstance(image, Image.Image):
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(self.brightness_factor)
        elif isinstance(image, torch.Tensor):
            return torch.clamp(image * self.brightness_factor, 0, 1)
        else:
            return image * self.brightness_factor

# Augmentation pipeline
augmentation = ImageAugmentation([
    CustomTransform(brightness_factor=1.1),
    ImageTransform.random_rotation(degrees=15),
    ImageTransform.random_crop(size=(200, 200)),
    ImageTransform.random_horizontal_flip(p=0.5),
    ImageTransform.color_jitter(brightness=0.2, contrast=0.2)
])

# Applica augmentation
augmented_image = augmentation.apply(image)
augmented_batch = augmentation.apply_batch(images)
```

### Object Detection

```python
from src.multimodal.vision.object_detection import (
    YOLODetector, FasterRCNNDetector, DetectionResult, BoundingBox
)

# YOLO Detector
yolo_detector = YOLODetector(
    model_name="yolov8n",  # "yolov8s", "yolov8m", "yolov8l", "yolov8x"
    confidence_threshold=0.5,
    iou_threshold=0.45,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Carica modello
await yolo_detector.load_model()

# Rileva oggetti in singola immagine
image = Image.open("street_scene.jpg")
detections = await yolo_detector.detect(image)

print(f"Trovati {len(detections)} oggetti:")
for detection in detections:
    bbox = detection.bounding_box
    print(f"  {detection.class_name}: {detection.confidence:.3f} "
          f"[{bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}]")

# Batch detection
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
batch_detections = await yolo_detector.detect_batch(images)

for i, detections in enumerate(batch_detections):
    print(f"Immagine {i}: {len(detections)} oggetti rilevati")

# Faster R-CNN per detection più accurata
frcnn_detector = FasterRCNNDetector(
    backbone="resnet50",  # "resnet101", "mobilenet_v3"
    pretrained=True,
    num_classes=91,  # COCO classes
    confidence_threshold=0.7
)

await frcnn_detector.load_model()

# Detection con Faster R-CNN
detections = await frcnn_detector.detect(image)

# Filtra detection per classe specifica
car_detections = [d for d in detections if d.class_name == "car"]
print(f"Trovate {len(car_detections)} automobili")

# Tracking oggetti nel tempo
from src.multimodal.vision.tracking import MultiObjectTracker

tracker = MultiObjectTracker(
    max_disappeared=30,  # Frame prima di considerare oggetto scomparso
    max_distance=50      # Distanza massima per associazione
)

# Simula tracking su video
video_frames = [Image.open(f"frame_{i:04d}.jpg") for i in range(100)]
tracked_objects = {}

for frame_idx, frame in enumerate(video_frames):
    detections = await yolo_detector.detect(frame)
    tracked_objects[frame_idx] = tracker.update(detections)
    
    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}: {len(tracked_objects[frame_idx])} oggetti tracciati")

# Analizza traiettorie
trajectories = tracker.get_trajectories()
for obj_id, trajectory in trajectories.items():
    print(f"Oggetto {obj_id}: {len(trajectory)} posizioni")
```

### Image Segmentation

```python
from src.multimodal.vision.segmentation import (
    SemanticSegmentation, InstanceSegmentation, PanopticSegmentation
)

# Segmentazione semantica
semantic_model = SemanticSegmentation(
    model_name="deeplabv3_resnet101",
    num_classes=21,  # PASCAL VOC classes
    pretrained=True
)

await semantic_model.load_model()

# Segmenta immagine
image = Image.open("scene.jpg")
segmentation_mask = await semantic_model.segment(image)

print(f"Segmentation mask shape: {segmentation_mask.shape}")
print(f"Classi presenti: {np.unique(segmentation_mask)}")

# Visualizza risultati
colored_mask = semantic_model.colorize_mask(segmentation_mask)
colored_mask.save("segmentation_result.png")

# Calcola statistiche per classe
class_stats = semantic_model.compute_class_statistics(segmentation_mask)
for class_id, stats in class_stats.items():
    class_name = semantic_model.get_class_name(class_id)
    print(f"{class_name}: {stats['pixel_count']} pixels ({stats['percentage']:.1f}%)")

# Segmentazione di istanza con Mask R-CNN
instance_model = InstanceSegmentation(
    model_name="maskrcnn_resnet50_fpn",
    confidence_threshold=0.7,
    pretrained=True
)

await instance_model.load_model()

# Segmenta istanze
instances = await instance_model.segment(image)

print(f"Trovate {len(instances)} istanze:")
for i, instance in enumerate(instances):
    print(f"  Istanza {i}: {instance.class_name} (conf: {instance.confidence:.3f})")
    
    # Salva maschera dell'istanza
    mask = instance.mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(f"instance_{i}_mask.png")

# Segmentazione panoptica (combina semantica e istanza)
panoptic_model = PanopticSegmentation(
    semantic_model=semantic_model,
    instance_model=instance_model
)

panoptic_result = await panoptic_model.segment(image)

print(f"Segmentazione panoptica:")
print(f"  Classi stuff: {len(panoptic_result.stuff_classes)}")
print(f"  Istanze thing: {len(panoptic_result.thing_instances)}")

# Salva risultato panoptico
panoptic_visualization = panoptic_model.visualize_result(panoptic_result)
panoptic_visualization.save("panoptic_result.png")
```

### Vision Transformers

```python
from src.multimodal.vision.vision_transformers import (
    ViTClassifier, DeiTClassifier, SwinTransformer, ViTFeatureExtractor
)

# Vision Transformer per classificazione
vit_classifier = ViTClassifier(
    model_name="vit_base_patch16_224",
    num_classes=1000,  # ImageNet classes
    pretrained=True,
    patch_size=16,
    embed_dim=768
)

await vit_classifier.load_model()

# Classifica immagine
image = Image.open("cat.jpg")
predictions = await vit_classifier.classify(image)

print("Top 5 predizioni:")
for pred in predictions[:5]:
    print(f"  {pred.class_name}: {pred.confidence:.3f}")

# Estrai features con ViT
feature_extractor = ViTFeatureExtractor(
    model_name="vit_base_patch16_224",
    layer_index=-2,  # Penultimo layer
    pretrained=True
)

await feature_extractor.load_model()

# Estrai features
features = await feature_extractor.extract_features(image)
print(f"Feature shape: {features.shape}")  # [1, 197, 768] (CLS + patches)

# Features solo del token CLS
cls_features = features[:, 0, :]  # [1, 768]
print(f"CLS features shape: {cls_features.shape}")

# Batch feature extraction
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
batch_features = await feature_extractor.extract_features_batch(images)
print(f"Batch features shape: {batch_features.shape}")  # [10, 197, 768]

# DeiT (Data-efficient Image Transformer)
deit_classifier = DeiTClassifier(
    model_name="deit_base_patch16_224",
    num_classes=1000,
    pretrained=True,
    use_distillation=True  # Usa token di distillazione
)

await deit_classifier.load_model()

# Classifica con DeiT
deit_predictions = await deit_classifier.classify(image)
print(f"DeiT top prediction: {deit_predictions[0].class_name} ({deit_predictions[0].confidence:.3f})")

# Swin Transformer per tasks gerarchici
swin_model = SwinTransformer(
    model_name="swin_base_patch4_window7_224",
    num_classes=1000,
    pretrained=True,
    window_size=7,
    patch_size=4
)

await swin_model.load_model()

# Estrai features multi-scala
multi_scale_features = await swin_model.extract_hierarchical_features(image)

for i, features in enumerate(multi_scale_features):
    print(f"Stage {i} features shape: {features.shape}")

# Fine-tuning per task personalizzato
class CustomViTClassifier(ViTClassifier):
    def __init__(self, num_custom_classes):
        super().__init__(
            model_name="vit_base_patch16_224",
            num_classes=num_custom_classes,
            pretrained=True
        )
    
    async def fine_tune(self, train_loader, val_loader, epochs=10):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Validation
            val_accuracy = await self.evaluate(val_loader)
            scheduler.step()
            
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")

# Usa modello personalizzato
custom_classifier = CustomViTClassifier(num_custom_classes=10)
# await custom_classifier.fine_tune(train_loader, val_loader)
```

## Audio

### Audio Processing

```python
from src.multimodal.audio.audio_processing import (
    AudioProcessor, AudioFeatureExtractor, AudioAugmentation
)
import librosa
import soundfile as sf

# Processore audio base
audio_processor = AudioProcessor(
    sample_rate=16000,
    n_fft=2048,
    hop_length=512,
    n_mels=80,
    normalize=True
)

# Carica file audio
audio_path = "speech.wav"
audio_data, sr = librosa.load(audio_path, sr=16000)
print(f"Audio shape: {audio_data.shape}, Sample rate: {sr}")

# Estrai spettrogramma mel
mel_spectrogram = audio_processor.extract_mel_spectrogram(audio_data)
print(f"Mel spectrogram shape: {mel_spectrogram.shape}")  # [n_mels, time_frames]

# Estrai MFCC
mfcc_features = audio_processor.extract_mfcc(audio_data, n_mfcc=13)
print(f"MFCC shape: {mfcc_features.shape}")  # [n_mfcc, time_frames]

# Feature extractor avanzato
feature_extractor = AudioFeatureExtractor([
    "mel_spectrogram",
    "mfcc",
    "chroma",
    "spectral_centroid",
    "zero_crossing_rate",
    "tempo"
])

# Estrai tutte le features
features = feature_extractor.extract_features(audio_data, sr)

for feature_name, feature_data in features.items():
    if isinstance(feature_data, np.ndarray):
        print(f"{feature_name}: {feature_data.shape}")
    else:
        print(f"{feature_name}: {feature_data}")

# Augmentation audio
augmentation = AudioAugmentation([
    "time_stretch",    # Cambia velocità
    "pitch_shift",     # Cambia pitch
    "add_noise",       # Aggiungi rumore
    "time_mask",       # Maschera temporale
    "frequency_mask"   # Maschera frequenziale
])

# Applica augmentation
augmented_audio = augmentation.apply(audio_data, sr)
print(f"Augmented audio shape: {augmented_audio.shape}")

# Salva audio processato
sf.write("augmented_speech.wav", augmented_audio, sr)

# Batch processing
audio_files = ["speech1.wav", "speech2.wav", "speech3.wav"]
batch_features = []

for audio_file in audio_files:
    audio, _ = librosa.load(audio_file, sr=16000)
    features = feature_extractor.extract_features(audio, 16000)
    batch_features.append(features)

print(f"Processed {len(batch_features)} audio files")
```

### Speech Recognition

```python
from src.multimodal.audio.speech_recognition import (
    WhisperASR, Wav2Vec2ASR, SpeechRecognitionResult
)

# Whisper ASR (multilingue)
whisper_asr = WhisperASR(
    model_size="base",  # "tiny", "small", "medium", "large"
    language="auto",   # Auto-detect o specifica ("en", "it", "es", etc.)
    device="cuda" if torch.cuda.is_available() else "cpu"
)

await whisper_asr.load_model()

# Riconosci speech da file
audio_file = "speech.wav"
result = await whisper_asr.transcribe(audio_file)

print(f"Trascrizione: {result.text}")
print(f"Lingua rilevata: {result.language}")
print(f"Confidenza: {result.confidence:.3f}")

# Trascrizione con timestamp
result_with_timestamps = await whisper_asr.transcribe(
    audio_file,
    return_timestamps=True
)

for segment in result_with_timestamps.segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")

# Wav2Vec2 per inglese
wav2vec_asr = Wav2Vec2ASR(
    model_name="facebook/wav2vec2-base-960h",
    language="en",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

await wav2vec_asr.load_model()

# Riconosci speech
result = await wav2vec_asr.transcribe(audio_file)
print(f"Wav2Vec2 trascrizione: {result.text}")

# Streaming ASR
class StreamingASR:
    def __init__(self, model, chunk_duration=1.0):
        self.model = model
        self.chunk_duration = chunk_duration
        self.buffer = []
    
    async def process_audio_chunk(self, audio_chunk):
        self.buffer.extend(audio_chunk)
        
        # Processa quando buffer è abbastanza grande
        chunk_size = int(self.chunk_duration * 16000)
        if len(self.buffer) >= chunk_size:
            chunk_audio = np.array(self.buffer[:chunk_size])
            self.buffer = self.buffer[chunk_size//2:]  # Overlap del 50%
            
            result = await self.model.transcribe_array(chunk_audio)
            return result
        
        return None

# Usa streaming ASR
streaming_asr = StreamingASR(whisper_asr)

# Simula streaming audio
audio_data, sr = librosa.load("long_speech.wav", sr=16000)
chunk_size = int(0.5 * sr)  # 0.5 secondi per chunk

for i in range(0, len(audio_data), chunk_size):
    chunk = audio_data[i:i+chunk_size]
    result = await streaming_asr.process_audio_chunk(chunk)
    
    if result:
        print(f"Chunk {i//chunk_size}: {result.text}")
```

### Speech Synthesis

```python
from src.multimodal.audio.speech_synthesis import (
    TacotronTTS, FastSpeechTTS, VocoderModel, SpeechSynthesisConfig
)

# Configurazione TTS
tts_config = SpeechSynthesisConfig(
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    n_mel_channels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0
)

# Tacotron2 + WaveGlow
tacotron_tts = TacotronTTS(
    model_path="tacotron2_statedict.pt",
    vocoder_path="waveglow_256channels.pt",
    config=tts_config
)

await tacotron_tts.load_model()

# Sintetizza speech
text = "Ciao, questo è un esempio di sintesi vocale con intelligenza artificiale."
audio = await tacotron_tts.synthesize(text)

print(f"Audio sintetizzato shape: {audio.shape}")
sf.write("synthesized_speech.wav", audio, tts_config.sample_rate)

# FastSpeech2 per sintesi più veloce
fastspeech_tts = FastSpeechTTS(
    model_path="fastspeech2_model.pt",
    vocoder_path="hifigan_vocoder.pt",
    config=tts_config
)

await fastspeech_tts.load_model()

# Sintesi con controllo prosodia
audio_with_prosody = await fastspeech_tts.synthesize(
    text,
    speed=1.2,        # Velocità
    pitch_shift=0.1,  # Pitch
    energy_scale=1.1  # Energia
)

sf.write("prosody_speech.wav", audio_with_prosody, tts_config.sample_rate)

# Multi-speaker TTS
class MultiSpeakerTTS:
    def __init__(self, model_path, speaker_embeddings_path):
        self.model = None
        self.speaker_embeddings = None
        self.model_path = model_path
        self.speaker_embeddings_path = speaker_embeddings_path
    
    async def load_model(self):
        # Carica modello e embedding speaker
        self.model = torch.load(self.model_path)
        self.speaker_embeddings = torch.load(self.speaker_embeddings_path)
    
    async def synthesize_with_speaker(self, text, speaker_id):
        speaker_embedding = self.speaker_embeddings[speaker_id]
        
        # Sintetizza con embedding speaker specifico
        with torch.no_grad():
            mel_spectrogram = self.model.inference(
                text, 
                speaker_embedding=speaker_embedding
            )
            audio = self.vocoder(mel_spectrogram)
        
        return audio.cpu().numpy()

# Voice cloning
class VoiceCloning:
    def __init__(self, encoder_model, synthesizer_model, vocoder_model):
        self.encoder = encoder_model
        self.synthesizer = synthesizer_model
        self.vocoder = vocoder_model
    
    async def clone_voice(self, reference_audio, target_text):
        # Estrai embedding dalla voce di riferimento
        speaker_embedding = await self.encoder.encode_audio(reference_audio)
        
        # Sintetizza con la voce clonata
        mel_spectrogram = await self.synthesizer.synthesize(
            target_text,
            speaker_embedding=speaker_embedding
        )
        
        # Converti in audio
        audio = await self.vocoder.generate(mel_spectrogram)
        
        return audio

# Batch synthesis
texts = [
    "Primo testo da sintetizzare.",
    "Secondo testo con contenuto diverso.",
    "Terzo esempio di sintesi vocale."
]

batch_audio = await tacotron_tts.synthesize_batch(texts)

for i, audio in enumerate(batch_audio):
    sf.write(f"batch_speech_{i}.wav", audio, tts_config.sample_rate)
    print(f"Salvato batch_speech_{i}.wav")
```

### Audio Classification

```python
from src.multimodal.audio.audio_classification import (
    AudioClassifier, EnvironmentalSoundClassifier, MusicGenreClassifier
)

# Classificatore audio generico
audio_classifier = AudioClassifier(
    model_name="ast_base_patch16_384",  # Audio Spectrogram Transformer
    num_classes=527,  # AudioSet classes
    pretrained=True
)

await audio_classifier.load_model()

# Classifica audio
audio_file = "environmental_sound.wav"
predictions = await audio_classifier.classify(audio_file)

print("Top 5 predizioni:")
for pred in predictions[:5]:
    print(f"  {pred.class_name}: {pred.confidence:.3f}")

# Classificatore suoni ambientali
esc_classifier = EnvironmentalSoundClassifier(
    model_name="esc50_cnn",
    classes=[
        "dog_bark", "rain", "sea_waves", "baby_cry", "clock_tick",
        "person_sneeze", "helicopter", "chainsaw", "rooster", "fire_crackling"
    ]
)

await esc_classifier.load_model()

# Classifica suono ambientale
esc_predictions = await esc_classifier.classify(audio_file)
print(f"Suono ambientale: {esc_predictions[0].class_name} ({esc_predictions[0].confidence:.3f})")

# Classificatore generi musicali
music_classifier = MusicGenreClassifier(
    model_name="musicnn",
    genres=[
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]
)

await music_classifier.load_model()

# Classifica genere musicale
music_file = "song.wav"
genre_predictions = await music_classifier.classify(music_file)
print(f"Genere musicale: {genre_predictions[0].class_name} ({genre_predictions[0].confidence:.3f})")

# Analisi audio avanzata
class AudioAnalyzer:
    def __init__(self):
        self.classifiers = {
            "general": audio_classifier,
            "environmental": esc_classifier,
            "music": music_classifier
        }
    
    async def analyze_audio(self, audio_file):
        results = {}
        
        for classifier_name, classifier in self.classifiers.items():
            try:
                predictions = await classifier.classify(audio_file)
                results[classifier_name] = predictions[:3]  # Top 3
            except Exception as e:
                results[classifier_name] = f"Error: {str(e)}"
        
        return results
    
    def determine_audio_type(self, analysis_results):
        # Logica per determinare il tipo di audio
        music_confidence = max([p.confidence for p in analysis_results.get("music", [])], default=0)
        env_confidence = max([p.confidence for p in analysis_results.get("environmental", [])], default=0)
        
        if music_confidence > 0.7:
            return "music"
        elif env_confidence > 0.6:
            return "environmental"
        else:
            return "speech_or_other"

# Usa analyzer
analyzer = AudioAnalyzer()
analysis = await analyzer.analyze_audio("unknown_audio.wav")
audio_type = analyzer.determine_audio_type(analysis)

print(f"Tipo audio rilevato: {audio_type}")
for classifier_name, predictions in analysis.items():
    print(f"\n{classifier_name.capitalize()} predictions:")
    if isinstance(predictions, list):
        for pred in predictions:
            print(f"  {pred.class_name}: {pred.confidence:.3f}")
    else:
        print(f"  {predictions}")
```

## Cross-Modal

### Vision-Language Models

```python
from src.multimodal.cross_modal.vision_language import (
    CLIPModel, BLIPModel, VisionLanguageResult
)

# CLIP per zero-shot classification
clip_model = CLIPModel(
    model_name="ViT-B/32",  # "ViT-L/14", "RN50", "RN101"
    device="cuda" if torch.cuda.is_available() else "cpu"
)

await clip_model.load_model()

# Zero-shot image classification
image = Image.open("animal.jpg")
class_labels = ["a cat", "a dog", "a bird", "a fish", "a horse"]

classification_result = await clip_model.classify_image(image, class_labels)

print("CLIP Classification:")
for result in classification_result:
    print(f"  {result.label}: {result.confidence:.3f}")

# Image-text similarity
text_queries = [
    "a cute cat sitting on a sofa",
    "a dog playing in the park",
    "a bird flying in the sky"
]

similarities = await clip_model.compute_similarity(image, text_queries)

print("\nImage-Text Similarities:")
for query, similarity in zip(text_queries, similarities):
    print(f"  '{query}': {similarity:.3f}")

# Text-to-image retrieval
image_database = [Image.open(f"image_{i}.jpg") for i in range(100)]
query_text = "a sunset over the ocean"

retrieval_results = await clip_model.retrieve_images(
    query_text, 
    image_database, 
    top_k=5
)

print(f"\nTop 5 images for '{query_text}':")
for i, result in enumerate(retrieval_results):
    print(f"  {i+1}. Image {result.image_id}: {result.similarity:.3f}")

# BLIP per image captioning e VQA
blip_model = BLIPModel(
    model_name="blip_base",  # "blip_large"
    task="captioning",      # "vqa", "retrieval"
    device="cuda" if torch.cuda.is_available() else "cpu"
)

await blip_model.load_model()

# Image captioning
caption_result = await blip_model.generate_caption(image)
print(f"\nGenerated caption: {caption_result.caption}")
print(f"Confidence: {caption_result.confidence:.3f}")

# Visual Question Answering
questions = [
    "What animal is in the image?",
    "What color is the animal?",
    "Where is the animal located?",
    "Is the animal sleeping?"
]

for question in questions:
    vqa_result = await blip_model.answer_question(image, question)
    print(f"Q: {question}")
    print(f"A: {vqa_result.answer} (confidence: {vqa_result.confidence:.3f})\n")

# Batch processing
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
batch_captions = await blip_model.generate_captions_batch(images)

for i, caption_result in enumerate(batch_captions):
    print(f"Image {i}: {caption_result.caption}")
```

### Audio-Language Models

```python
from src.multimodal.cross_modal.audio_language import (
    AudioCLIPModel, SpeechT5Model, AudioLanguageResult
)

# AudioCLIP per audio-text matching
audio_clip = AudioCLIPModel(
    model_name="audioclip_base",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

await audio_clip.load_model()

# Audio-text similarity
audio_file = "nature_sounds.wav"
text_descriptions = [
    "birds singing in the forest",
    "waves crashing on the beach",
    "rain falling on leaves",
    "wind blowing through trees"
]

audio_text_similarities = await audio_clip.compute_similarity(
    audio_file, 
    text_descriptions
)

print("Audio-Text Similarities:")
for desc, similarity in zip(text_descriptions, audio_text_similarities):
    print(f"  '{desc}': {similarity:.3f}")

# Audio captioning
audio_caption = await audio_clip.generate_audio_caption(audio_file)
print(f"\nAudio caption: {audio_caption.caption}")

# SpeechT5 per speech-to-speech translation
speech_t5 = SpeechT5Model(
    model_name="speecht5_base",
    task="speech_to_speech",
    source_language="en",
    target_language="es"
)

await speech_t5.load_model()

# Speech-to-speech translation
input_speech = "english_speech.wav"
translated_speech = await speech_t5.translate_speech(input_speech)

sf.write("translated_spanish_speech.wav", translated_speech, 16000)
print("Speech translated from English to Spanish")

# Text-to-speech in multiple languages
texts_and_languages = [
    ("Hello, how are you?", "en"),
    ("Hola, ¿cómo estás?", "es"),
    ("Bonjour, comment allez-vous?", "fr"),
    ("Ciao, come stai?", "it")
]

for text, lang in texts_and_languages:
    speech = await speech_t5.text_to_speech(text, language=lang)
    sf.write(f"tts_{lang}.wav", speech, 16000)
    print(f"Generated speech for {lang}: '{text}'")
```

### Multimodal Embeddings

```python
from src.multimodal.cross_modal.embeddings import (
    MultimodalEmbedding, EmbeddingFusion, CrossModalRetrieval
)

# Embedding multimodali
multimodal_embedding = MultimodalEmbedding(
    vision_model="clip_vit_b32",
    text_model="sentence_transformers",
    audio_model="audioclip",
    embedding_dim=512
)

await multimodal_embedding.load_models()

# Estrai embedding per diverse modalità
image = Image.open("example.jpg")
text = "A beautiful sunset over the mountains"
audio_file = "nature_sound.wav"

image_embedding = await multimodal_embedding.encode_image(image)
text_embedding = await multimodal_embedding.encode_text(text)
audio_embedding = await multimodal_embedding.encode_audio(audio_file)

print(f"Image embedding shape: {image_embedding.shape}")
print(f"Text embedding shape: {text_embedding.shape}")
print(f"Audio embedding shape: {audio_embedding.shape}")

# Fusione di embedding
fusion_model = EmbeddingFusion(
    fusion_method="attention",  # "concat", "average", "attention", "transformer"
    input_dims=[512, 512, 512],
    output_dim=512
)

# Fonde embedding di diverse modalità
fused_embedding = fusion_model.fuse([
    image_embedding,
    text_embedding,
    audio_embedding
])

print(f"Fused embedding shape: {fused_embedding.shape}")

# Cross-modal retrieval
retrieval_system = CrossModalRetrieval(
    embedding_model=multimodal_embedding,
    index_type="faiss",  # "faiss", "annoy", "hnswlib"
    similarity_metric="cosine"
)

# Costruisci database multimodale
media_database = {
    "images": [Image.open(f"img_{i}.jpg") for i in range(1000)],
    "texts": [f"Description of image {i}" for i in range(1000)],
    "audios": [f"audio_{i}.wav" for i in range(500)]
}

await retrieval_system.build_index(media_database)

# Ricerca cross-modale
query_text = "a cat playing with a ball"
results = await retrieval_system.search(
    query=query_text,
    modality="text",
    target_modalities=["images", "audios"],
    top_k=10
)

print(f"\nCross-modal search results for '{query_text}':")
for result in results:
    print(f"  {result.modality}: {result.item_id} (similarity: {result.similarity:.3f})")

# Ricerca multimodale con query composita
composite_query = {
    "image": Image.open("query_image.jpg"),
    "text": "similar scene but at night",
    "weights": {"image": 0.7, "text": 0.3}
}

composite_results = await retrieval_system.search_multimodal(
    composite_query,
    target_modalities=["images"],
    top_k=5
)

print("\nComposite multimodal search results:")
for result in composite_results:
    print(f"  Image {result.item_id}: {result.similarity:.3f}")
```

## Modality Fusion

### Attention-based Fusion

```python
from src.multimodal.modality_fusion.attention_fusion import (
    CrossModalAttention, MultiHeadCrossAttention, AttentionFusionConfig
)

# Configurazione attention fusion
fusion_config = AttentionFusionConfig(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    temperature=0.07
)

# Cross-modal attention
cross_attention = CrossModalAttention(
    query_dim=512,    # Dimensione modalità query
    key_dim=768,      # Dimensione modalità key
    value_dim=768,    # Dimensione modalità value
    output_dim=512,
    config=fusion_config
)

# Esempio: Vision-Language attention
vision_features = torch.randn(1, 196, 768)  # [batch, patches, dim]
language_features = torch.randn(1, 50, 512)  # [batch, tokens, dim]

# Attention da language a vision
lang_to_vision = cross_attention(
    query=language_features,
    key=vision_features,
    value=vision_features
)

print(f"Language-to-Vision attention output: {lang_to_vision.shape}")

# Multi-head cross-modal attention
multi_head_attention = MultiHeadCrossAttention(
    modalities=["vision", "language", "audio"],
    modality_dims={"vision": 768, "language": 512, "audio": 256},
    output_dim=512,
    num_heads=8
)

# Features multimodali
modality_features = {
    "vision": torch.randn(1, 196, 768),
    "language": torch.randn(1, 50, 512),
    "audio": torch.randn(1, 100, 256)
}

# Fusione con attention
fused_features = multi_head_attention(modality_features)
print(f"Fused features shape: {fused_features.shape}")

# Attention weights per interpretabilità
attention_weights = multi_head_attention.get_attention_weights()
for modality_pair, weights in attention_weights.items():
    print(f"Attention {modality_pair}: {weights.shape}")
```

### Transformer-based Fusion

```python
from src.multimodal.modality_fusion.transformer_fusion import (
    MultimodalTransformer, ModalityEncoder, FusionTransformerConfig
)

# Configurazione transformer fusion
transformer_config = FusionTransformerConfig(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation="gelu"
)

# Encoder per ogni modalità
modality_encoders = {
    "vision": ModalityEncoder(
        input_dim=768,
        output_dim=512,
        num_layers=2
    ),
    "language": ModalityEncoder(
        input_dim=512,
        output_dim=512,
        num_layers=2
    ),
    "audio": ModalityEncoder(
        input_dim=256,
        output_dim=512,
        num_layers=2
    )
}

# Transformer multimodale
multimodal_transformer = MultimodalTransformer(
    modality_encoders=modality_encoders,
    config=transformer_config,
    fusion_strategy="early"  # "early", "late", "hierarchical"
)

# Processa features multimodali
raw_features = {
    "vision": torch.randn(1, 196, 768),
    "language": torch.randn(1, 50, 512),
    "audio": torch.randn(1, 100, 256)
}

# Encoding e fusione
encoded_features = {}
for modality, features in raw_features.items():
    encoded_features[modality] = modality_encoders[modality](features)

# Fusione con transformer
fused_output = multimodal_transformer(encoded_features)
print(f"Transformer fused output: {fused_output.shape}")

# Fusione gerarchica
class HierarchicalFusion(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Primo livello: fusione pairwise
        self.vision_language_fusion = MultimodalTransformer(
            modality_encoders={"vision": modality_encoders["vision"], 
                             "language": modality_encoders["language"]},
            config=config
        )
        
        self.audio_fusion = ModalityEncoder(
            input_dim=256,
            output_dim=512,
            num_layers=2
        )
        
        # Secondo livello: fusione finale
        self.final_fusion = MultimodalTransformer(
            modality_encoders={
                "vision_language": torch.nn.Identity(),
                "audio": torch.nn.Identity()
            },
            config=config
        )
    
    def forward(self, features):
        # Primo livello
        vl_fused = self.vision_language_fusion({
            "vision": features["vision"],
            "language": features["language"]
        })
        
        audio_encoded = self.audio_fusion(features["audio"])
        
        # Secondo livello
        final_output = self.final_fusion({
            "vision_language": vl_fused,
            "audio": audio_encoded
        })
        
        return final_output

hierarchical_fusion = HierarchicalFusion(transformer_config)
hierarchical_output = hierarchical_fusion(raw_features)
print(f"Hierarchical fusion output: {hierarchical_output.shape}")
```

### Graph-based Fusion

```python
from src.multimodal.modality_fusion.graph_fusion import (
    MultimodalGraphNetwork, GraphAttentionLayer, ModalityNode
)

# Nodi per ogni modalità
modality_nodes = {
    "vision": ModalityNode(
        node_id="vision",
        feature_dim=768,
        node_type="visual"
    ),
    "language": ModalityNode(
        node_id="language",
        feature_dim=512,
        node_type="textual"
    ),
    "audio": ModalityNode(
        node_id="audio",
        feature_dim=256,
        node_type="auditory"
    )
}

# Graph attention network
graph_fusion = MultimodalGraphNetwork(
    modality_nodes=modality_nodes,
    hidden_dim=512,
    num_attention_heads=4,
    num_layers=3,
    dropout=0.1
)

# Costruisci grafo delle modalità
modality_graph = graph_fusion.build_modality_graph(raw_features)

# Applica graph attention
graph_output = graph_fusion(modality_graph)
print(f"Graph fusion output: {graph_output.shape}")

# Analizza attention weights nel grafo
graph_attention_weights = graph_fusion.get_attention_weights()
for layer_idx, layer_weights in enumerate(graph_attention_weights):
    print(f"Layer {layer_idx} attention weights:")
    for edge, weight in layer_weights.items():
        print(f"  {edge[0]} -> {edge[1]}: {weight:.3f}")
```

## Esempi Completi

### Sistema Multimodale per Analisi Contenuti

```python
import asyncio
from src.multimodal import (
    ImageProcessor, AudioProcessor, CLIPModel, BLIPModel,
    MultimodalEmbedding, CrossModalRetrieval
)

class MultimodalContentAnalyzer:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.clip_model = CLIPModel()
        self.blip_model = BLIPModel()
        self.embedding_model = MultimodalEmbedding()
        self.retrieval_system = CrossModalRetrieval(self.embedding_model)
    
    async def initialize(self):
        """Inizializza tutti i modelli"""
        await self.clip_model.load_model()
        await self.blip_model.load_model()
        await self.embedding_model.load_models()
    
    async def analyze_content(self, content_path, content_type):
        """Analizza contenuto multimodale"""
        results = {}
        
        if content_type == "image":
            results = await self._analyze_image(content_path)
        elif content_type == "audio":
            results = await self._analyze_audio(content_path)
        elif content_type == "video":
            results = await self._analyze_video(content_path)
        
        return results
    
    async def _analyze_image(self, image_path):
        image = Image.open(image_path)
        
        # Genera caption
        caption = await self.blip_model.generate_caption(image)
        
        # Classifica con CLIP
        categories = ["person", "animal", "vehicle", "building", "nature", "food"]
        classification = await self.clip_model.classify_image(image, categories)
        
        # Estrai embedding
        embedding = await self.embedding_model.encode_image(image)
        
        return {
            "caption": caption.caption,
            "classification": classification[0].label,
            "confidence": classification[0].confidence,
            "embedding": embedding,
            "type": "image"
        }
    
    async def _analyze_audio(self, audio_path):
        # Carica audio
        audio_data, sr = librosa.load(audio_path, sr=16000)
        
        # Estrai features
        features = self.audio_processor.extract_mel_spectrogram(audio_data)
        
        # Classifica tipo audio
        # (implementazione semplificata)
        audio_type = "speech" if np.mean(features) > 0.5 else "music"
        
        # Estrai embedding
        embedding = await self.embedding_model.encode_audio(audio_path)
        
        return {
            "type": "audio",
            "audio_type": audio_type,
            "duration": len(audio_data) / sr,
            "embedding": embedding
        }
    
    async def search_similar_content(self, query, query_type, database, top_k=5):
        """Cerca contenuti simili nel database"""
        if query_type == "text":
            query_embedding = await self.embedding_model.encode_text(query)
        elif query_type == "image":
            query_embedding = await self.embedding_model.encode_image(query)
        elif query_type == "audio":
            query_embedding = await self.embedding_model.encode_audio(query)
        
        # Calcola similarità
        similarities = []
        for item in database:
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                item["embedding"].unsqueeze(0)
            ).item()
            similarities.append((item, similarity))
        
        # Ordina per similarità
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

# Uso del sistema
async def main():
    analyzer = MultimodalContentAnalyzer()
    await analyzer.initialize()
    
    # Analizza contenuti
    content_database = []
    
    # Analizza immagini
    image_files = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
    for image_file in image_files:
        result = await analyzer.analyze_content(image_file, "image")
        content_database.append(result)
        print(f"Analyzed {image_file}: {result['caption']}")
    
    # Analizza audio
    audio_files = ["sound1.wav", "sound2.wav"]
    for audio_file in audio_files:
        result = await analyzer.analyze_content(audio_file, "audio")
        content_database.append(result)
        print(f"Analyzed {audio_file}: {result['audio_type']}")
    
    # Ricerca per similarità
    query_text = "a cat sitting on a chair"
    similar_content = await analyzer.search_similar_content(
        query_text, "text", content_database, top_k=3
    )
    
    print(f"\nSimilar content for '{query_text}':")
    for content, similarity in similar_content:
        if content["type"] == "image":
            print(f"  Image: {content['caption']} (similarity: {similarity:.3f})")
        elif content["type"] == "audio":
            print(f"  Audio: {content['audio_type']} (similarity: {similarity:.3f})")

# Esegui analisi
# await main()
```

Per ulteriori dettagli e esempi avanzati, consulta la [documentazione completa](../guides/) e gli [esempi pratici](../examples/).