# Core Neural Networks Tutorial

## Overview

This tutorial covers the core neural network architectures available in AIBF, including feedforward networks, convolutional networks, recurrent networks, and transformers. You'll learn how to build, train, and deploy these models effectively.

## Prerequisites

- Completion of [Getting Started Tutorial](getting_started.md)
- Basic understanding of neural networks
- Familiarity with PyTorch (recommended)
- Python 3.10+

## Neural Network Architectures

### 1. Feedforward Neural Networks

The most basic neural network architecture for general-purpose learning:

```python
from src.core.architectures.neural_networks import NeuralNetwork
from src.core.utils.data_loader import DataLoader
from src.core.utils.trainer import Trainer
import numpy as np
import torch

# Create a feedforward network
model = NeuralNetwork(
    input_size=784,  # MNIST: 28x28 = 784
    hidden_sizes=[512, 256, 128],  # Three hidden layers
    output_size=10,  # 10 classes
    activation='relu',
    dropout_rate=0.2,
    batch_norm=True
)

print(f"Model architecture:\n{model}")
print(f"Total parameters: {model.count_parameters()}")

# Generate sample data (replace with real data)
X_train = torch.randn(5000, 784)
y_train = torch.randint(0, 10, (5000,))
X_val = torch.randn(1000, 784)
y_val = torch.randint(0, 10, (1000,))

# Create data loaders
train_loader = DataLoader(
    X_train, y_train, 
    batch_size=64, 
    shuffle=True
)
val_loader = DataLoader(
    X_val, y_val, 
    batch_size=64, 
    shuffle=False
)

# Configure training
trainer = Trainer(
    model=model,
    optimizer='adam',
    learning_rate=0.001,
    loss_function='cross_entropy',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train the model
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    early_stopping=True,
    patience=5
)

# Evaluate performance
test_loss, test_acc = trainer.evaluate(val_loader)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
```

### 2. Convolutional Neural Networks (CNNs)

Perfect for image processing and computer vision tasks:

```python
from src.core.architectures.cnn import ConvolutionalNetwork
from src.core.utils.image_processor import ImageProcessor

# Create a CNN for image classification
cnn_model = ConvolutionalNetwork(
    input_channels=3,  # RGB images
    num_classes=1000,  # ImageNet classes
    architecture='resnet50',  # or 'vgg16', 'densenet121'
    pretrained=True,
    freeze_backbone=False
)

# Custom CNN architecture
custom_cnn = ConvolutionalNetwork.custom(
    layers=[
        {'type': 'conv2d', 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
        {'type': 'relu'},
        {'type': 'maxpool2d', 'kernel_size': 2},
        {'type': 'conv2d', 'out_channels': 64, 'kernel_size': 3, 'padding': 1},
        {'type': 'relu'},
        {'type': 'maxpool2d', 'kernel_size': 2},
        {'type': 'conv2d', 'out_channels': 128, 'kernel_size': 3, 'padding': 1},
        {'type': 'relu'},
        {'type': 'adaptive_avgpool2d', 'output_size': (1, 1)},
        {'type': 'flatten'},
        {'type': 'linear', 'out_features': 256},
        {'type': 'relu'},
        {'type': 'dropout', 'p': 0.5},
        {'type': 'linear', 'out_features': 10}
    ]
)

# Image preprocessing
image_processor = ImageProcessor(
    resize=(224, 224),
    normalize=True,
    augmentation={
        'horizontal_flip': 0.5,
        'rotation': 15,
        'color_jitter': 0.2
    }
)

# Load and preprocess images
images = image_processor.load_batch(['path/to/image1.jpg', 'path/to/image2.jpg'])
processed_images = image_processor.preprocess(images)

# Make predictions
with torch.no_grad():
    predictions = cnn_model(processed_images)
    probabilities = torch.softmax(predictions, dim=1)
    
print(f"Predictions shape: {predictions.shape}")
print(f"Top-5 predictions: {torch.topk(probabilities, 5)}")
```

### 3. Recurrent Neural Networks (RNNs)

Ideal for sequential data and time series:

```python
from src.core.architectures.rnn import RecurrentNetwork
from src.core.utils.sequence_processor import SequenceProcessor

# Create LSTM for sequence classification
lstm_model = RecurrentNetwork(
    input_size=100,  # Embedding dimension
    hidden_size=256,
    num_layers=2,
    output_size=5,  # Number of classes
    rnn_type='lstm',  # 'lstm', 'gru', or 'rnn'
    bidirectional=True,
    dropout=0.3
)

# Create GRU for sequence-to-sequence
seq2seq_model = RecurrentNetwork.seq2seq(
    encoder_config={
        'input_size': 1000,  # Vocabulary size
        'hidden_size': 512,
        'num_layers': 2,
        'rnn_type': 'gru'
    },
    decoder_config={
        'output_size': 1000,  # Target vocabulary size
        'hidden_size': 512,
        'num_layers': 2,
        'rnn_type': 'gru'
    },
    attention=True
)

# Sequence processing
seq_processor = SequenceProcessor(
    max_length=128,
    padding='post',
    truncation='post'
)

# Sample sequence data
sequences = [
    "This is a sample sentence for classification",
    "Another example of text data",
    "Machine learning is fascinating"
]

# Process sequences
processed_seqs = seq_processor.encode_sequences(sequences)
padded_seqs = seq_processor.pad_sequences(processed_seqs)

# Convert to tensors
input_tensor = torch.tensor(padded_seqs, dtype=torch.long)

# Forward pass
with torch.no_grad():
    output = lstm_model(input_tensor)
    predictions = torch.argmax(output, dim=-1)
    
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Predictions: {predictions}")
```

### 4. Transformer Networks

State-of-the-art architecture for NLP and beyond:

```python
from src.core.architectures.transformers import TransformerModel
from src.core.utils.tokenizer import Tokenizer

# Create a transformer for text classification
transformer = TransformerModel(
    vocab_size=30000,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    max_seq_length=512,
    num_classes=3,  # For classification
    dropout=0.1
)

# Pre-trained transformer (BERT-like)
bert_model = TransformerModel.from_pretrained(
    model_name='bert-base-uncased',
    num_classes=2,  # Binary classification
    fine_tune=True
)

# Custom transformer configuration
custom_transformer = TransformerModel.custom(
    config={
        'vocab_size': 50000,
        'd_model': 768,
        'nhead': 12,
        'num_layers': 12,
        'dim_feedforward': 3072,
        'activation': 'gelu',
        'layer_norm_eps': 1e-12,
        'attention_dropout': 0.1,
        'hidden_dropout': 0.1
    }
)

# Tokenization
tokenizer = Tokenizer(
    model_name='bert-base-uncased',
    max_length=512,
    padding=True,
    truncation=True
)

# Sample text data
texts = [
    "The transformer architecture revolutionized NLP",
    "Attention is all you need for sequence modeling",
    "BERT and GPT are popular transformer variants"
]

# Tokenize texts
tokenized = tokenizer.encode_batch(texts)
input_ids = torch.tensor(tokenized['input_ids'])
attention_mask = torch.tensor(tokenized['attention_mask'])

# Forward pass
with torch.no_grad():
    outputs = transformer(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
print(f"Input IDs shape: {input_ids.shape}")
print(f"Outputs shape: {outputs.shape}")
```

## Advanced Training Techniques

### 1. Transfer Learning

```python
from src.core.utils.transfer_learning import TransferLearner

# Initialize transfer learning
transfer_learner = TransferLearner(
    base_model='resnet50',
    pretrained=True,
    num_classes=10  # Your specific task
)

# Freeze backbone layers
transfer_learner.freeze_backbone()

# Fine-tune specific layers
transfer_learner.unfreeze_layers(['layer4', 'fc'])

# Progressive unfreezing
for epoch in range(50):
    if epoch == 10:
        transfer_learner.unfreeze_layers(['layer3'])
    elif epoch == 20:
        transfer_learner.unfreeze_layers(['layer2'])
    elif epoch == 30:
        transfer_learner.unfreeze_all()
    
    # Training code here
    trainer.train_epoch(train_loader)
```

### 2. Learning Rate Scheduling

```python
from src.core.utils.schedulers import LearningRateScheduler

# Create scheduler
scheduler = LearningRateScheduler(
    scheduler_type='cosine_annealing',
    T_max=50,  # Maximum epochs
    eta_min=1e-6  # Minimum learning rate
)

# Alternative schedulers
step_scheduler = LearningRateScheduler(
    scheduler_type='step',
    step_size=10,
    gamma=0.1
)

plateau_scheduler = LearningRateScheduler(
    scheduler_type='reduce_on_plateau',
    patience=5,
    factor=0.5,
    min_lr=1e-6
)

# Use in training loop
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)
    
    # Update learning rate
    scheduler.step(val_loss)  # For plateau scheduler
    # scheduler.step()  # For other schedulers
    
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch}, LR: {current_lr:.6f}")
```

### 3. Regularization Techniques

```python
from src.core.utils.regularization import RegularizationManager

# Initialize regularization
reg_manager = RegularizationManager()

# Add different regularization techniques
reg_manager.add_l1_regularization(weight=1e-4)
reg_manager.add_l2_regularization(weight=1e-3)
reg_manager.add_dropout(p=0.5)
reg_manager.add_batch_norm()
reg_manager.add_layer_norm()

# Apply to model
regularized_model = reg_manager.apply_to_model(model)

# Custom regularization loss
def custom_loss_with_regularization(outputs, targets, model):
    base_loss = torch.nn.functional.cross_entropy(outputs, targets)
    reg_loss = reg_manager.compute_regularization_loss(model)
    return base_loss + reg_loss
```

## Model Optimization

### 1. Quantization

```python
from src.core.utils.quantization import ModelQuantizer

# Post-training quantization
quantizer = ModelQuantizer()
quantized_model = quantizer.quantize_dynamic(
    model=trained_model,
    qconfig_spec={
        torch.nn.Linear: torch.quantization.default_dynamic_qconfig
    }
)

# Quantization-aware training
qat_model = quantizer.prepare_qat(
    model=model,
    qconfig=torch.quantization.get_default_qat_qconfig('fbgemm')
)

# Train with quantization awareness
for epoch in range(num_epochs):
    trainer.train_epoch(train_loader, model=qat_model)
    
# Convert to quantized model
final_quantized = quantizer.convert(qat_model)

# Compare model sizes
original_size = quantizer.get_model_size(trained_model)
quantized_size = quantizer.get_model_size(final_quantized)
print(f"Size reduction: {original_size / quantized_size:.2f}x")
```

### 2. Pruning

```python
from src.core.utils.pruning import ModelPruner

# Initialize pruner
pruner = ModelPruner()

# Structured pruning
structured_pruned = pruner.structured_prune(
    model=trained_model,
    amount=0.3,  # Remove 30% of channels
    method='l1_norm'
)

# Unstructured pruning
unstructured_pruned = pruner.unstructured_prune(
    model=trained_model,
    amount=0.5,  # Remove 50% of weights
    method='magnitude'
)

# Gradual pruning during training
for epoch in range(num_epochs):
    # Gradually increase pruning
    current_sparsity = min(0.8, epoch * 0.02)
    pruner.apply_pruning(model, sparsity=current_sparsity)
    
    trainer.train_epoch(train_loader)
    
    # Remove pruning masks periodically
    if epoch % 10 == 0:
        pruner.remove_pruning(model)
```

### 3. Knowledge Distillation

```python
from src.core.utils.distillation import KnowledgeDistiller

# Create teacher and student models
teacher_model = TransformerModel(d_model=768, num_layers=12)  # Large model
student_model = TransformerModel(d_model=384, num_layers=6)   # Small model

# Load pre-trained teacher
teacher_model.load_state_dict(torch.load('teacher_weights.pth'))
teacher_model.eval()

# Initialize distiller
distiller = KnowledgeDistiller(
    teacher=teacher_model,
    student=student_model,
    temperature=4.0,
    alpha=0.7  # Weight for distillation loss
)

# Distillation training
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        
        # Train student
        student_outputs = student_model(inputs)
        loss = distiller.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            targets=targets
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Model Evaluation and Analysis

### 1. Comprehensive Evaluation

```python
from src.core.utils.evaluation import ModelEvaluator
from src.core.utils.metrics import MetricsCalculator

# Initialize evaluator
evaluator = ModelEvaluator(model=trained_model)
metrics_calc = MetricsCalculator()

# Evaluate on test set
test_results = evaluator.evaluate(
    test_loader=test_loader,
    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc']
)

print(f"Test Results: {test_results}")

# Confusion matrix
confusion_matrix = metrics_calc.confusion_matrix(
    y_true=test_labels,
    y_pred=predictions
)

# Classification report
report = metrics_calc.classification_report(
    y_true=test_labels,
    y_pred=predictions,
    target_names=class_names
)

print(f"Classification Report:\n{report}")
```

### 2. Model Interpretability

```python
from src.core.utils.interpretability import ModelInterpreter

# Initialize interpreter
interpreter = ModelInterpreter(model=trained_model)

# Feature importance
feature_importance = interpreter.get_feature_importance(
    input_data=sample_input,
    method='integrated_gradients'
)

# Attention visualization (for transformers)
attention_weights = interpreter.visualize_attention(
    input_text="The model pays attention to important words",
    layer=6,
    head=8
)

# LIME explanations
lime_explanation = interpreter.explain_with_lime(
    instance=sample_instance,
    num_features=10
)

# SHAP values
shap_values = interpreter.compute_shap_values(
    background_data=background_samples,
    test_data=test_samples
)
```

## Production Deployment

### 1. Model Serialization

```python
from src.core.utils.serialization import ModelSerializer

# Initialize serializer
serializer = ModelSerializer()

# Save model with metadata
serializer.save_model(
    model=trained_model,
    path='models/my_model.pth',
    metadata={
        'architecture': 'transformer',
        'vocab_size': 30000,
        'num_classes': 10,
        'training_data': 'custom_dataset',
        'accuracy': 0.95,
        'created_at': '2024-01-22'
    }
)

# Load model
loaded_model, metadata = serializer.load_model('models/my_model.pth')
print(f"Loaded model metadata: {metadata}")

# Export to different formats
serializer.export_to_onnx(
    model=trained_model,
    dummy_input=sample_input,
    path='models/my_model.onnx'
)

serializer.export_to_torchscript(
    model=trained_model,
    path='models/my_model.pt'
)
```

### 2. Model Serving

```python
from src.core.utils.serving import ModelServer

# Initialize model server
server = ModelServer(
    model=loaded_model,
    preprocessing_fn=preprocess_function,
    postprocessing_fn=postprocess_function
)

# Start serving
server.start(
    host='0.0.0.0',
    port=8080,
    workers=4
)

# Health check endpoint
@server.route('/health')
def health_check():
    return {'status': 'healthy', 'model_loaded': True}

# Prediction endpoint
@server.route('/predict', methods=['POST'])
def predict():
    data = request.json
    predictions = server.predict(data['inputs'])
    return {'predictions': predictions.tolist()}
```

## Best Practices

### 1. Experiment Tracking

```python
from src.core.utils.experiment_tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    project_name='neural_network_experiments',
    experiment_name='transformer_classification'
)

# Log hyperparameters
tracker.log_params({
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50,
    'model_architecture': 'transformer',
    'd_model': 512,
    'num_layers': 6
})

# Log metrics during training
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_loss, val_acc = trainer.validate(val_loader)
    
    tracker.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }, step=epoch)

# Log final results
tracker.log_metrics({
    'final_test_accuracy': test_accuracy,
    'model_size_mb': model_size,
    'inference_time_ms': inference_time
})

# Save artifacts
tracker.log_artifact('models/best_model.pth')
tracker.log_artifact('plots/training_curves.png')
```

### 2. Error Handling and Logging

```python
import logging
from src.core.utils.error_handling import ErrorHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize error handler
error_handler = ErrorHandler(logger=logger)

# Robust training loop
try:
    for epoch in range(num_epochs):
        try:
            train_loss = trainer.train_epoch(train_loader)
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("GPU OOM detected, reducing batch size")
                train_loader.batch_size //= 2
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                
        except Exception as e:
            error_handler.handle_training_error(e, epoch)
            
except KeyboardInterrupt:
    logger.info("Training interrupted by user")
    error_handler.save_checkpoint(model, optimizer, epoch)
    
except Exception as e:
    error_handler.handle_critical_error(e)
    raise
```

## Summary

This tutorial covered:

- ✅ **Feedforward Networks** - Basic neural architectures
- ✅ **CNNs** - Convolutional networks for vision
- ✅ **RNNs** - Recurrent networks for sequences
- ✅ **Transformers** - State-of-the-art attention models
- ✅ **Advanced Training** - Transfer learning, scheduling, regularization
- ✅ **Model Optimization** - Quantization, pruning, distillation
- ✅ **Evaluation** - Comprehensive testing and interpretability
- ✅ **Production** - Deployment and serving
- ✅ **Best Practices** - Experiment tracking and error handling

## Next Steps

1. **[Healthcare AI Example](../examples/healthcare_ai.py)** - Apply these concepts to real healthcare data
2. **[Enhancement Tutorial](enhancement_tutorial.md)** - Learn about RAG and fine-tuning
3. **[Agents Tutorial](agents_tutorial.md)** - Build intelligent agent systems
4. **[API Reference](../api/core.md)** - Detailed API documentation

You now have the knowledge to build sophisticated neural networks with AIBF. Experiment with different architectures and find what works best for your specific use case!