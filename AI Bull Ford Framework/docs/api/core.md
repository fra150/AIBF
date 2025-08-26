# Core API Reference

## Panoramica

Il modulo Core di AIBF fornisce le funzionalità fondamentali per l'intelligenza artificiale, incluse reti neurali, transformer e reinforcement learning.

## Neural Networks

### Architetture

#### MLP (Multi-Layer Perceptron)

```python
from src.core.neural_networks.architectures.mlp import MLP, MLPConfig

# Configurazione
config = MLPConfig(
    input_size=784,
    hidden_layers=[512, 256, 128],
    output_size=10,
    activation='relu',
    dropout=0.2,
    batch_norm=True
)

# Creazione del modello
model = MLP(config)

# Training
trainer = MLPTrainer(model)
result = await trainer.train(
    train_data=train_loader,
    val_data=val_loader,
    epochs=100,
    learning_rate=0.001
)
```

**Parametri MLPConfig:**
- `input_size` (int): Dimensione dell'input
- `hidden_layers` (List[int]): Lista delle dimensioni dei layer nascosti
- `output_size` (int): Dimensione dell'output
- `activation` (str): Funzione di attivazione ('relu', 'tanh', 'sigmoid')
- `dropout` (float): Tasso di dropout (0.0-1.0)
- `batch_norm` (bool): Abilita batch normalization

#### CNN (Convolutional Neural Network)

```python
from src.core.neural_networks.architectures.cnn import CNN, CNNConfig

# Configurazione
config = CNNConfig(
    input_channels=3,
    input_size=(224, 224),
    conv_layers=[
        {'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'filters': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'filters': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    ],
    fc_layers=[512, 256],
    num_classes=1000,
    dropout=0.5
)

# Creazione e training
model = CNN(config)
trainer = CNNTrainer(model)
result = await trainer.train(train_data, val_data)
```

**Parametri CNNConfig:**
- `input_channels` (int): Numero di canali di input
- `input_size` (Tuple[int, int]): Dimensioni dell'immagine (H, W)
- `conv_layers` (List[Dict]): Configurazione dei layer convoluzionali
- `fc_layers` (List[int]): Dimensioni dei layer fully connected
- `num_classes` (int): Numero di classi per classificazione
- `dropout` (float): Tasso di dropout

#### RNN (Recurrent Neural Network)

```python
from src.core.neural_networks.architectures.rnn import RNN, RNNConfig

# Configurazione
config = RNNConfig(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    rnn_type='LSTM',  # 'RNN', 'LSTM', 'GRU'
    bidirectional=True,
    dropout=0.3,
    output_size=50
)

# Creazione del modello
model = RNN(config)

# Per sequenze
sequence_input = torch.randn(32, 100, 100)  # (batch, seq_len, input_size)
output, hidden = model(sequence_input)
```

**Parametri RNNConfig:**
- `input_size` (int): Dimensione dell'input per timestep
- `hidden_size` (int): Dimensione dello stato nascosto
- `num_layers` (int): Numero di layer RNN
- `rnn_type` (str): Tipo di RNN ('RNN', 'LSTM', 'GRU')
- `bidirectional` (bool): RNN bidirezionale
- `dropout` (float): Dropout tra i layer
- `output_size` (int): Dimensione dell'output

### Custom Layers

#### Attention Layer

```python
from src.core.neural_networks.layers.attention import AttentionLayer

# Multi-Head Attention
attention = AttentionLayer(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    bias=True
)

# Forward pass
query = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
key = torch.randn(32, 100, 512)
value = torch.randn(32, 100, 512)

output, attention_weights = attention(query, key, value)
```

#### Residual Block

```python
from src.core.neural_networks.layers.residual import ResidualBlock

# Residual block per CNN
resblock = ResidualBlock(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)

input_tensor = torch.randn(32, 64, 56, 56)
output = resblock(input_tensor)
```

### Optimizers

#### Custom Adam Optimizer

```python
from src.core.neural_networks.optimizers.adam import AdamOptimizer

# Configurazione
optimizer = AdamOptimizer(
    parameters=model.parameters(),
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    amsgrad=False
)

# Training step
for batch in dataloader:
    optimizer.zero_grad()
    loss = model.compute_loss(batch)
    loss.backward()
    optimizer.step()
```

#### Learning Rate Schedulers

```python
from src.core.neural_networks.optimizers.schedulers import (
    CosineAnnealingScheduler,
    StepLRScheduler,
    ExponentialScheduler
)

# Cosine Annealing
scheduler = CosineAnnealingScheduler(
    optimizer=optimizer,
    T_max=100,  # Numero di epoche per un ciclo completo
    eta_min=1e-6  # Learning rate minimo
)

# Step LR
step_scheduler = StepLRScheduler(
    optimizer=optimizer,
    step_size=30,  # Riduci ogni 30 epoche
    gamma=0.1  # Fattore di riduzione
)

# Exponential
exp_scheduler = ExponentialScheduler(
    optimizer=optimizer,
    gamma=0.95  # Fattore di decadimento
)

# Uso durante il training
for epoch in range(num_epochs):
    train_epoch(model, optimizer, train_loader)
    scheduler.step()
```

## Transformers

### Modelli Pre-addestrati

#### BERT

```python
from src.core.transformers.models.bert import BERT, BERTConfig

# Configurazione
config = BERTConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    dropout=0.1,
    attention_dropout=0.1
)

# Carica modello pre-addestrato
bert = BERT.from_pretrained('bert-base-uncased')

# Fine-tuning per classificazione
from src.core.transformers.models.bert import BERTForClassification

classifier = BERTForClassification(
    bert_model=bert,
    num_classes=2,
    dropout=0.3
)

# Training
trainer = BERTTrainer(classifier)
result = await trainer.fine_tune(
    train_data=train_dataset,
    val_data=val_dataset,
    epochs=3,
    learning_rate=2e-5,
    batch_size=16
)
```

#### GPT

```python
from src.core.transformers.models.gpt import GPT, GPTConfig

# Configurazione
config = GPTConfig(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    dropout=0.1,
    resid_dropout=0.1,
    attn_dropout=0.1
)

# Creazione del modello
gpt = GPT(config)

# Generazione di testo
from src.core.transformers.generation import TextGenerator

generator = TextGenerator(gpt)
generated_text = await generator.generate(
    prompt="Il futuro dell'intelligenza artificiale",
    max_length=200,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)
```

#### T5

```python
from src.core.transformers.models.t5 import T5, T5Config

# Configurazione
config = T5Config(
    vocab_size=32128,
    d_model=512,
    d_kv=64,
    d_ff=2048,
    num_layers=6,
    num_heads=8,
    dropout_rate=0.1,
    layer_norm_epsilon=1e-6
)

# Modello per text-to-text
t5 = T5.from_pretrained('t5-base')

# Task di summarization
from src.core.transformers.tasks import SummarizationTask

summarizer = SummarizationTask(t5)
summary = await summarizer.summarize(
    text="Lungo testo da riassumere...",
    max_length=150,
    min_length=50
)
```

### Attention Mechanisms

#### Multi-Head Attention

```python
from src.core.transformers.attention.multi_head import MultiHeadAttention

# Configurazione
attention = MultiHeadAttention(
    d_model=512,
    num_heads=8,
    dropout=0.1,
    bias=True,
    scale_factor=None  # Usa sqrt(d_k) di default
)

# Forward pass
query = torch.randn(32, 100, 512)
key = torch.randn(32, 100, 512)
value = torch.randn(32, 100, 512)
mask = torch.ones(32, 100, 100)  # Attention mask

output, attention_weights = attention(
    query=query,
    key=key,
    value=value,
    mask=mask
)
```

#### Sparse Attention

```python
from src.core.transformers.attention.sparse import SparseAttention

# Attention sparsa per sequenze lunghe
sparse_attention = SparseAttention(
    d_model=512,
    num_heads=8,
    sparsity_pattern='local',  # 'local', 'strided', 'random'
    block_size=64,
    num_random_blocks=3
)

# Per sequenze molto lunghe
long_sequence = torch.randn(4, 4096, 512)
output, _ = sparse_attention(long_sequence, long_sequence, long_sequence)
```

### Tokenizers

#### WordPiece Tokenizer

```python
from src.core.transformers.tokenizers.wordpiece import WordPieceTokenizer

# Carica tokenizer
tokenizer = WordPieceTokenizer.from_pretrained('bert-base-uncased')

# Tokenizzazione
text = "Ciao, come stai oggi?"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Encoding completo
encoding = tokenizer.encode(
    text,
    add_special_tokens=True,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print(f"Input IDs: {encoding['input_ids']}")
print(f"Attention Mask: {encoding['attention_mask']}")
```

#### SentencePiece Tokenizer

```python
from src.core.transformers.tokenizers.sentencepiece import SentencePieceTokenizer

# Addestra tokenizer custom
tokenizer = SentencePieceTokenizer()
tokenizer.train(
    input_files=['corpus.txt'],
    vocab_size=32000,
    model_type='bpe',  # 'bpe', 'unigram', 'char', 'word'
    character_coverage=0.9995
)

# Salva e carica
tokenizer.save('custom_tokenizer.model')
tokenizer = SentencePieceTokenizer.load('custom_tokenizer.model')
```

## Reinforcement Learning

### Algoritmi

#### DQN (Deep Q-Network)

```python
from src.core.reinforcement_learning.algorithms.dqn import DQN, DQNConfig

# Configurazione
config = DQNConfig(
    state_size=84*84*4,  # Atari frames
    action_size=4,
    hidden_layers=[512, 512],
    learning_rate=0.0001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    memory_size=100000,
    batch_size=32,
    target_update_freq=1000
)

# Creazione dell'agente
agent = DQN(config)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        if len(agent.memory) > batch_size:
            agent.replay()
        
        state = next_state
        total_reward += reward
    
    agent.update_target_network()
```

#### PPO (Proximal Policy Optimization)

```python
from src.core.reinforcement_learning.algorithms.ppo import PPO, PPOConfig

# Configurazione
config = PPOConfig(
    state_size=8,  # CartPole
    action_size=2,
    hidden_layers=[64, 64],
    learning_rate=0.0003,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
    ppo_epochs=4,
    mini_batch_size=64
)

# Creazione dell'agente
agent = PPO(config)

# Training
for episode in range(num_episodes):
    states, actions, rewards, dones, values, log_probs = collect_trajectories(
        agent, env, num_steps=2048
    )
    
    advantages, returns = compute_gae(
        rewards, values, dones, config.gamma, config.gae_lambda
    )
    
    agent.update(states, actions, returns, advantages, log_probs)
```

#### A3C (Asynchronous Actor-Critic)

```python
from src.core.reinforcement_learning.algorithms.a3c import A3C, A3CConfig
import multiprocessing as mp

# Configurazione
config = A3CConfig(
    state_size=84*84*4,
    action_size=6,
    hidden_layers=[256, 256],
    learning_rate=0.0001,
    gamma=0.99,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=40,
    num_workers=mp.cpu_count()
)

# Training distribuito
a3c = A3C(config)
a3c.train(
    env_name='BreakoutNoFrameskip-v4',
    num_episodes=10000,
    save_interval=1000
)
```

### Environments

#### Custom Environment

```python
from src.core.reinforcement_learning.environments.base import BaseEnvironment
import numpy as np

class CustomEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.reset()
    
    def reset(self):
        self.state = np.random.randn(self.state_size)
        self.done = False
        self.step_count = 0
        return self.state
    
    def step(self, action):
        # Implementa la logica dell'environment
        next_state = self.state + np.random.randn(self.state_size) * 0.1
        reward = self.compute_reward(action)
        self.done = self.step_count >= 1000
        self.step_count += 1
        
        self.state = next_state
        return next_state, reward, self.done, {}
    
    def compute_reward(self, action):
        # Logica del reward
        return -np.sum(np.square(self.state))
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"State: {self.state}, Step: {self.step_count}")

# Uso
env = CustomEnvironment(config)
state = env.reset()

for step in range(1000):
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    
    if done:
        break
    
    state = next_state
```

#### Gym Integration

```python
from src.core.reinforcement_learning.environments.gym_wrapper import GymWrapper

# Wrapper per ambienti Gym
env = GymWrapper(
    env_name='CartPole-v1',
    frame_stack=4,  # Stack di frame per Atari
    frame_skip=4,   # Skip frames
    noop_max=30,    # Random no-ops all'inizio
    episodic_life=True,  # Tratta vite come episodi
    clip_rewards=True,   # Clip rewards a [-1, 1]
    normalize_obs=True   # Normalizza osservazioni
)

# Training con wrapper
agent = DQN(config)
trainer = RLTrainer(agent, env)
result = await trainer.train(num_episodes=10000)
```

### Policy Networks

#### Actor-Critic Network

```python
from src.core.reinforcement_learning.policies.actor_critic import ActorCriticNetwork

# Rete Actor-Critic
network = ActorCriticNetwork(
    state_size=8,
    action_size=2,
    hidden_layers=[64, 64],
    activation='tanh',
    output_activation='softmax'  # Per azioni discrete
)

# Forward pass
state = torch.randn(32, 8)
action_probs, value = network(state)

# Sampling azioni
action_dist = torch.distributions.Categorical(action_probs)
action = action_dist.sample()
log_prob = action_dist.log_prob(action)
```

#### Continuous Policy

```python
from src.core.reinforcement_learning.policies.continuous import ContinuousPolicy

# Policy per azioni continue
policy = ContinuousPolicy(
    state_size=17,  # MuJoCo environment
    action_size=6,
    hidden_layers=[256, 256],
    log_std_init=-0.5,
    log_std_min=-20,
    log_std_max=2
)

# Sampling azioni continue
state = torch.randn(32, 17)
mean, log_std = policy(state)
std = log_std.exp()

action_dist = torch.distributions.Normal(mean, std)
action = action_dist.sample()
log_prob = action_dist.log_prob(action).sum(dim=-1)
```

## Utilities

### Model Manager

```python
from src.core.utils.model_manager import ModelManager

# Gestione modelli
manager = ModelManager()

# Salva modello
model_id = await manager.save_model(
    model=trained_model,
    name="my_bert_classifier",
    version="1.0.0",
    metadata={
        "task": "text_classification",
        "dataset": "imdb",
        "accuracy": 0.92
    }
)

# Carica modello
loaded_model = await manager.load_model(
    model_id=model_id,
    device="cuda:0"
)

# Lista modelli
models = await manager.list_models(
    task="text_classification",
    min_accuracy=0.9
)

# Elimina modello
await manager.delete_model(model_id)
```

### Training Utilities

```python
from src.core.utils.training import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    TrainingLogger
)

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    mode='min'
)

# Model checkpoint
checkpoint = ModelCheckpoint(
    filepath='checkpoints/model_{epoch:02d}_{val_loss:.2f}.pt',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# Learning rate monitor
lr_monitor = LearningRateMonitor()

# Training logger
logger = TrainingLogger(
    log_dir='logs',
    experiment_name='bert_classification'
)

# Uso durante il training
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    callbacks=[
        early_stopping,
        checkpoint,
        lr_monitor,
        logger
    ]
)

result = await trainer.fit(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=100
)
```

### Metrics

```python
from src.core.utils.metrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    ConfusionMatrix
)

# Metriche per classificazione
accuracy = Accuracy()
precision = Precision(num_classes=3, average='macro')
recall = Recall(num_classes=3, average='macro')
f1 = F1Score(num_classes=3, average='macro')
auc = AUC()
confusion_matrix = ConfusionMatrix(num_classes=3)

# Calcolo metriche
predictions = model(test_data)
targets = test_labels

accuracy_score = accuracy(predictions, targets)
precision_score = precision(predictions, targets)
recall_score = recall(predictions, targets)
f1_score = f1(predictions, targets)
auc_score = auc(predictions, targets)
cm = confusion_matrix(predictions, targets)

print(f"Accuracy: {accuracy_score:.4f}")
print(f"Precision: {precision_score:.4f}")
print(f"Recall: {recall_score:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"AUC: {auc_score:.4f}")
print(f"Confusion Matrix:\n{cm}")
```

## Esempi Completi

### Classificazione di Testo con BERT

```python
import asyncio
from src.core.transformers.models.bert import BERTForClassification
from src.core.transformers.tokenizers.wordpiece import WordPieceTokenizer
from src.core.utils.training import Trainer
from src.core.utils.metrics import Accuracy, F1Score

async def train_text_classifier():
    # Carica tokenizer e modello
    tokenizer = WordPieceTokenizer.from_pretrained('bert-base-uncased')
    model = BERTForClassification.from_pretrained(
        'bert-base-uncased',
        num_classes=2
    )
    
    # Prepara dati
    train_dataset = TextClassificationDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=512
    )
    
    val_dataset = TextClassificationDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Configura training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=1000
    )
    
    # Metriche
    metrics = {
        'accuracy': Accuracy(),
        'f1': F1Score(num_classes=2, average='macro')
    }
    
    # Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics
    )
    
    result = await trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        batch_size=16
    )
    
    return result

# Esegui training
result = asyncio.run(train_text_classifier())
print(f"Best validation accuracy: {result.best_metrics['accuracy']:.4f}")
```

### Reinforcement Learning con DQN

```python
import gym
from src.core.reinforcement_learning.algorithms.dqn import DQN, DQNConfig
from src.core.reinforcement_learning.environments.gym_wrapper import GymWrapper

def train_dqn_agent():
    # Configura environment
    env = GymWrapper('CartPole-v1')
    
    # Configura DQN
    config = DQNConfig(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_layers=[128, 128],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=100
    )
    
    agent = DQN(config)
    
    # Training loop
    scores = []
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > config.batch_size:
                agent.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        agent.update_target_network()
    
    return agent, scores

# Esegui training
trained_agent, training_scores = train_dqn_agent()
```

## Error Handling

Tutte le funzioni del modulo Core utilizzano il sistema di gestione errori di AIBF:

```python
from src.core.exceptions import (
    ModelError,
    TrainingError,
    ValidationError,
    ConfigurationError
)

try:
    model = BERT.from_pretrained('invalid-model-name')
except ModelError as e:
    logger.error(f"Errore nel caricamento del modello: {e}")
    # Fallback a modello di default
    model = BERT.from_pretrained('bert-base-uncased')

try:
    result = await trainer.train(train_data, val_data)
except TrainingError as e:
    logger.error(f"Errore durante il training: {e}")
    # Riprova con parametri diversi
    result = await trainer.train(train_data, val_data, learning_rate=0.0001)
```

## Performance Tips

1. **Usa GPU quando disponibile**:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```

2. **Ottimizza batch size**:
   ```python
   # Trova batch size ottimale automaticamente
   optimal_batch_size = find_optimal_batch_size(model, sample_data)
   ```

3. **Usa mixed precision**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       output = model(input_data)
       loss = criterion(output, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

4. **Parallelizza quando possibile**:
   ```python
   if torch.cuda.device_count() > 1:
       model = torch.nn.DataParallel(model)
   ```

Per ulteriori dettagli, consulta la [documentazione completa](../guides/) e gli [esempi](../examples/).