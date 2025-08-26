# Guida di Configurazione AIBF

## Panoramica

AIBF utilizza un sistema di configurazione flessibile che supporta:
- File di configurazione YAML
- Variabili d'ambiente
- Configurazione programmatica
- Override dinamici

## Struttura di Configurazione

### File Principali

```
config/
├── config.yaml          # Configurazione principale
├── development.yaml     # Configurazioni per sviluppo
├── production.yaml      # Configurazioni per produzione
├── testing.yaml         # Configurazioni per test
└── local.yaml          # Override locali (git-ignored)
```

### Variabili d'Ambiente

```bash
# File .env
AIBF_ENV=development
AIBF_CONFIG_PATH=config/config.yaml
AIBF_LOG_LEVEL=INFO
```

## Configurazione Base

### config.yaml

```yaml
# Configurazione principale AIBF
app:
  name: "AIBF Framework"
  version: "1.0.0"
  environment: "${AIBF_ENV:development}"
  debug: false

# Database
database:
  url: "${DATABASE_URL:sqlite:///aibf.db}"
  pool_size: 10
  max_overflow: 20
  echo: false

# Logging
logging:
  level: "${LOG_LEVEL:INFO}"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "${LOG_FILE:logs/aibf.log}"
  max_size: "10MB"
  backup_count: 5

# Security
security:
  secret_key: "${SECRET_KEY}"
  jwt_secret: "${JWT_SECRET}"
  jwt_expiration: 3600
  password_hash_rounds: 12
  rate_limit:
    requests_per_minute: 100
    burst_size: 10

# API Configuration
api:
  rest:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    timeout: 30
  websocket:
    host: "0.0.0.0"
    port: 8001
    max_connections: 1000
  grpc:
    host: "0.0.0.0"
    port: 50051
    max_workers: 10
  graphql:
    host: "0.0.0.0"
    port: 8002
    playground: true

# Core AI Configuration
core:
  neural_networks:
    default_activation: "relu"
    default_optimizer: "adam"
    learning_rate: 0.001
    batch_size: 32
  transformers:
    model_cache_dir: "models/cache"
    max_sequence_length: 512
    attention_heads: 8
  reinforcement_learning:
    gamma: 0.99
    epsilon: 0.1
    learning_rate: 0.0001

# Enhancement Modules
enhancement:
  rag:
    chunk_size: 1000
    chunk_overlap: 200
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: "faiss"
  fine_tuning:
    max_epochs: 10
    patience: 3
    validation_split: 0.2
  memory:
    cache_size: 1000
    ttl: 3600
    backend: "redis"

# Agent Configuration
agents:
  planning:
    max_steps: 100
    timeout: 300
    strategy: "hierarchical"
  multi_agent:
    max_agents: 10
    communication_protocol: "message_passing"
  autonomy:
    decision_threshold: 0.8
    learning_rate: 0.01

# Multimodal Configuration
multimodal:
  vision:
    image_size: [224, 224]
    preprocessing: "normalize"
    model: "resnet50"
  audio:
    sample_rate: 16000
    n_mels: 80
    hop_length: 512
  cross_modal:
    fusion_strategy: "attention"
    embedding_dim: 512

# Applications
applications:
  robotics:
    control_frequency: 100
    safety_checks: true
  industrial:
    monitoring_interval: 1
    alert_threshold: 0.95
  edge_ai:
    model_compression: true
    quantization: "int8"

# Emerging Technologies
emerging:
  quantum:
    backend: "qasm_simulator"
    shots: 1024
    optimization_level: 1
  neuromorphic:
    time_step: 0.1
    membrane_potential_threshold: 1.0
  federated_learning:
    rounds: 10
    clients_per_round: 5
    aggregation_strategy: "fedavg"

# Monitoring
monitoring:
  metrics:
    enabled: true
    interval: 60
    retention: "7d"
  alerts:
    enabled: true
    channels: ["email", "slack"]
  performance:
    profiling: false
    memory_tracking: true

# External Services
external:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"
    max_tokens: 1000
  huggingface:
    token: "${HUGGINGFACE_TOKEN}"
    cache_dir: "models/huggingface"
  redis:
    url: "${REDIS_URL:redis://localhost:6379}"
    db: 0
    max_connections: 10
```

## Configurazioni per Ambiente

### development.yaml

```yaml
# Override per sviluppo
app:
  debug: true

logging:
  level: "DEBUG"

api:
  rest:
    workers: 1
    reload: true

monitoring:
  performance:
    profiling: true

external:
  openai:
    model: "gpt-3.5-turbo"  # Modello più economico per dev
```

### production.yaml

```yaml
# Override per produzione
app:
  debug: false

logging:
  level: "WARNING"

api:
  rest:
    workers: 8
    timeout: 60

security:
  rate_limit:
    requests_per_minute: 1000
    burst_size: 50

monitoring:
  alerts:
    enabled: true
    channels: ["email", "slack", "pagerduty"]

external:
  openai:
    model: "gpt-4"  # Modello migliore per produzione
```

## Configurazione Programmatica

### Python Configuration

```python
from src.config import ConfigManager, Config

# Carica configurazione
config = ConfigManager.load()

# Accesso ai valori
api_port = config.api.rest.port
db_url = config.database.url

# Override dinamico
config.set("api.rest.port", 9000)

# Configurazione custom
custom_config = Config({
    "model": {
        "name": "custom-model",
        "parameters": {
            "learning_rate": 0.01,
            "batch_size": 64
        }
    }
})

# Merge configurazioni
config.merge(custom_config)
```

### Validazione Configurazione

```python
from src.config import validate_config

# Validazione automatica
errors = validate_config(config)
if errors:
    for error in errors:
        print(f"Errore di configurazione: {error}")
    exit(1)

# Validazione custom
from pydantic import BaseModel, validator

class APIConfig(BaseModel):
    host: str
    port: int
    workers: int
    
    @validator('port')
    def port_must_be_valid(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port deve essere tra 1 e 65535')
        return v
```

## Gestione Secrets

### Variabili d'Ambiente

```bash
# .env (per sviluppo)
SECRET_KEY=dev_secret_key_here
JWT_SECRET=dev_jwt_secret
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost/aibf
```

### Azure Key Vault

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Configurazione Key Vault
vault_url = "https://your-vault.vault.azure.net/"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=vault_url, credential=credential)

# Carica secrets
secrets = {
    "secret_key": client.get_secret("secret-key").value,
    "jwt_secret": client.get_secret("jwt-secret").value,
    "openai_api_key": client.get_secret("openai-api-key").value
}

# Applica alla configurazione
config.update_secrets(secrets)
```

### AWS Secrets Manager

```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name, region_name="us-east-1"):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except ClientError as e:
        raise e

# Carica secrets
secrets = {
    "database_url": get_secret("aibf/database-url"),
    "openai_api_key": get_secret("aibf/openai-key")
}
```

## Configurazione per Moduli

### Core AI

```yaml
core:
  neural_networks:
    architectures:
      mlp:
        hidden_layers: [128, 64, 32]
        dropout: 0.2
        activation: "relu"
      cnn:
        filters: [32, 64, 128]
        kernel_size: 3
        pooling: "max"
      rnn:
        hidden_size: 256
        num_layers: 2
        bidirectional: true
```

### Quantum Computing

```yaml
emerging:
  quantum:
    providers:
      ibm:
        token: "${IBM_QUANTUM_TOKEN}"
        backend: "ibmq_qasm_simulator"
      google:
        project_id: "${GOOGLE_QUANTUM_PROJECT}"
        processor: "rainbow"
    circuits:
      max_qubits: 20
      max_depth: 100
      optimization_level: 3
```

### Federated Learning

```yaml
emerging:
  federated_learning:
    server:
      host: "0.0.0.0"
      port: 8080
      ssl_enabled: true
    client:
      batch_size: 32
      local_epochs: 5
      learning_rate: 0.01
    privacy:
      differential_privacy: true
      epsilon: 1.0
      delta: 1e-5
```

## Best Practices

### 1. Separazione degli Ambienti

```bash
# Usa file separati per ogni ambiente
AIBF_ENV=development python main.py
AIBF_ENV=production python main.py
```

### 2. Validazione Configurazione

```python
# Sempre validare la configurazione all'avvio
if not config.validate():
    logger.error("Configurazione non valida")
    sys.exit(1)
```

### 3. Configurazione Immutabile

```python
# Usa configurazioni immutabili in produzione
config = ConfigManager.load(immutable=True)
```

### 4. Logging delle Configurazioni

```python
# Log delle configurazioni (senza secrets)
logger.info(f"Configurazione caricata: {config.safe_dict()}")
```

### 5. Hot Reload

```python
# Ricarica configurazione senza restart
config_watcher = ConfigWatcher(config_path)
config_watcher.on_change(lambda: config.reload())
```

## Troubleshooting

### Problemi Comuni

1. **Variabile d'ambiente non trovata**
   ```bash
   export MISSING_VAR=value
   # oppure
   echo "MISSING_VAR=value" >> .env
   ```

2. **Errore di parsing YAML**
   ```bash
   # Valida sintassi YAML
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

3. **Conflitti di configurazione**
   ```python
   # Debug ordine di precedenza
   config.debug_sources()
   ```

### Debug Configurazione

```python
# Stampa configurazione completa
print(config.to_yaml())

# Verifica sorgenti
for key, source in config.get_sources().items():
    print(f"{key}: {source}")

# Valida schema
schema_errors = config.validate_schema()
if schema_errors:
    for error in schema_errors:
        print(f"Schema error: {error}")
```

## Prossimi Passi

1. Leggi l'[Architettura del Sistema](architecture.md)
2. Esplora le [Best Practices](best_practices.md)
3. Consulta la [API Reference](../api/)
4. Prova gli [Esempi di Configurazione](../examples/configuration/)