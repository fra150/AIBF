# Guida di Installazione AIBF

## Requisiti di Sistema

### Requisiti Minimi
- **Python**: 3.9 o superiore
- **RAM**: 8 GB (16 GB raccomandati)
- **Storage**: 10 GB di spazio libero
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

### Requisiti Raccomandati
- **Python**: 3.11+
- **RAM**: 32 GB
- **GPU**: NVIDIA con CUDA 11.8+ (per accelerazione)
- **Storage**: SSD con 50 GB di spazio libero

## Installazione

### 1. Clonare il Repository

```bash
git clone https://github.com/your-repo/aibf.git
cd aibf
```

### 2. Creare Ambiente Virtuale

```bash
# Con venv
python -m venv aibf_env

# Attivazione
# Windows
aibf_env\Scripts\activate
# macOS/Linux
source aibf_env/bin/activate
```

### 3. Installare Dipendenze

```bash
# Installazione base
pip install -r requirements.txt

# Per sviluppo
pip install -r requirements-dev.txt

# Per funzionalità GPU (opzionale)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Configurazione

```bash
# Copia il file di configurazione esempio
cp .env.example .env

# Modifica le configurazioni
nano .env  # o il tuo editor preferito
```

#### Configurazioni Essenziali

```env
# Database
DATABASE_URL=sqlite:///aibf.db

# API Keys (opzionali)
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/aibf.log

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret

# Performance
MAX_WORKERS=4
CACHE_SIZE=1000
```

### 5. Inizializzazione Database

```bash
# Crea le tabelle del database
python scripts/init_db.py

# Carica dati di esempio (opzionale)
python scripts/load_sample_data.py
```

### 6. Verifica Installazione

```bash
# Test di base
python -m pytest tests/test_installation.py

# Test completo
python -m pytest

# Avvio del framework
python main.py --test
```

## Installazione con Docker

### 1. Usando Docker Compose (Raccomandato)

```bash
# Avvio completo
docker-compose up -d

# Solo il framework
docker-compose up aibf

# Con rebuild
docker-compose up --build
```

### 2. Docker Manuale

```bash
# Build dell'immagine
docker build -t aibf:latest .

# Avvio del container
docker run -d \
  --name aibf \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  aibf:latest
```

## Configurazioni Avanzate

### GPU Support

```bash
# Verifica CUDA
nvidia-smi

# Installa PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verifica installazione GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Quantum Computing

```bash
# Installa Qiskit
pip install qiskit qiskit-aer

# Configura IBM Quantum (opzionale)
export QISKIT_IBM_TOKEN=your_ibm_token
```

### Distributed Computing

```bash
# Installa Ray per distributed computing
pip install ray[default]

# Configura cluster (opzionale)
ray start --head --port=6379
```

## Troubleshooting

### Problemi Comuni

#### 1. Errori di Dipendenze
```bash
# Aggiorna pip
pip install --upgrade pip setuptools wheel

# Reinstalla dipendenze
pip install --force-reinstall -r requirements.txt
```

#### 2. Problemi di Memoria
```bash
# Riduce l'uso di memoria
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Configura swap (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. Errori di Permessi
```bash
# Windows: Esegui come amministratore
# macOS/Linux:
sudo chown -R $USER:$USER .
chmod +x scripts/*.py
```

### Log di Debug

```bash
# Abilita logging dettagliato
export LOG_LEVEL=DEBUG

# Avvia con profiling
python -m cProfile -o profile.stats main.py

# Analizza performance
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## Aggiornamenti

```bash
# Aggiorna il codice
git pull origin main

# Aggiorna dipendenze
pip install --upgrade -r requirements.txt

# Migra database (se necessario)
python scripts/migrate_db.py

# Riavvia servizi
docker-compose restart  # se usi Docker
# oppure
python main.py --restart
```

## Prossimi Passi

1. Leggi la [Guida di Configurazione](configuration.md)
2. Esplora l'[Architettura del Sistema](architecture.md)
3. Prova il [Tutorial di Getting Started](../tutorials/getting_started.md)
4. Consulta gli [Esempi](../examples/)

## Supporto

Se incontri problemi durante l'installazione:

- Controlla i [Issues noti](https://github.com/your-repo/aibf/issues)
- Consulta le [FAQ](faq.md)
- Chiedi aiuto nelle [Discussioni](https://github.com/your-repo/aibf/discussions)
- Contatta il supporto: support@aibf.dev