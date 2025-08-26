# Enhancement API Reference

## Panoramica

Il modulo Enhancement di AIBF fornisce funzionalità avanzate per migliorare le prestazioni dei modelli AI, inclusi RAG (Retrieval-Augmented Generation), fine-tuning e gestione della memoria.

## RAG (Retrieval-Augmented Generation)

### Vector Store

#### FAISS Vector Store

```python
from src.enhancement.rag.vector_stores.faiss_store import FAISSVectorStore
from src.enhancement.rag.embeddings.sentence_transformers import SentenceTransformerEmbeddings

# Configura embeddings
embeddings = SentenceTransformerEmbeddings(
    model_name='all-MiniLM-L6-v2',
    device='cuda'
)

# Crea vector store
vector_store = FAISSVectorStore(
    embeddings=embeddings,
    dimension=384,  # Dimensione degli embeddings
    index_type='IVF',  # 'Flat', 'IVF', 'HNSW'
    metric='cosine',  # 'cosine', 'euclidean', 'dot_product'
    nlist=100  # Numero di cluster per IVF
)

# Aggiungi documenti
documents = [
    "Il machine learning è un sottoinsieme dell'intelligenza artificiale.",
    "Le reti neurali sono ispirate al funzionamento del cervello umano.",
    "Il deep learning utilizza reti neurali profonde."
]

metadata = [
    {"source": "ml_basics.txt", "category": "introduction"},
    {"source": "neural_nets.txt", "category": "architecture"},
    {"source": "deep_learning.txt", "category": "advanced"}
]

# Indicizza documenti
await vector_store.add_documents(documents, metadata)

# Ricerca semantica
query = "Cosa sono le reti neurali?"
results = await vector_store.similarity_search(
    query=query,
    k=3,  # Numero di risultati
    score_threshold=0.7,  # Soglia di similarità
    filter_metadata={"category": "architecture"}  # Filtro opzionale
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text}")
    print(f"Metadata: {result.metadata}")
    print("---")
```

#### Chroma Vector Store

```python
from src.enhancement.rag.vector_stores.chroma_store import ChromaVectorStore

# Configura Chroma
vector_store = ChromaVectorStore(
    collection_name="my_documents",
    embeddings=embeddings,
    persist_directory="./chroma_db",
    client_settings={
        "chroma_db_impl": "duckdb+parquet",
        "persist_directory": "./chroma_db"
    }
)

# Operazioni simili a FAISS
await vector_store.add_documents(documents, metadata)
results = await vector_store.similarity_search(query, k=5)
```

#### Pinecone Vector Store

```python
from src.enhancement.rag.vector_stores.pinecone_store import PineconeVectorStore

# Configura Pinecone (richiede API key)
vector_store = PineconeVectorStore(
    api_key="your-pinecone-api-key",
    environment="us-west1-gcp",
    index_name="aibf-documents",
    embeddings=embeddings,
    dimension=384,
    metric="cosine",
    pod_type="p1.x1"
)

# Crea indice se non esiste
await vector_store.create_index_if_not_exists()

# Usa come gli altri vector stores
await vector_store.add_documents(documents, metadata)
results = await vector_store.similarity_search(query, k=5)
```

### Document Loaders

#### Text Loader

```python
from src.enhancement.rag.document_loaders.text_loader import TextLoader

# Carica file di testo
loader = TextLoader(
    file_path="documents/knowledge_base.txt",
    encoding="utf-8",
    chunk_size=1000,  # Dimensione dei chunk
    chunk_overlap=200,  # Sovrapposizione tra chunk
    separator="\n\n"  # Separatore per chunk
)

documents = await loader.load()
print(f"Caricati {len(documents)} documenti")

for doc in documents[:3]:
    print(f"Chunk: {doc.text[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

#### PDF Loader

```python
from src.enhancement.rag.document_loaders.pdf_loader import PDFLoader

# Carica PDF
loader = PDFLoader(
    file_path="documents/manual.pdf",
    extract_images=False,  # Estrai anche immagini
    extract_tables=True,   # Estrai tabelle
    chunk_size=1500,
    chunk_overlap=300
)

documents = await loader.load()

# Accesso a metadati specifici del PDF
for doc in documents:
    print(f"Pagina: {doc.metadata['page']}")
    print(f"Testo: {doc.text[:200]}...")
```

#### Web Loader

```python
from src.enhancement.rag.document_loaders.web_loader import WebLoader

# Carica contenuto web
loader = WebLoader(
    urls=[
        "https://example.com/article1",
        "https://example.com/article2"
    ],
    extract_links=True,  # Estrai link interni
    follow_links=False,  # Segui link automaticamente
    max_depth=2,         # Profondità massima per link
    chunk_size=1200
)

documents = await loader.load()

for doc in documents:
    print(f"URL: {doc.metadata['url']}")
    print(f"Titolo: {doc.metadata['title']}")
    print(f"Testo: {doc.text[:150]}...")
```

#### Database Loader

```python
from src.enhancement.rag.document_loaders.database_loader import DatabaseLoader

# Carica da database
loader = DatabaseLoader(
    connection_string="postgresql://user:pass@localhost/db",
    query="SELECT id, title, content, created_at FROM articles WHERE published = true",
    text_columns=["title", "content"],  # Colonne da usare come testo
    metadata_columns=["id", "created_at"],  # Colonne per metadata
    chunk_size=1000
)

documents = await loader.load()
```

### Retrievers

#### Dense Retriever

```python
from src.enhancement.rag.retrievers.dense_retriever import DenseRetriever

# Retriever basato su embeddings densi
retriever = DenseRetriever(
    vector_store=vector_store,
    top_k=10,  # Numero di documenti da recuperare
    score_threshold=0.6,  # Soglia minima di similarità
    diversity_threshold=0.8,  # Soglia per diversità (MMR)
    use_mmr=True,  # Usa Maximum Marginal Relevance
    lambda_mult=0.5  # Bilanciamento rilevanza/diversità
)

# Recupera documenti
query = "Come funziona il machine learning?"
retrieved_docs = await retriever.retrieve(query)

for doc in retrieved_docs:
    print(f"Score: {doc.score:.3f}")
    print(f"Text: {doc.text[:200]}...")
```

#### Sparse Retriever (BM25)

```python
from src.enhancement.rag.retrievers.sparse_retriever import SparseRetriever

# Retriever basato su BM25
retriever = SparseRetriever(
    documents=documents,
    k1=1.2,  # Parametro BM25
    b=0.75,  # Parametro BM25
    top_k=10,
    language="italian",  # Per stemming
    remove_stopwords=True
)

# Indicizza documenti
await retriever.build_index()

# Recupera documenti
retrieved_docs = await retriever.retrieve(query)
```

#### Hybrid Retriever

```python
from src.enhancement.rag.retrievers.hybrid_retriever import HybridRetriever

# Combina dense e sparse retrieval
retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    dense_weight=0.7,  # Peso per dense retrieval
    sparse_weight=0.3,  # Peso per sparse retrieval
    top_k=15,
    rerank=True,  # Usa reranking
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

retrieved_docs = await retriever.retrieve(query)
```

### RAG Chain

#### Basic RAG Chain

```python
from src.enhancement.rag.chains.basic_rag import BasicRAGChain
from src.core.transformers.models.gpt import GPT

# Configura modello generativo
generator = GPT.from_pretrained('gpt-3.5-turbo')

# Crea RAG chain
rag_chain = BasicRAGChain(
    retriever=retriever,
    generator=generator,
    prompt_template="""
    Contesto: {context}
    
    Domanda: {question}
    
    Rispondi alla domanda basandoti sul contesto fornito. Se il contesto non contiene informazioni sufficienti, dillo chiaramente.
    
    Risposta:
    """,
    max_context_length=2000,  # Lunghezza massima del contesto
    context_separator="\n\n---\n\n"  # Separatore tra documenti
)

# Genera risposta
question = "Quali sono i vantaggi del deep learning?"
response = await rag_chain.generate(
    question=question,
    max_length=300,
    temperature=0.7
)

print(f"Domanda: {question}")
print(f"Risposta: {response.answer}")
print(f"Documenti utilizzati: {len(response.source_documents)}")
print(f"Score di confidenza: {response.confidence_score:.3f}")
```

#### Conversational RAG

```python
from src.enhancement.rag.chains.conversational_rag import ConversationalRAGChain

# RAG con memoria conversazionale
rag_chain = ConversationalRAGChain(
    retriever=retriever,
    generator=generator,
    memory_size=10,  # Numero di turni da ricordare
    context_window=3000,
    prompt_template="""
    Cronologia conversazione:
    {chat_history}
    
    Contesto rilevante:
    {context}
    
    Domanda corrente: {question}
    
    Rispondi considerando sia la cronologia che il contesto.
    
    Risposta:
    """
)

# Conversazione multi-turno
conversation_id = "user_123"

# Primo turno
response1 = await rag_chain.generate(
    question="Cos'è il machine learning?",
    conversation_id=conversation_id
)

# Secondo turno (con memoria)
response2 = await rag_chain.generate(
    question="Quali sono le sue applicazioni principali?",
    conversation_id=conversation_id
)

print(f"Risposta 1: {response1.answer}")
print(f"Risposta 2: {response2.answer}")
```

## Fine-Tuning

### LoRA (Low-Rank Adaptation)

```python
from src.enhancement.fine_tuning.lora import LoRAConfig, LoRATrainer
from src.core.transformers.models.bert import BERT

# Configura LoRA
lora_config = LoRAConfig(
    r=16,  # Rank della decomposizione
    alpha=32,  # Scaling factor
    dropout=0.1,
    target_modules=["query", "value"],  # Moduli da adattare
    bias="none",  # "none", "all", "lora_only"
    task_type="SEQUENCE_CLASSIFICATION"
)

# Carica modello base
base_model = BERT.from_pretrained('bert-base-uncased')

# Applica LoRA
lora_model = LoRATrainer.prepare_model(base_model, lora_config)

# Training
trainer = LoRATrainer(
    model=lora_model,
    config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

result = await trainer.train(
    epochs=5,
    learning_rate=3e-4,
    batch_size=16,
    warmup_steps=100
)

# Salva solo i parametri LoRA (molto più piccoli)
await trainer.save_lora_weights("./lora_weights")

# Carica LoRA weights
lora_model = LoRATrainer.load_lora_model(
    base_model_path="bert-base-uncased",
    lora_weights_path="./lora_weights"
)
```

### QLoRA (Quantized LoRA)

```python
from src.enhancement.fine_tuning.qlora import QLoRAConfig, QLoRATrainer

# Configura QLoRA con quantizzazione
qlora_config = QLoRAConfig(
    r=64,
    alpha=128,
    dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4"
    }
)

# Carica modello con quantizzazione
model = QLoRATrainer.load_quantized_model(
    model_name="meta-llama/Llama-2-7b-hf",
    config=qlora_config
)

# Training efficiente in memoria
trainer = QLoRATrainer(model, qlora_config)
result = await trainer.train(
    train_dataset=instruction_dataset,
    epochs=3,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    dataloader_pin_memory=False
)
```

### Adapter Tuning

```python
from src.enhancement.fine_tuning.adapters import AdapterConfig, AdapterTrainer

# Configura adapter
adapter_config = AdapterConfig(
    adapter_size=64,  # Dimensione bottleneck
    non_linearity="relu",
    dropout=0.1,
    init_weights="bert",  # Inizializzazione pesi
    leave_out=[11]  # Layer da escludere
)

# Aggiungi adapter al modello
model_with_adapters = AdapterTrainer.add_adapters(
    model=base_model,
    adapter_name="task_adapter",
    config=adapter_config
)

# Training
trainer = AdapterTrainer(
    model=model_with_adapters,
    adapter_name="task_adapter"
)

result = await trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    epochs=10,
    learning_rate=1e-3
)

# Salva solo l'adapter
await trainer.save_adapter("./task_adapter")

# Carica adapter su modello base
model_with_adapters = AdapterTrainer.load_adapter(
    base_model=base_model,
    adapter_path="./task_adapter"
)
```

### Prefix Tuning

```python
from src.enhancement.fine_tuning.prefix_tuning import PrefixTuningConfig, PrefixTuner

# Configura prefix tuning
prefix_config = PrefixTuningConfig(
    num_virtual_tokens=20,  # Numero di token virtuali
    encoder_hidden_size=768,
    decoder_hidden_size=768,
    prefix_projection=True,  # Usa proiezione MLP
    prefix_hidden_size=512
)

# Applica prefix tuning
model_with_prefix = PrefixTuner.prepare_model(
    model=base_model,
    config=prefix_config
)

# Training
trainer = PrefixTuner(
    model=model_with_prefix,
    config=prefix_config
)

result = await trainer.train(
    train_dataset=generation_dataset,
    epochs=5,
    learning_rate=5e-4
)
```

### P-Tuning v2

```python
from src.enhancement.fine_tuning.p_tuning_v2 import PTuningV2Config, PTuningV2Trainer

# Configura P-Tuning v2
ptuning_config = PTuningV2Config(
    num_virtual_tokens=100,
    token_dim=768,
    num_transformer_submodules=2,
    num_attention_heads=12,
    num_layers=2,
    dropout=0.1
)

# Applica P-Tuning v2
model_with_ptuning = PTuningV2Trainer.prepare_model(
    model=base_model,
    config=ptuning_config
)

# Training
trainer = PTuningV2Trainer(
    model=model_with_ptuning,
    config=ptuning_config
)

result = await trainer.train(
    train_dataset=train_dataset,
    epochs=8,
    learning_rate=1e-3
)
```

## Memory Management

### Memory Bank

```python
from src.enhancement.memory.memory_bank import MemoryBank, MemoryConfig

# Configura memory bank
memory_config = MemoryConfig(
    capacity=10000,  # Numero massimo di memorie
    embedding_dim=768,
    similarity_threshold=0.8,
    decay_factor=0.99,  # Decadimento temporale
    clustering_enabled=True,
    cluster_update_freq=100
)

memory_bank = MemoryBank(memory_config)

# Aggiungi memoria
memory_embedding = torch.randn(768)
memory_content = "Informazione importante da ricordare"
memory_metadata = {
    "timestamp": time.time(),
    "importance": 0.9,
    "source": "user_interaction"
}

memory_id = await memory_bank.store(
    embedding=memory_embedding,
    content=memory_content,
    metadata=memory_metadata
)

# Recupera memorie simili
query_embedding = torch.randn(768)
similar_memories = await memory_bank.retrieve(
    query_embedding=query_embedding,
    top_k=5,
    min_similarity=0.7
)

for memory in similar_memories:
    print(f"Similarity: {memory.similarity:.3f}")
    print(f"Content: {memory.content}")
    print(f"Age: {memory.age} steps")
```

### Episodic Memory

```python
from src.enhancement.memory.episodic_memory import EpisodicMemory, Episode

# Memoria episodica per agenti
episodic_memory = EpisodicMemory(
    max_episodes=1000,
    embedding_dim=512,
    temporal_decay=0.95,
    importance_threshold=0.5
)

# Crea episodio
episode = Episode(
    state=current_state,
    action=action_taken,
    reward=reward_received,
    next_state=next_state,
    context="Situazione di emergenza",
    importance_score=0.8
)

# Memorizza episodio
episode_id = await episodic_memory.store_episode(episode)

# Recupera episodi simili
similar_episodes = await episodic_memory.retrieve_similar(
    current_state=current_state,
    context="Situazione di emergenza",
    top_k=3
)

# Aggiorna importanza basata su outcome
await episodic_memory.update_importance(
    episode_id=episode_id,
    new_importance=0.9
)
```

### Working Memory

```python
from src.enhancement.memory.working_memory import WorkingMemory

# Memoria di lavoro per task correnti
working_memory = WorkingMemory(
    capacity=7,  # Limite classico 7±2
    decay_rate=0.1,
    refresh_on_access=True
)

# Aggiungi item alla memoria di lavoro
item_id = working_memory.add_item(
    content="Obiettivo corrente: completare il task",
    priority=0.9,
    item_type="goal"
)

# Accedi a item (refresh automatico)
item = working_memory.get_item(item_id)

# Lista tutti gli item attivi
active_items = working_memory.get_active_items()

# Aggiorna priorità
working_memory.update_priority(item_id, new_priority=0.95)

# Rimuovi item
working_memory.remove_item(item_id)
```

### Long-term Memory

```python
from src.enhancement.memory.long_term_memory import LongTermMemory

# Memoria a lungo termine con consolidamento
long_term_memory = LongTermMemory(
    storage_backend="postgresql",  # "sqlite", "postgresql", "mongodb"
    connection_string="postgresql://user:pass@localhost/memory_db",
    consolidation_threshold=10,  # Accessi per consolidamento
    forgetting_curve_enabled=True,
    compression_enabled=True
)

# Memorizza informazione
memory_id = await long_term_memory.store(
    content="Conoscenza importante acquisita",
    category="learned_facts",
    associations=["machine_learning", "neural_networks"],
    strength=0.8
)

# Recupera con associazioni
related_memories = await long_term_memory.retrieve_by_association(
    associations=["machine_learning"],
    min_strength=0.5
)

# Consolida memoria (aumenta forza)
await long_term_memory.consolidate(memory_id)

# Ricerca semantica
search_results = await long_term_memory.semantic_search(
    query="reti neurali",
    top_k=10
)
```

### Memory Consolidation

```python
from src.enhancement.memory.consolidation import MemoryConsolidator

# Consolidatore per trasferire da working a long-term
consolidator = MemoryConsolidator(
    working_memory=working_memory,
    long_term_memory=long_term_memory,
    consolidation_criteria={
        "min_access_count": 3,
        "min_importance": 0.7,
        "max_age_hours": 24
    },
    consolidation_interval=3600  # Ogni ora
)

# Avvia consolidamento automatico
await consolidator.start_automatic_consolidation()

# Consolidamento manuale
consolidated_count = await consolidator.consolidate_memories()
print(f"Consolidate {consolidated_count} memorie")

# Statistiche memoria
stats = await consolidator.get_memory_stats()
print(f"Working memory: {stats['working_memory_items']} items")
print(f"Long-term memory: {stats['long_term_memory_items']} items")
print(f"Consolidation rate: {stats['consolidation_rate']:.2f}/hour")
```

## Utilities

### Embedding Models

```python
from src.enhancement.rag.embeddings import (
    SentenceTransformerEmbeddings,
    OpenAIEmbeddings,
    HuggingFaceEmbeddings
)

# Sentence Transformers
st_embeddings = SentenceTransformerEmbeddings(
    model_name='all-mpnet-base-v2',
    device='cuda',
    normalize_embeddings=True,
    batch_size=32
)

# OpenAI Embeddings
openai_embeddings = OpenAIEmbeddings(
    api_key="your-openai-api-key",
    model="text-embedding-ada-002",
    chunk_size=1000
)

# Hugging Face Embeddings
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# Genera embeddings
texts = ["Primo testo", "Secondo testo"]
embeddings = await st_embeddings.embed_documents(texts)
query_embedding = await st_embeddings.embed_query("Query di ricerca")
```

### Text Splitters

```python
from src.enhancement.rag.text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SemanticTextSplitter
)

# Splitter ricorsivo per caratteri
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
    keep_separator=True
)

# Splitter basato su token
token_splitter = TokenTextSplitter(
    chunk_size=512,  # Token
    chunk_overlap=50,
    tokenizer_name="gpt-3.5-turbo",
    allowed_special_tokens={"<|endoftext|>"}
)

# Splitter semantico
semantic_splitter = SemanticTextSplitter(
    embeddings=embeddings,
    chunk_size=1000,
    similarity_threshold=0.8,
    min_chunk_size=100
)

# Usa splitter
long_text = "Testo molto lungo da dividere..."
chunks = char_splitter.split_text(long_text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} caratteri")
    print(f"Contenuto: {chunk[:100]}...")
```

### Rerankers

```python
from src.enhancement.rag.rerankers import (
    CrossEncoderReranker,
    ColBERTReranker,
    CohereReranker
)

# Cross-encoder reranker
cross_encoder = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda",
    batch_size=16
)

# ColBERT reranker
colbert_reranker = ColBERTReranker(
    model_name="colbert-ir/colbertv2.0",
    device="cuda",
    max_length=512
)

# Cohere reranker
cohere_reranker = CohereReranker(
    api_key="your-cohere-api-key",
    model="rerank-english-v2.0"
)

# Rerank documenti
query = "Domanda di ricerca"
documents = retrieved_docs  # Da retriever

reranked_docs = await cross_encoder.rerank(
    query=query,
    documents=documents,
    top_k=5
)

for doc in reranked_docs:
    print(f"Rerank Score: {doc.rerank_score:.3f}")
    print(f"Original Score: {doc.original_score:.3f}")
    print(f"Text: {doc.text[:150]}...")
```

## Esempi Completi

### Sistema RAG Completo

```python
import asyncio
from src.enhancement.rag import (
    FAISSVectorStore,
    SentenceTransformerEmbeddings,
    PDFLoader,
    HybridRetriever,
    ConversationalRAGChain
)
from src.core.transformers.models.gpt import GPT

async def setup_rag_system():
    # 1. Configura embeddings
    embeddings = SentenceTransformerEmbeddings(
        model_name='all-mpnet-base-v2',
        device='cuda'
    )
    
    # 2. Carica documenti
    pdf_loader = PDFLoader(
        file_path="knowledge_base.pdf",
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = await pdf_loader.load()
    
    # 3. Crea vector store
    vector_store = FAISSVectorStore(
        embeddings=embeddings,
        dimension=768,
        index_type='HNSW'
    )
    await vector_store.add_documents([doc.text for doc in documents])
    
    # 4. Configura retriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        top_k=10,
        rerank=True
    )
    
    # 5. Configura generatore
    generator = GPT.from_pretrained('gpt-3.5-turbo')
    
    # 6. Crea RAG chain
    rag_chain = ConversationalRAGChain(
        retriever=retriever,
        generator=generator,
        memory_size=10
    )
    
    return rag_chain

# Uso del sistema
rag_system = await setup_rag_system()

# Conversazione
response = await rag_system.generate(
    question="Quali sono i principi del machine learning?",
    conversation_id="user_123"
)

print(f"Risposta: {response.answer}")
print(f"Fonti: {[doc.metadata['source'] for doc in response.source_documents]}")
```

### Fine-tuning con LoRA

```python
import asyncio
from src.enhancement.fine_tuning.lora import LoRAConfig, LoRATrainer
from src.core.transformers.models.bert import BERTForClassification
from torch.utils.data import DataLoader

async def fine_tune_with_lora():
    # 1. Carica modello base
    base_model = BERTForClassification.from_pretrained(
        'bert-base-uncased',
        num_classes=3
    )
    
    # 2. Configura LoRA
    lora_config = LoRAConfig(
        r=16,
        alpha=32,
        dropout=0.1,
        target_modules=["query", "value"],
        task_type="SEQUENCE_CLASSIFICATION"
    )
    
    # 3. Applica LoRA
    lora_model = LoRATrainer.prepare_model(base_model, lora_config)
    
    # 4. Prepara dati
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16)
    
    # 5. Training
    trainer = LoRATrainer(
        model=lora_model,
        config=lora_config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader
    )
    
    result = await trainer.train(
        epochs=5,
        learning_rate=3e-4,
        warmup_steps=100,
        save_steps=500
    )
    
    # 6. Salva modello
    await trainer.save_lora_weights("./fine_tuned_lora")
    
    return result

# Esegui fine-tuning
result = await fine_tune_with_lora()
print(f"Best accuracy: {result.best_metrics['accuracy']:.4f}")
```

### Sistema di Memoria Avanzato

```python
import asyncio
from src.enhancement.memory import (
    MemoryBank,
    EpisodicMemory,
    WorkingMemory,
    LongTermMemory,
    MemoryConsolidator
)

async def setup_memory_system():
    # 1. Configura componenti memoria
    working_memory = WorkingMemory(capacity=7)
    
    long_term_memory = LongTermMemory(
        storage_backend="postgresql",
        connection_string="postgresql://user:pass@localhost/memory"
    )
    
    episodic_memory = EpisodicMemory(
        max_episodes=1000,
        embedding_dim=768
    )
    
    memory_bank = MemoryBank(
        capacity=10000,
        embedding_dim=768
    )
    
    # 2. Configura consolidatore
    consolidator = MemoryConsolidator(
        working_memory=working_memory,
        long_term_memory=long_term_memory
    )
    
    # 3. Avvia consolidamento automatico
    await consolidator.start_automatic_consolidation()
    
    return {
        'working': working_memory,
        'long_term': long_term_memory,
        'episodic': episodic_memory,
        'bank': memory_bank,
        'consolidator': consolidator
    }

# Uso del sistema di memoria
memory_system = await setup_memory_system()

# Aggiungi memoria di lavoro
item_id = memory_system['working'].add_item(
    content="Obiettivo: rispondere alla domanda dell'utente",
    priority=0.9
)

# Memorizza episodio
episode = Episode(
    state=current_state,
    action=action,
    reward=reward,
    context="Interazione utente"
)
episode_id = await memory_system['episodic'].store_episode(episode)

# Recupera memorie correlate
similar_memories = await memory_system['bank'].retrieve(
    query_embedding=query_embedding,
    top_k=5
)
```

Per ulteriori dettagli e esempi avanzati, consulta la [documentazione completa](../guides/) e gli [esempi pratici](../examples/).