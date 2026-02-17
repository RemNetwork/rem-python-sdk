# REM Vector Database - Python SDK

[![PyPI version](https://badge.fury.io/py/rem-vectordb.svg)](https://pypi.org/project/rem-vectordb/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for **REM Network** — the decentralized vector database for AI applications. A Pinecone-compatible API powered by 2,000+ distributed miners on the Sui blockchain.

## Installation

```bash
pip install rem-vectordb
```

With LangChain support:
```bash
pip install rem-vectordb[langchain]
```

With LlamaIndex support:
```bash
pip install rem-vectordb[llamaindex]
```

## Quick Start

```python
from rem import REM

client = REM(api_key="rem_your_api_key")

# Create a collection
collection = client.create_collection("my-docs", dimension=1536)

# Upsert vectors
collection.upsert([
    {"id": "doc1", "values": [0.1, 0.2, ...], "metadata": {"title": "Hello"}},
    {"id": "doc2", "values": [0.3, 0.4, ...], "metadata": {"title": "World"}},
])

# Semantic search
results = collection.query(vector=[0.1, 0.2, ...], top_k=10)
for match in results.matches:
    print(f"{match.id}: {match.score:.4f}")
```

## Features

### Encrypted Metadata (AES-256-GCM)

Protect sensitive metadata fields with per-namespace encryption. Miners never see your plaintext data.

```python
collection = client.create_collection(
    name="secure-docs",
    dimension=1536,
    encrypted_fields=["text", "user_id", "email"]
)
```

### Hybrid Search (Vector + Keyword)

Combine semantic vector search with BM25 keyword matching via Reciprocal Rank Fusion.

```python
results = collection.query(
    vector=[0.1, 0.2, ...],
    query_text="machine learning",
    hybrid_alpha=0.7,  # 0.0=pure keyword, 1.0=pure vector
    top_k=10,
)
```

### Metadata Filtering

Pinecone-compatible filter operators: `$eq`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`.

```python
results = collection.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    filter={
        "$and": [
            {"category": {"$eq": "science"}},
            {"year": {"$gte": 2020}}
        ]
    },
)
```

### Batch Queries

Execute up to 10 queries in a single API call for recommendation systems and AI agents.

```python
results = collection.query_batch([
    {"vector": [0.1, ...], "top_k": 5},
    {"vector": [0.3, ...], "top_k": 5, "filter": {"type": "article"}},
    {"query_text": "neural networks", "top_k": 3},
])
```

### Fetch & Delete

Retrieve or remove vectors by ID.

```python
# Fetch vectors
fetched = collection.fetch(ids=["doc1", "doc2"])
for v in fetched.vectors:
    print(f"{v.id}: {v.metadata}")

# Delete vectors
result = collection.delete(ids=["doc1"])
print(f"Deleted {result.deleted_count} vectors")
```

### Async Support

Full async/await interface for high-throughput applications.

```python
from rem import AsyncREM

async with AsyncREM(api_key="rem_xxx") as client:
    collection = await client.create_collection("my-docs", dimension=1536)
    await collection.upsert([...])
    results = await collection.query(vector=[...], top_k=10)
```

## Framework Integrations

### LangChain

```python
from langchain_openai import OpenAIEmbeddings
from rem.integrations.langchain import REMVectorStore

store = REMVectorStore(
    api_key="rem_xxx",
    collection_name="docs",
    embedding=OpenAIEmbeddings(),
)

# Add documents
store.add_texts(["Hello world", "REM is great"], metadatas=[{"source": "test"}])

# Similarity search
results = store.similarity_search("greeting", k=5)

# Use in RAG chains
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever())
```

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex
from rem.integrations.llamaindex import REMVectorStore

vector_store = REMVectorStore(api_key="rem_xxx", collection_name="docs")
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
response = query_engine.query("What is REM Network?")
```

## API Reference

### Client

```python
client = REM(
    api_key="rem_xxx",                           # Required (starts with rem_)
    base_url="https://api.getrem.online",         # Default
    timeout=30.0,                                  # Seconds
)
```

### Collections

| Method | Description |
|--------|-------------|
| `client.create_collection(name, dimension, metric, encrypted_fields)` | Create collection |
| `client.get_collection(id)` | Get by ID |
| `client.list_collections()` | List all |
| `client.delete_collection(id)` | Delete |

### Vectors

| Method | Description |
|--------|-------------|
| `collection.upsert(vectors)` | Insert/update vectors |
| `collection.query(vector, top_k, filter, query_text, hybrid_alpha)` | Search |
| `collection.query_batch(queries)` | Batch search (up to 10) |
| `collection.fetch(ids)` | Fetch by ID |
| `collection.delete(ids)` | Delete by ID |
| `collection.stats()` | Collection stats |

### Distance Metrics

- `cosine` (default) — Normalized similarity
- `euclidean` — L2 distance
- `dot_product` — Inner product

## Getting Started

1. Sign up at [app.getrem.online](https://app.getrem.online) to get your API key
2. You get $20 in free API credits
3. `pip install rem-vectordb`
4. Start building!

Full documentation: [app.getrem.online/docs](https://app.getrem.online/docs)

## Links

- [Homepage](https://getrem.online)
- [Platform & Docs](https://app.getrem.online)
- [Network Explorer](https://getrem.online/explorer.html)
- [Discord](https://discord.gg/9ndMQY4PYP)
- [Twitter/X](https://x.com/RemNetwork)
- [Telegram](https://t.me/RemDepin)

## License

MIT - See [LICENSE](LICENSE) for details.

Built by [BeClever OÜ](https://getrem.online) (Estonia).
