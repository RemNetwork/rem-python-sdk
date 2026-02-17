"""
REM Vector Database - Quickstart Example

This example shows how to:
1. Create a collection with AES-256-GCM encrypted fields
2. Upsert vectors with metadata
3. Vector similarity search
4. Hybrid search (vector + keyword)
5. Metadata filtering
6. Fetch and delete vectors

Prerequisites:
    pip install rem-vectordb
"""

from rem import REM

# Initialize client
client = REM(
    api_key="rem_your_api_key_here",
    base_url="https://api.getrem.online",
)

# 1. Create a collection with encrypted fields
#    Fields listed in encrypted_fields are AES-256-GCM encrypted at rest
collection = client.create_collection(
    name="products",
    dimension=384,
    metric="cosine",
    encrypted_fields=["email", "pii_data"],  # These fields are encrypted on miners
)
print(f"Created collection: {collection.name} (id={collection.id})")

# 2. Upsert vectors with metadata
collection.upsert([
    {
        "id": "p1",
        "values": [0.1] * 384,  # Replace with real embeddings
        "metadata": {
            "title": "Wireless Headphones",
            "category": "electronics",
            "price": 299.99,
            "description": "Premium noise cancelling wireless headphones",
        },
    },
    {
        "id": "p2",
        "values": [0.2] * 384,
        "metadata": {
            "title": "Bluetooth Speaker",
            "category": "electronics",
            "price": 149.99,
            "description": "Portable waterproof bluetooth speaker",
        },
    },
    {
        "id": "p3",
        "values": [0.3] * 384,
        "metadata": {
            "title": "Python Cookbook",
            "category": "books",
            "price": 39.99,
            "description": "Advanced Python programming recipes",
        },
    },
])
print("Upserted 3 vectors")

# 3. Vector similarity search
results = collection.query(
    vector=[0.15] * 384,
    top_k=3,
    include_metadata=True,
)
print(f"\nVector search ({results.took_ms:.1f}ms):")
for match in results.matches:
    print(f"  {match.id}: score={match.score:.4f}, metadata={match.metadata}")

# 4. Hybrid search: vector similarity + BM25 keyword matching
results = collection.query(
    vector=[0.15] * 384,
    query_text="noise cancelling",  # BM25 keyword boost
    hybrid_alpha=0.5,               # 50% vector, 50% keyword
    top_k=3,
)
print(f"\nHybrid search ({results.took_ms:.1f}ms):")
for match in results.matches:
    print(f"  {match.id}: score={match.score:.4f}")

# 5. Metadata filtering with Pinecone-compatible operators
results = collection.query(
    vector=[0.15] * 384,
    top_k=3,
    filter={
        "$and": [
            {"category": {"$eq": "electronics"}},
            {"price": {"$lte": 200}},
        ]
    },
)
print(f"\nFiltered search ({results.took_ms:.1f}ms):")
for match in results.matches:
    print(f"  {match.id}: score={match.score:.4f}")

# 6. Pure keyword search (no vector needed)
results = collection.query(
    query_text="bluetooth speaker",
    top_k=3,
)
print(f"\nKeyword-only search ({results.took_ms:.1f}ms):")
for match in results.matches:
    print(f"  {match.id}: score={match.score:.4f}")

# 7. Fetch specific vectors by ID
fetched = collection.fetch(ids=["p1", "p2"])
print(f"\nFetched {len(fetched.vectors)} vectors")

# 8. Delete vectors
deleted = collection.delete(ids=["p3"])
print(f"Deleted {deleted.deleted_count} vectors")

# Cleanup
client.delete_collection(collection.id)
client.close()
print("\nDone!")
