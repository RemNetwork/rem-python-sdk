"""
REM Vector Database - Async Example

Shows how to use the async client for high-throughput applications.

Prerequisites:
    pip install rem-vectordb
"""

import asyncio
from rem import AsyncREM


async def main():
    # Use async context manager for automatic cleanup
    async with AsyncREM(api_key="rem_your_api_key_here") as client:

        # Create collection
        collection = await client.create_collection(
            name="async-demo",
            dimension=384,  # e.g., sentence-transformers/all-MiniLM-L6-v2
            metric="cosine",
        )
        print(f"Created: {collection}")

        # Batch upsert
        vectors = [
            {"id": f"vec-{i}", "values": [float(i) / 100] * 384, "metadata": {"index": i}}
            for i in range(100)
        ]
        result = await collection.upsert(vectors)
        print(f"Upserted {result.upserted_count} vectors")

        # Parallel queries
        queries = [
            collection.query(vector=[float(i) / 100] * 384, top_k=5)
            for i in range(10)
        ]
        results = await asyncio.gather(*queries)
        for i, r in enumerate(results):
            print(f"Query {i}: {len(r.matches)} matches in {r.took_ms:.1f}ms")

        # Cleanup
        await client.delete_collection(collection.id)
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
