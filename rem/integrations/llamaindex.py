"""
REM Vector Database — LlamaIndex Integration

Drop-in LlamaIndex VectorStore backed by the REM decentralized network.

Usage:
    from llama_index.core import VectorStoreIndex
    from rem.integrations.llamaindex import REMVectorStore

    vector_store = REMVectorStore(
        api_key="rem_xxx",
        collection_name="docs",
    )
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is REM?")

Install:
    pip install rem-vectordb[llamaindex]
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from rem import REM
from rem.types import ScoredVector


class REMVectorStore(BasePydanticVectorStore):
    """LlamaIndex VectorStore backed by the REM decentralized vector database."""

    stores_text: bool = True
    flat_metadata: bool = True

    # Pydantic fields
    api_key: str
    collection_name: str
    base_url: str = "https://api.getrem.online"
    dimension: Optional[int] = None
    metric: str = "cosine"

    # Private (not serialized)
    _client: Any = None
    _collection: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = REM(api_key=self.api_key, base_url=self.base_url)
        self._collection = None

    def _get_collection(self):
        """Get or create the collection (lazy init)."""
        if self._collection is not None:
            return self._collection

        # Try to find existing collection
        collections = self._client.list_collections()
        for c in collections:
            if c.name == self.collection_name:
                self._collection = self._client.get_collection(c.id)
                return self._collection

        # Need dimension to create — will be set on first add()
        if self.dimension is None:
            return None

        self._collection = self._client.create_collection(
            name=self.collection_name,
            dimension=self.dimension,
            metric=self.metric,
        )
        return self._collection

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """
        Add nodes to the vector store.

        Args:
            nodes: List of LlamaIndex nodes with embeddings

        Returns:
            List of node IDs
        """
        if not nodes:
            return []

        # Auto-detect dimension from first node's embedding
        if self.dimension is None and nodes[0].embedding:
            self.dimension = len(nodes[0].embedding)

        collection = self._get_collection()
        if collection is None:
            raise ValueError(
                "Cannot create collection without dimension. "
                "Ensure nodes have embeddings or set dimension explicitly."
            )

        vectors = []
        ids = []
        for node in nodes:
            node_id = node.node_id or str(uuid.uuid4())
            ids.append(node_id)

            metadata: Dict[str, Any] = {}
            # Store text content
            if hasattr(node, "get_content"):
                metadata["text"] = node.get_content()
            # Store flat metadata
            if hasattr(node, "metadata") and node.metadata:
                for k, v in node.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
            # Store ref_doc_id for delete-by-document
            if hasattr(node, "ref_doc_id") and node.ref_doc_id:
                metadata["ref_doc_id"] = node.ref_doc_id

            vectors.append({
                "id": node_id,
                "values": node.embedding,
                "metadata": metadata,
            })

        # Batch upsert in chunks of 1000
        for start in range(0, len(vectors), 1000):
            batch = vectors[start : start + 1000]
            collection.upsert(batch)

        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """
        Delete nodes by reference document ID.

        Args:
            ref_doc_id: The reference document ID to delete
        """
        collection = self._get_collection()
        if collection is None:
            return

        # Query for all nodes with this ref_doc_id
        result = collection.query(
            query_text=ref_doc_id,
            top_k=1000,
            filter={"ref_doc_id": {"$eq": ref_doc_id}},
            include_metadata=True,
        )

        if result.matches:
            ids_to_delete = [m.id for m in result.matches]
            collection.delete(ids_to_delete)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the vector store.

        Args:
            query: LlamaIndex VectorStoreQuery

        Returns:
            VectorStoreQueryResult with nodes, similarities, and IDs
        """
        collection = self._get_collection()
        if collection is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        top_k = query.similarity_top_k or 10

        # Build filter from query filters
        rem_filter = None
        if query.filters and query.filters.filters:
            conditions = []
            for f in query.filters.filters:
                conditions.append({f.key: {"$eq": f.value}})
            if len(conditions) == 1:
                rem_filter = conditions[0]
            else:
                rem_filter = {"$and": conditions}

        result = collection.query(
            vector=query.query_embedding,
            top_k=top_k,
            filter=rem_filter,
            include_metadata=True,
        )

        nodes = []
        similarities = []
        ids = []

        for match in result.matches:
            metadata = match.metadata or {}
            text = metadata.pop("text", "")
            ref_doc_id = metadata.pop("ref_doc_id", None)

            node = TextNode(
                id_=match.id,
                text=text,
                metadata=metadata,
            )
            if ref_doc_id:
                node.ref_doc_id = ref_doc_id

            nodes.append(node)
            similarities.append(match.score)
            ids.append(match.id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )
