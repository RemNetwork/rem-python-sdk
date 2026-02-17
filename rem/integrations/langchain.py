"""
REM Vector Database — LangChain Integration

Drop-in LangChain VectorStore backed by the REM decentralized network.

Usage:
    from langchain_openai import OpenAIEmbeddings
    from rem.integrations.langchain import REMVectorStore

    store = REMVectorStore.from_texts(
        texts=["Hello world", "REM is great"],
        embedding=OpenAIEmbeddings(),
        api_key="rem_xxx",
        collection_name="docs",
    )
    results = store.similarity_search("greeting", k=2)

Install:
    pip install rem-vectordb[langchain]
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from rem import REM


class REMVectorStore(VectorStore):
    """LangChain VectorStore backed by the REM decentralized vector database."""

    def __init__(
        self,
        api_key: str,
        collection_name: str,
        embedding: Embeddings,
        base_url: str = "https://api.getrem.online",
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **kwargs: Any,
    ):
        """
        Initialize REMVectorStore.

        Args:
            api_key: REM API key (starts with 'rem_')
            collection_name: Name of the collection to use
            embedding: LangChain Embeddings instance
            base_url: REM API base URL
            dimension: Vector dimension (auto-detected from embedding if not set)
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        self._client = REM(api_key=api_key, base_url=base_url)
        self._embedding = embedding
        self._collection_name = collection_name
        self._dimension = dimension
        self._metric = metric
        self._collection = None

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _get_collection(self):
        """Get or create the collection (lazy init)."""
        if self._collection is not None:
            return self._collection

        # Try to find existing collection
        collections = self._client.list_collections()
        for c in collections:
            if c.name == self._collection_name:
                self._collection = self._client.get_collection(c.id)
                return self._collection

        # Auto-detect dimension from embedding
        if self._dimension is None:
            sample = self._embedding.embed_query("dimension probe")
            self._dimension = len(sample)

        # Create new collection
        self._collection = self._client.create_collection(
            name=self._collection_name,
            dimension=self._dimension,
            metric=self._metric,
        )
        return self._collection

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Embed texts and add them to the vector store.

        Args:
            texts: Texts to embed and store
            metadatas: Optional metadata dicts per text
            ids: Optional IDs (auto-generated if not provided)

        Returns:
            List of IDs for the added texts
        """
        texts_list = list(texts)
        embeddings = self._embedding.embed_documents(texts_list)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        vectors = []
        for i, (text, emb) in enumerate(zip(texts_list, embeddings)):
            meta = metadatas[i] if metadatas else {}
            meta = dict(meta)  # copy
            meta["text"] = text  # store original text for retrieval
            vectors.append({
                "id": ids[i],
                "values": emb,
                "metadata": meta,
            })

        collection = self._get_collection()
        # Batch in chunks of 1000
        for start in range(0, len(vectors), 1000):
            batch = vectors[start : start + 1000]
            collection.upsert(batch)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for similar documents by text query.

        Args:
            query: Query text
            k: Number of results
            filter: Optional metadata filter

        Returns:
            List of LangChain Documents
        """
        results = self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents and return scores.

        Args:
            query: Query text
            k: Number of results
            filter: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        query_embedding = self._embedding.embed_query(query)
        collection = self._get_collection()

        result = collection.query(
            vector=query_embedding,
            top_k=k,
            filter=filter,
            include_metadata=True,
        )

        docs_with_scores = []
        for match in result.matches:
            metadata = match.metadata or {}
            text = metadata.pop("text", "")
            doc = Document(page_content=text, metadata=metadata)
            docs_with_scores.append((doc, match.score))

        return docs_with_scores

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search by raw embedding vector."""
        collection = self._get_collection()
        result = collection.query(
            vector=embedding,
            top_k=k,
            filter=filter,
            include_metadata=True,
        )

        docs = []
        for match in result.matches:
            metadata = match.metadata or {}
            text = metadata.pop("text", "")
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete vectors by IDs."""
        if not ids:
            return False
        collection = self._get_collection()
        result = collection.delete(ids)
        return result.deleted_count > 0

    @classmethod
    def from_texts(
        cls: Type["REMVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        api_key: str = "",
        collection_name: str = "langchain",
        base_url: str = "https://api.getrem.online",
        **kwargs: Any,
    ) -> "REMVectorStore":
        """
        Create a REMVectorStore from texts.

        Args:
            texts: Texts to embed and store
            embedding: LangChain Embeddings instance
            metadatas: Optional metadata dicts
            ids: Optional IDs
            api_key: REM API key
            collection_name: Collection name
            base_url: REM API base URL

        Returns:
            REMVectorStore instance with texts added
        """
        store = cls(
            api_key=api_key,
            collection_name=collection_name,
            embedding=embedding,
            base_url=base_url,
            **kwargs,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
