"""
REM SDK Collection - Vector operations on a collection.

Provides upsert, query, fetch, and delete operations.
Both sync (Collection) and async (AsyncCollection) variants.
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from rem.types import (
    CollectionInfo,
    CollectionStats,
    DeleteResult,
    FetchResult,
    QueryResult,
    ScoredVector,
    UpsertResult,
    Vector,
)


def _raise_for_error(response: httpx.Response) -> None:
    """Import and call the shared error handler."""
    from rem.client import _raise_for_error as _raise
    _raise(response)


# =============================================================================
# ASYNC COLLECTION
# =============================================================================


class AsyncCollection:
    """
    Async interface for vector operations on a collection.

    Usage:
        collection = await client.create_collection("my-docs", dimension=1536)
        await collection.upsert([{"id": "doc1", "values": [...]}])
        results = await collection.query(vector=[...], top_k=10)
    """

    def __init__(self, client: httpx.AsyncClient, info: CollectionInfo):
        self._client = client
        self._info = info

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def dimension(self) -> int:
        return self._info.dimension

    @property
    def metric(self) -> str:
        return self._info.metric

    @property
    def vector_count(self) -> int:
        return self._info.vector_count

    async def upsert(
        self,
        vectors: List[Union[Dict[str, Any], Vector]],
    ) -> UpsertResult:
        """
        Insert or update vectors.

        Args:
            vectors: List of vectors. Each can be a dict with keys
                     {id, values, metadata} or a Vector object.

        Returns:
            UpsertResult with upserted_count
        """
        vec_dicts = []
        for v in vectors:
            if isinstance(v, Vector):
                vec_dicts.append(v.model_dump())
            elif isinstance(v, dict):
                vec_dicts.append(v)
            else:
                raise TypeError(f"Expected dict or Vector, got {type(v)}")

        resp = await self._client.post(
            f"/collections/{self.id}/vectors/upsert",
            json={"vectors": vec_dicts},
        )
        _raise_for_error(resp)
        return UpsertResult(**resp.json())

    async def query(
        self,
        vector: Optional[List[float]] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
        query_text: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> QueryResult:
        """
        Search for similar vectors.

        Args:
            vector: Query vector (optional if query_text provided for pure keyword search)
            top_k: Number of results (1-1000)
            filter: Optional metadata filter
            include_metadata: Include metadata in results
            include_values: Include vector values in results
            query_text: Optional keyword search text (BM25)
            hybrid_alpha: Hybrid search weight (0.0=pure vector, 1.0=pure keyword, 0.5=balanced)

        Returns:
            QueryResult with matches and latency
        """
        payload: Dict[str, Any] = {
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_values": include_values,
        }
        if vector is not None:
            payload["vector"] = vector
        if filter:
            payload["filter"] = filter
        if query_text:
            payload["query_text"] = query_text
        if hybrid_alpha is not None:
            payload["hybrid_alpha"] = hybrid_alpha

        resp = await self._client.post(
            f"/collections/{self.id}/vectors/query",
            json=payload,
        )
        _raise_for_error(resp)
        data = resp.json()
        return QueryResult(
            matches=[ScoredVector(**m) for m in data.get("matches", [])],
            took_ms=data.get("took_ms"),
        )

    async def query_batch(
        self,
        queries: List[Dict[str, Any]],
    ) -> List[QueryResult]:
        """
        Execute multiple queries in a single API call.

        Args:
            queries: List of query dicts, each with keys matching query() params
                     (vector, top_k, filter, include_metadata, query_text, hybrid_alpha)
                     Max 10 queries per batch.

        Returns:
            List of QueryResult, one per query
        """
        resp = await self._client.post(
            f"/collections/{self.id}/vectors/query/batch",
            json={"queries": queries},
        )
        _raise_for_error(resp)
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append(QueryResult(
                matches=[ScoredVector(**m) for m in r.get("matches", [])],
                took_ms=r.get("took_ms"),
            ))
        return results

    async def fetch(self, ids: List[str]) -> FetchResult:
        """
        Fetch vectors by their IDs.

        Args:
            ids: List of vector IDs to fetch

        Returns:
            FetchResult with vectors
        """
        resp = await self._client.post(
            f"/collections/{self.id}/vectors/fetch",
            json={"ids": ids},
        )
        _raise_for_error(resp)
        data = resp.json()
        return FetchResult(
            vectors=[Vector(**v) for v in data.get("vectors", [])]
        )

    async def delete(self, ids: List[str]) -> DeleteResult:
        """
        Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete

        Returns:
            DeleteResult with deleted_count
        """
        resp = await self._client.post(
            f"/collections/{self.id}/vectors/delete",
            json={"ids": ids},
        )
        _raise_for_error(resp)
        return DeleteResult(**resp.json())

    async def stats(self) -> CollectionStats:
        """Get collection statistics."""
        resp = await self._client.get(f"/collections/{self.id}/stats")
        _raise_for_error(resp)
        return CollectionStats(**resp.json())

    async def refresh(self) -> None:
        """Refresh collection info from server."""
        resp = await self._client.get(f"/collections/{self.id}")
        _raise_for_error(resp)
        self._info = CollectionInfo(**resp.json())

    def __repr__(self) -> str:
        return (
            f"AsyncCollection(name='{self.name}', dim={self.dimension}, "
            f"metric='{self.metric}', vectors={self.vector_count})"
        )


# =============================================================================
# SYNC COLLECTION
# =============================================================================


class Collection:
    """
    Sync interface for vector operations on a collection.

    Usage:
        collection = client.create_collection("my-docs", dimension=1536)
        collection.upsert([{"id": "doc1", "values": [...]}])
        results = collection.query(vector=[...], top_k=10)
    """

    def __init__(self, client: httpx.Client, info: CollectionInfo):
        self._client = client
        self._info = info

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def dimension(self) -> int:
        return self._info.dimension

    @property
    def metric(self) -> str:
        return self._info.metric

    @property
    def vector_count(self) -> int:
        return self._info.vector_count

    def upsert(
        self,
        vectors: List[Union[Dict[str, Any], Vector]],
    ) -> UpsertResult:
        """Insert or update vectors."""
        vec_dicts = []
        for v in vectors:
            if isinstance(v, Vector):
                vec_dicts.append(v.model_dump())
            elif isinstance(v, dict):
                vec_dicts.append(v)
            else:
                raise TypeError(f"Expected dict or Vector, got {type(v)}")

        resp = self._client.post(
            f"/collections/{self.id}/vectors/upsert",
            json={"vectors": vec_dicts},
        )
        _raise_for_error(resp)
        return UpsertResult(**resp.json())

    def query(
        self,
        vector: Optional[List[float]] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
        query_text: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> QueryResult:
        """
        Search for similar vectors.

        Args:
            vector: Query vector (optional if query_text provided for pure keyword search)
            top_k: Number of results (1-1000)
            filter: Optional metadata filter
            include_metadata: Include metadata in results
            include_values: Include vector values in results
            query_text: Optional keyword search text (BM25)
            hybrid_alpha: Hybrid search weight (0.0=pure vector, 1.0=pure keyword, 0.5=balanced)

        Returns:
            QueryResult with matches and latency
        """
        payload: Dict[str, Any] = {
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_values": include_values,
        }
        if vector is not None:
            payload["vector"] = vector
        if filter:
            payload["filter"] = filter
        if query_text:
            payload["query_text"] = query_text
        if hybrid_alpha is not None:
            payload["hybrid_alpha"] = hybrid_alpha

        resp = self._client.post(
            f"/collections/{self.id}/vectors/query",
            json=payload,
        )
        _raise_for_error(resp)
        data = resp.json()
        return QueryResult(
            matches=[ScoredVector(**m) for m in data.get("matches", [])],
            took_ms=data.get("took_ms"),
        )

    def query_batch(
        self,
        queries: List[Dict[str, Any]],
    ) -> List[QueryResult]:
        """
        Execute multiple queries in a single API call.

        Args:
            queries: List of query dicts, each with keys matching query() params
                     Max 10 queries per batch.

        Returns:
            List of QueryResult, one per query
        """
        resp = self._client.post(
            f"/collections/{self.id}/vectors/query/batch",
            json={"queries": queries},
        )
        _raise_for_error(resp)
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append(QueryResult(
                matches=[ScoredVector(**m) for m in r.get("matches", [])],
                took_ms=r.get("took_ms"),
            ))
        return results

    def fetch(self, ids: List[str]) -> FetchResult:
        """Fetch vectors by their IDs."""
        resp = self._client.post(
            f"/collections/{self.id}/vectors/fetch",
            json={"ids": ids},
        )
        _raise_for_error(resp)
        data = resp.json()
        return FetchResult(
            vectors=[Vector(**v) for v in data.get("vectors", [])]
        )

    def delete(self, ids: List[str]) -> DeleteResult:
        """Delete vectors by their IDs."""
        resp = self._client.post(
            f"/collections/{self.id}/vectors/delete",
            json={"ids": ids},
        )
        _raise_for_error(resp)
        return DeleteResult(**resp.json())

    def stats(self) -> CollectionStats:
        """Get collection statistics."""
        resp = self._client.get(f"/collections/{self.id}/stats")
        _raise_for_error(resp)
        return CollectionStats(**resp.json())

    def refresh(self) -> None:
        """Refresh collection info from server."""
        resp = self._client.get(f"/collections/{self.id}")
        _raise_for_error(resp)
        self._info = CollectionInfo(**resp.json())

    def __repr__(self) -> str:
        return (
            f"Collection(name='{self.name}', dim={self.dimension}, "
            f"metric='{self.metric}', vectors={self.vector_count})"
        )
