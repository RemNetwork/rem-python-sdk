"""
REM SDK Client - Sync and Async clients for the REM Vector Database API.

Usage:
    # Sync
    from rem import REM
    client = REM(api_key="rem_xxx")
    collection = client.create_collection("my-docs", dimension=1536)

    # Async
    from rem import AsyncREM
    client = AsyncREM(api_key="rem_xxx")
    collection = await client.create_collection("my-docs", dimension=1536)
"""

import asyncio
from typing import Any, Dict, List, Optional

import httpx

from rem.collection import Collection, AsyncCollection
from rem.exceptions import (
    AuthenticationError,
    NotFoundError,
    QuotaExceededError,
    REMError,
    ServerError,
    ValidationError,
)
from rem.types import CollectionInfo, NamespaceInfo

DEFAULT_BASE_URL = "https://api.getrem.online"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3


def _raise_for_error(response: httpx.Response) -> None:
    """Convert HTTP error responses to SDK exceptions."""
    if response.is_success:
        return

    try:
        body = response.json()
        error = body.get("error", body.get("detail", {}).get("error", {}))
        message = error.get("message", response.text)
        code = error.get("code", "")
    except Exception:
        message = response.text
        code = ""

    status = response.status_code
    if status == 401:
        raise AuthenticationError(message)
    elif status == 404:
        raise NotFoundError(message)
    elif status == 429:
        raise QuotaExceededError(message)
    elif status == 400:
        raise ValidationError(message)
    elif status >= 500:
        raise ServerError(message)
    else:
        raise REMError(message, status_code=status, error_code=code)


# =============================================================================
# ASYNC CLIENT
# =============================================================================


class AsyncREM:
    """
    Async client for the REM Vector Database API.

    Usage:
        client = AsyncREM(api_key="rem_xxx")
        collection = await client.create_collection("my-docs", dimension=1536)
        await collection.upsert([{"id": "doc1", "values": [...]}])
        results = await collection.query(vector=[...], top_k=10)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key or not api_key.startswith("rem_"):
            raise ValueError("API key must start with 'rem_'")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=f"{self._base_url}/v1",
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # -- Collections --

    async def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        replication_factor: int = 3,
        description: Optional[str] = None,
        encrypted_fields: Optional[List[str]] = None,
    ) -> "AsyncCollection":
        """Create a new vector collection.

        Args:
            name: Collection name
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            replication_factor: Number of miner replicas
            description: Optional description
            encrypted_fields: List of metadata field names to encrypt with AES-256-GCM
        """
        payload: Dict[str, Any] = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "replication_factor": replication_factor,
        }
        if description:
            payload["description"] = description
        if encrypted_fields:
            payload["encrypted_fields"] = encrypted_fields

        resp = await self._client.post("/collections", json=payload)
        _raise_for_error(resp)
        info = CollectionInfo(**resp.json())
        return AsyncCollection(self._client, info)

    async def get_collection(self, collection_id: str) -> "AsyncCollection":
        """Get an existing collection by ID."""
        resp = await self._client.get(f"/collections/{collection_id}")
        _raise_for_error(resp)
        info = CollectionInfo(**resp.json())
        return AsyncCollection(self._client, info)

    async def list_collections(self) -> List[CollectionInfo]:
        """List all collections in the namespace."""
        resp = await self._client.get("/collections")
        _raise_for_error(resp)
        data = resp.json()
        return [CollectionInfo(**c) for c in data.get("collections", [])]

    async def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        resp = await self._client.delete(f"/collections/{collection_id}")
        _raise_for_error(resp)
        return resp.json().get("success", False)

    # -- Namespaces --

    async def list_namespaces(self) -> List[NamespaceInfo]:
        """List all namespaces."""
        resp = await self._client.get("/namespaces")
        _raise_for_error(resp)
        data = resp.json()
        return [NamespaceInfo(**ns) for ns in data.get("namespaces", [])]


# =============================================================================
# SYNC CLIENT (wraps AsyncREM)
# =============================================================================


class REM:
    """
    Sync client for the REM Vector Database API.

    Wraps AsyncREM with synchronous methods for convenience.

    Usage:
        client = REM(api_key="rem_xxx")
        collection = client.create_collection("my-docs", dimension=1536)
        collection.upsert([{"id": "doc1", "values": [...]}])
        results = collection.query(vector=[...], top_k=10)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key or not api_key.startswith("rem_"):
            raise ValueError("API key must start with 'rem_'")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=f"{self._base_url}/v1",
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -- Collections --

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        replication_factor: int = 3,
        description: Optional[str] = None,
        encrypted_fields: Optional[List[str]] = None,
    ) -> "Collection":
        """Create a new vector collection.

        Args:
            name: Collection name
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            replication_factor: Number of miner replicas
            description: Optional description
            encrypted_fields: List of metadata field names to encrypt with AES-256-GCM
        """
        payload: Dict[str, Any] = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "replication_factor": replication_factor,
        }
        if description:
            payload["description"] = description
        if encrypted_fields:
            payload["encrypted_fields"] = encrypted_fields

        resp = self._client.post("/collections", json=payload)
        _raise_for_error(resp)
        info = CollectionInfo(**resp.json())
        return Collection(self._client, info)

    def get_collection(self, collection_id: str) -> "Collection":
        """Get an existing collection by ID."""
        resp = self._client.get(f"/collections/{collection_id}")
        _raise_for_error(resp)
        info = CollectionInfo(**resp.json())
        return Collection(self._client, info)

    def list_collections(self) -> List[CollectionInfo]:
        """List all collections in the namespace."""
        resp = self._client.get("/collections")
        _raise_for_error(resp)
        data = resp.json()
        return [CollectionInfo(**c) for c in data.get("collections", [])]

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        resp = self._client.delete(f"/collections/{collection_id}")
        _raise_for_error(resp)
        return resp.json().get("success", False)

    # -- Namespaces --

    def list_namespaces(self) -> List[NamespaceInfo]:
        """List all namespaces."""
        resp = self._client.get("/namespaces")
        _raise_for_error(resp)
        data = resp.json()
        return [NamespaceInfo(**ns) for ns in data.get("namespaces", [])]
