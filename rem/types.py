"""
REM SDK Type Definitions

Pydantic models for request/response serialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Vector(BaseModel):
    """A vector with ID, values, and optional metadata."""

    id: str
    values: List[float]
    metadata: Optional[Dict[str, Any]] = None


class ScoredVector(BaseModel):
    """A search result with similarity score."""

    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    values: Optional[List[float]] = None


class CollectionInfo(BaseModel):
    """Collection metadata."""

    id: str
    name: str
    dimension: int
    metric: str
    replication_factor: int
    vector_count: int
    storage_bytes: int
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CollectionStats(CollectionInfo):
    """Extended collection stats with miner info."""

    assigned_miners: int = 0
    synced_miners: int = 0
    is_active: bool = True


class UpsertResult(BaseModel):
    """Result of an upsert operation."""

    upserted_count: int


class QueryResult(BaseModel):
    """Result of a query operation."""

    matches: List[ScoredVector]
    took_ms: Optional[float] = None


class FetchResult(BaseModel):
    """Result of a fetch operation."""

    vectors: List[Vector]


class DeleteResult(BaseModel):
    """Result of a delete operation."""

    deleted_count: int


class NamespaceInfo(BaseModel):
    """Namespace metadata."""

    id: str
    name: str
    display_name: Optional[str] = None
    billing_tier: str
    max_vectors: int
    max_queries_per_month: int
    max_collections: int
    current_vector_count: int
    current_storage_bytes: int
    queries_this_month: int
    created_at: Optional[str] = None


class APIKeyInfo(BaseModel):
    """API key metadata (key value is never returned after creation)."""

    id: str
    name: str
    key_prefix: str
    namespace_id: str
    is_read_only: bool
    rate_limit_rpm: int
    is_active: bool
    total_requests: int
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None


class APIKeyCreated(APIKeyInfo):
    """API key with full key value (only returned at creation time)."""

    key: str
