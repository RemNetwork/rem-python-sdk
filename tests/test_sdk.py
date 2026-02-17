"""
Tests for the REM SDK.

Uses httpx mock transport to test without hitting the real API.
"""

import pytest
import json
import httpx

from rem import REM, AsyncREM
from rem.client import _raise_for_error
from rem.types import (
    CollectionInfo,
    Vector,
    ScoredVector,
    QueryResult,
    UpsertResult,
    FetchResult,
    DeleteResult,
)
from rem.exceptions import (
    REMError,
    AuthenticationError,
    NotFoundError,
    QuotaExceededError,
    ValidationError,
    ServerError,
)


# =============================================================================
# FIXTURES
# =============================================================================

MOCK_COLLECTION = {
    "id": "col_test123",
    "name": "test-collection",
    "dimension": 384,
    "metric": "cosine",
    "replication_factor": 3,
    "vector_count": 100,
    "storage_bytes": 153600,
    "description": "Test collection",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}

MOCK_VECTOR = {"id": "vec_1", "values": [0.1, 0.2, 0.3], "metadata": {"title": "test"}}


def mock_transport(responses: dict):
    """Create a mock transport that returns predefined responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        key = f"{method} {path}"

        # Try exact match first
        if key in responses:
            resp = responses[key]
            return httpx.Response(
                status_code=resp.get("status", 200),
                json=resp.get("json", {}),
            )

        # Try partial match
        for pattern, resp in responses.items():
            p_method, p_path = pattern.split(" ", 1)
            if p_method == method and path.startswith(p_path.rstrip("*")):
                return httpx.Response(
                    status_code=resp.get("status", 200),
                    json=resp.get("json", {}),
                )

        return httpx.Response(status_code=404, json={"error": {"message": "Not found"}})

    return httpx.MockTransport(handler)


# =============================================================================
# CLIENT TESTS
# =============================================================================


class TestClientInit:
    def test_valid_api_key(self):
        client = REM(api_key="rem_test123")
        assert client._api_key == "rem_test123"
        client.close()

    def test_invalid_api_key_no_prefix(self):
        with pytest.raises(ValueError, match="must start with 'rem_'"):
            REM(api_key="invalid_key")

    def test_empty_api_key(self):
        with pytest.raises(ValueError):
            REM(api_key="")

    def test_custom_base_url(self):
        client = REM(api_key="rem_test", base_url="https://custom.api.com")
        assert client._base_url == "https://custom.api.com"
        client.close()

    def test_trailing_slash_stripped(self):
        client = REM(api_key="rem_test", base_url="https://api.example.com/")
        assert client._base_url == "https://api.example.com"
        client.close()

    def test_context_manager(self):
        with REM(api_key="rem_test") as client:
            assert client._api_key == "rem_test"


class TestAsyncClientInit:
    def test_valid_api_key(self):
        client = AsyncREM(api_key="rem_test123")
        assert client._api_key == "rem_test123"

    def test_invalid_api_key(self):
        with pytest.raises(ValueError, match="must start with 'rem_'"):
            AsyncREM(api_key="bad_key")


# =============================================================================
# COLLECTION OPERATIONS
# =============================================================================


class TestCreateCollection:
    def test_create_collection(self):
        transport = mock_transport({
            "POST /v1/collections": {"json": MOCK_COLLECTION},
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        resp = client.post("/collections", json={"name": "test", "dimension": 384})
        info = CollectionInfo(**resp.json())
        collection = Collection(client, info)

        assert collection.name == "test-collection"
        assert collection.dimension == 384
        assert collection.metric == "cosine"
        assert collection.id == "col_test123"
        client.close()


class TestUpsert:
    def test_upsert_dicts(self):
        transport = mock_transport({
            "POST /v1/collections": {"json": MOCK_COLLECTION},
            "POST /v1/collections/col_test123/vectors/upsert": {
                "json": {"upserted_count": 2},
            },
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        result = collection.upsert([
            {"id": "doc1", "values": [0.1, 0.2, 0.3]},
            {"id": "doc2", "values": [0.4, 0.5, 0.6]},
        ])
        assert result.upserted_count == 2
        client.close()

    def test_upsert_vector_objects(self):
        transport = mock_transport({
            "POST /v1/collections/col_test123/vectors/upsert": {
                "json": {"upserted_count": 1},
            },
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        vec = Vector(id="doc1", values=[0.1, 0.2, 0.3], metadata={"title": "test"})
        result = collection.upsert([vec])
        assert result.upserted_count == 1
        client.close()

    def test_upsert_invalid_type(self):
        transport = mock_transport({})
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        with pytest.raises(TypeError, match="Expected dict or Vector"):
            collection.upsert(["not_a_vector"])
        client.close()


class TestQuery:
    def test_basic_query(self):
        transport = mock_transport({
            "POST /v1/collections/col_test123/vectors/query": {
                "json": {
                    "matches": [
                        {"id": "doc1", "score": 0.95, "metadata": {"title": "Hello"}},
                        {"id": "doc2", "score": 0.87, "metadata": {"title": "World"}},
                    ],
                    "took_ms": 12.5,
                },
            },
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        result = collection.query(vector=[0.1, 0.2, 0.3], top_k=5)
        assert len(result.matches) == 2
        assert result.matches[0].id == "doc1"
        assert result.matches[0].score == 0.95
        assert result.took_ms == 12.5
        client.close()

    def test_query_with_filter(self):
        transport = mock_transport({
            "POST /v1/collections/col_test123/vectors/query": {
                "json": {"matches": [], "took_ms": 5.0},
            },
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        result = collection.query(
            vector=[0.1, 0.2, 0.3],
            top_k=10,
            filter={"category": {"$eq": "tech"}},
        )
        assert len(result.matches) == 0
        client.close()

    def test_hybrid_query(self):
        transport = mock_transport({
            "POST /v1/collections/col_test123/vectors/query": {
                "json": {
                    "matches": [{"id": "doc1", "score": 0.92}],
                    "took_ms": 15.0,
                },
            },
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        result = collection.query(
            vector=[0.1, 0.2, 0.3],
            query_text="machine learning",
            hybrid_alpha=0.7,
            top_k=5,
        )
        assert len(result.matches) == 1
        client.close()


class TestBatchQuery:
    def test_batch_query(self):
        transport = mock_transport({
            "POST /v1/collections/col_test123/vectors/query/batch": {
                "json": {
                    "results": [
                        {"matches": [{"id": "doc1", "score": 0.9}], "took_ms": 10},
                        {"matches": [{"id": "doc2", "score": 0.8}], "took_ms": 12},
                    ],
                },
            },
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        results = collection.query_batch([
            {"vector": [0.1, 0.2, 0.3], "top_k": 5},
            {"vector": [0.4, 0.5, 0.6], "top_k": 3},
        ])
        assert len(results) == 2
        assert results[0].matches[0].id == "doc1"
        assert results[1].matches[0].id == "doc2"
        client.close()


class TestFetch:
    def test_fetch_vectors(self):
        transport = mock_transport({
            "POST /v1/collections/col_test123/vectors/fetch": {
                "json": {
                    "vectors": [
                        {"id": "doc1", "values": [0.1, 0.2, 0.3], "metadata": {"title": "test"}},
                    ],
                },
            },
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        result = collection.fetch(ids=["doc1"])
        assert len(result.vectors) == 1
        assert result.vectors[0].id == "doc1"
        assert result.vectors[0].values == [0.1, 0.2, 0.3]
        client.close()


class TestDelete:
    def test_delete_vectors(self):
        transport = mock_transport({
            "POST /v1/collections/col_test123/vectors/delete": {
                "json": {"deleted_count": 2},
            },
        })
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        from rem.collection import Collection

        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        result = collection.delete(ids=["doc1", "doc2"])
        assert result.deleted_count == 2
        client.close()


# =============================================================================
# ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    def test_401_raises_auth_error(self):
        resp = httpx.Response(
            401, json={"error": {"message": "Invalid API key", "code": "UNAUTHORIZED"}}
        )
        with pytest.raises(AuthenticationError):
            _raise_for_error(resp)

    def test_404_raises_not_found(self):
        resp = httpx.Response(
            404, json={"error": {"message": "Not found", "code": "NOT_FOUND"}}
        )
        with pytest.raises(NotFoundError):
            _raise_for_error(resp)

    def test_429_raises_quota_exceeded(self):
        resp = httpx.Response(
            429, json={"error": {"message": "Rate limit", "code": "QUOTA_EXCEEDED"}}
        )
        with pytest.raises(QuotaExceededError):
            _raise_for_error(resp)

    def test_400_raises_validation_error(self):
        resp = httpx.Response(
            400, json={"error": {"message": "Bad request", "code": "VALIDATION_ERROR"}}
        )
        with pytest.raises(ValidationError):
            _raise_for_error(resp)

    def test_500_raises_server_error(self):
        resp = httpx.Response(
            500, json={"error": {"message": "Internal error", "code": "SERVER_ERROR"}}
        )
        with pytest.raises(ServerError):
            _raise_for_error(resp)

    def test_success_no_error(self):
        resp = httpx.Response(200, json={"ok": True})
        _raise_for_error(resp)  # Should not raise


# =============================================================================
# TYPE TESTS
# =============================================================================


class TestTypes:
    def test_vector_model(self):
        v = Vector(id="doc1", values=[0.1, 0.2, 0.3], metadata={"key": "val"})
        assert v.id == "doc1"
        assert len(v.values) == 3
        assert v.metadata == {"key": "val"}

    def test_vector_no_metadata(self):
        v = Vector(id="doc1", values=[0.1])
        assert v.metadata is None

    def test_scored_vector(self):
        sv = ScoredVector(id="doc1", score=0.95, metadata={"x": 1})
        assert sv.score == 0.95
        assert sv.values is None

    def test_collection_info(self):
        info = CollectionInfo(**MOCK_COLLECTION)
        assert info.name == "test-collection"
        assert info.dimension == 384

    def test_query_result(self):
        qr = QueryResult(matches=[], took_ms=5.0)
        assert len(qr.matches) == 0
        assert qr.took_ms == 5.0

    def test_upsert_result(self):
        ur = UpsertResult(upserted_count=10)
        assert ur.upserted_count == 10

    def test_fetch_result(self):
        fr = FetchResult(vectors=[])
        assert len(fr.vectors) == 0

    def test_delete_result(self):
        dr = DeleteResult(deleted_count=3)
        assert dr.deleted_count == 3


# =============================================================================
# COLLECTION REPR
# =============================================================================


class TestRepr:
    def test_collection_repr(self):
        from rem.collection import Collection

        transport = mock_transport({})
        client = httpx.Client(base_url="https://api.getrem.online/v1", transport=transport)
        info = CollectionInfo(**MOCK_COLLECTION)
        collection = Collection(client, info)

        repr_str = repr(collection)
        assert "test-collection" in repr_str
        assert "384" in repr_str
        assert "cosine" in repr_str
        client.close()
