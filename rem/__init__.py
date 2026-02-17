"""
REM Vector Database - Python SDK

Official Python client for the REM distributed vector database.

Usage:
    from rem import REM

    client = REM(api_key="rem_xxx", base_url="https://api.getrem.online")
    collection = client.create_collection("my-docs", dimension=1536)
    collection.upsert([{"id": "doc1", "values": [...], "metadata": {"title": "..."}}])
    results = collection.query(vector=[...], top_k=10)

Async Usage:
    from rem import AsyncREM

    client = AsyncREM(api_key="rem_xxx")
    collection = await client.create_collection("my-docs", dimension=1536)
    results = await collection.query(vector=[...], top_k=10)
"""

from rem.client import REM, AsyncREM
from rem.exceptions import (
    REMError,
    AuthenticationError,
    NotFoundError,
    QuotaExceededError,
    REMTimeoutError,
    ValidationError,
    ServerError,
)

__version__ = "0.2.0"
__all__ = [
    "REM",
    "AsyncREM",
    "REMError",
    "AuthenticationError",
    "NotFoundError",
    "QuotaExceededError",
    "REMTimeoutError",
    "ValidationError",
    "ServerError",
]
