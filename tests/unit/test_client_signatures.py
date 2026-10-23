import inspect

import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient

from qdrant_custom_sharding.sharded_client import ShardedQdrantClient

ROUTED = [
    "query_points",
    "query_points_groups",
    "search_matrix_pairs",
    "search_matrix_offsets",
    "count",
    "scroll",
    "facet",
    "query_batch_points",
    "upsert",
    "upload_points",
    "upload_collection",
    "retrieve",
    "update_vectors",
    "delete",
    "set_payload",
    "overwrite_payload",
    "delete_vectors",
    "delete_payload",
    "clear_payload",
    "batch_update_points",
    "delete_collection",
]


def _same(left, right, name: str) -> None:
    assert inspect.signature(getattr(left, name)) == inspect.signature(getattr(right, name))
    assert getattr(left, name).__doc__ == getattr(right, name).__doc__


@pytest.mark.parametrize("name", ROUTED)
def test_sync_signature_matches_qdrant(name):
    _same(ShardedQdrantClient, QdrantClient, name)


@pytest.mark.parametrize("name", ROUTED)
def test_async_signature_matches_qdrant(name):
    from qdrant_custom_sharding.async_sharded_client import AsyncShardedQdrantClient

    _same(AsyncShardedQdrantClient, AsyncQdrantClient, name)
