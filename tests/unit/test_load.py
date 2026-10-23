import asyncio
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from qdrant_custom_sharding import LoadSnapshot
from qdrant_custom_sharding.map_cache import MapCache, Reason
from qdrant_custom_sharding.models import FeatureState, Point
from qdrant_custom_sharding.services import (
    add_feature,
    alias_feature,
    create_sharded_collection,
    read_load,
    refresh_stats,
    start_move,
)
from qdrant_custom_sharding.sharded_client import ShardingAdmin
from tests.fakes import FakeMapFolder, FakeMapStore, FakeVectorStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


class _AsyncCounts:

    def __init__(self, inner: FakeVectorStore) -> None:
        self._inner = inner

    async def feature_counts(
        self, collection: str, slot: str, field: str
    ) -> dict[str, int]:
        return self._inner.feature_counts(collection, slot, field)


def _ready(
    n_slots: int = 2,
) -> tuple[FakeMapFolder, FakeMapStore, FakeVectorStore]:
    folder = FakeMapFolder()
    vectors = FakeVectorStore()
    store = folder.store_for("items")
    create_sharded_collection(
        store,
        vectors,
        "items",
        "feature",
        "keyword",
        n_slots=n_slots,
        vectors_config={"size": 2},
    )
    return folder, store, vectors


def _put(
    vectors: FakeVectorStore,
    slot: str,
    point_id: int,
    value: str,
) -> None:
    vectors.upsert(
        "items",
        slot,
        [Point(id=point_id, vector=[1.0, 0.0], payload={"feature": value})],
    )


def _admin(folder: FakeMapFolder, vectors: FakeVectorStore) -> tuple[ShardingAdmin, MapCache]:
    cache = MapCache(folder.store_for("items"), 30, lambda: NOW)
    admin = ShardingAdmin(
        folder,
        lambda _collection: cache,
        lambda: NOW,
        lambda _seconds: None,
        vectors,
    )
    return admin, cache


def test_read_load_returns_slot_counts_and_leaves_the_map():
    folder, store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "books")
    _put(vectors, "s0", 2, "orphan")
    before_version = store.load().version
    before_text = store._text
    snapshot = read_load(store, vectors)
    assert isinstance(snapshot, LoadSnapshot)
    assert snapshot.collection == "items"
    assert snapshot.slots == ("s0", "s1")
    assert "orphan" not in store.load().features
    assert snapshot.counts["s0"] == {"books": 1, "orphan": 1}
    assert snapshot.counts["s1"] == {}
    assert store.load().version == before_version
    assert store._text == before_text
    assert not hasattr(snapshot, "version")
    assert not hasattr(snapshot, "save")
    with pytest.raises(TypeError):
        snapshot.counts["s0"]["orphan"] = 9
    with pytest.raises(TypeError):
        snapshot.counts["s1"]["books"] = 1
    with pytest.raises(TypeError):
        snapshot.counts["extra"] = {}
    with pytest.raises(FrozenInstanceError):
        snapshot.collection = "other"
    plain = snapshot.as_facet_counts()
    plain["s0"]["orphan"] = 9
    assert snapshot.counts["s0"]["orphan"] == 1
    assert plain["s1"] == {}

    admin, cache = _admin(folder, vectors)
    loaded = admin.read_load("items")
    assert loaded.counts["s0"] == {"books": 1, "orphan": 1}
    assert loaded.counts["s1"] == {}
    assert store.load().version == before_version
    store.fail_with_unavailable = True
    cached, reason = cache.get()
    assert cached is None
    assert reason is Reason.no_map
    store.fail_with_unavailable = False
    asyncio.run(_async_read_leaves_the_map(folder, vectors, before_version))


def test_refresh_stats_rolls_aliases_and_writes_no_histogram():
    _folder, store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "books")
    _put(vectors, "s0", 2, "books")
    alias_feature(store, vectors, "books", "libros", lambda _seconds: None, 0)
    _put(vectors, "s0", 3, "libros")
    _put(vectors, "s0", 4, "books")
    _put(vectors, "s0", 5, "stray")
    snapshot = read_load(store, vectors)
    assert snapshot.counts["s0"] == {"libros": 3, "books": 1, "stray": 1}
    assert snapshot.counts["s1"] == {}
    doc = refresh_stats(store, vectors)
    assert doc.features["libros"].points_count == 4
    assert "stray" not in doc.features
    assert doc.slot_load == {"s0": 5, "s1": 0}
    saved_text = store._text
    assert saved_text is not None
    saved = json.loads(saved_text)
    assert "histogram" not in saved
    for row in saved["features"].values():
        assert "histogram" not in row


def test_copying_feature_counts_both_slots():
    _folder, store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "books")
    _put(vectors, "s0", 2, "books")
    start_move(store, "books", ["s1"], NOW, 60)
    _put(vectors, "s1", 3, "books")
    snapshot = read_load(store, vectors)
    assert snapshot.slots == ("s0", "s1")
    assert snapshot.counts["s0"] == {"books": 2}
    assert snapshot.counts["s1"] == {"books": 1}
    doc = refresh_stats(store, vectors)
    assert doc.features["books"].points_count == 3
    assert doc.features["books"].state == FeatureState.copying
    assert doc.slot_load == {"s0": 2, "s1": 1}


async def _async_read_leaves_the_map(
    folder: FakeMapFolder,
    vectors: FakeVectorStore,
    before_version: int,
) -> None:
    from qdrant_custom_sharding.async_sharded_client import AsyncShardingAdmin

    cache = MapCache(folder.store_for("items"), 30, lambda: NOW)
    admin = AsyncShardingAdmin(
        folder,
        lambda _collection: cache,
        lambda: NOW,
        _asleep,
        _AsyncCounts(vectors),
    )
    loaded = await admin.read_load("items")
    assert loaded.counts["s0"] == {"books": 1, "orphan": 1}
    assert loaded.counts["s1"] == {}
    store = folder.store_for("items")
    assert store.load().version == before_version
    store.fail_with_unavailable = True
    cached, reason = cache.get()
    assert cached is None
    assert reason is Reason.no_map


async def _asleep(_seconds: int) -> None:
    return None
