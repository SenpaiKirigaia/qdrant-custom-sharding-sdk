import asyncio
from datetime import datetime, timezone

from qdrant_custom_sharding.map_cache import MapCache
from qdrant_custom_sharding.models import Point
from qdrant_custom_sharding.services import add_feature, create_sharded_collection
from qdrant_custom_sharding.sharded_client import ShardingAdmin
from tests.fakes import FakeMapFolder, FakeVectorStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


class _AsyncVectors:

    def __init__(self, inner: FakeVectorStore) -> None:
        self._inner = inner

    async def rewrite_feature(
        self,
        collection: str,
        slots: list[str],
        field: str,
        old_values: list[int | str],
        new_value: int | str,
    ) -> int:
        return self._inner.rewrite_feature(collection, slots, field, old_values, new_value)


def _cache_for(folder: FakeMapFolder):
    caches: dict[str, MapCache] = {}

    def cache_for(collection: str) -> MapCache:
        cache = caches.get(collection)
        if cache is None:
            cache = MapCache(folder.store_for(collection), 30, lambda: NOW)
            caches[collection] = cache
        return cache

    return cache_for


def _ready() -> tuple[FakeMapFolder, FakeVectorStore, str]:
    folder = FakeMapFolder()
    vectors = FakeVectorStore()
    create_sharded_collection(
        folder.store_for("items"),
        vectors,
        "items",
        "feature",
        "keyword",
        n_slots=2,
        vectors_config={"size": 2},
    )
    add_feature(folder.store_for("items"), "books")
    slot = folder.store_for("items").load().features["books"].shard_keys[0]
    vectors.upsert(
        "items",
        slot,
        [Point(id=1, vector=[1.0, 0.0], payload={"feature": "books", "color": "red"})],
    )
    return folder, vectors, slot


def test_sync_admin_alias_sleeps_then_returns_the_rewrite_count():
    folder, vectors, slot = _ready()
    slept: list[int] = []

    def sleep(seconds: int) -> None:
        slept.append(seconds)
        stored = vectors.read_ids("items", slot, [1])[0]
        assert stored.payload["feature"] == "books"

    admin = ShardingAdmin(folder, _cache_for(folder), lambda: NOW, sleep, vectors)
    changed = admin.alias_feature("items", "books", "libros", grace_seconds=4)
    assert slept == [4]
    assert changed == 1
    stored = vectors.read_ids("items", slot, [1])[0]
    assert stored.payload == {"feature": "libros", "color": "red"}
    assert admin.get_map("items").features["libros"].aliases == ["books"]
    vectors.upsert(
        "items",
        slot,
        [Point(id=2, vector=[1.0, 0.0], payload={"feature": "books"})],
    )
    dropped = admin.drop_alias("items", "books")
    assert dropped == 1
    assert vectors.read_ids("items", slot, [2])[0].payload["feature"] == "libros"
    assert admin.get_map("items").features["libros"].aliases == []


def test_admin_add_feature_shows_up_on_the_next_get_map():
    folder, vectors, _slot = _ready()
    admin = ShardingAdmin(folder, _cache_for(folder), lambda: NOW, lambda _seconds: None, vectors)
    info = admin.add_feature("items", "toys")
    again = admin.add_feature("items", "toys")
    assert again.shard_keys == info.shard_keys
    mapped = admin.get_map("items")
    assert mapped.features["toys"].shard_keys == info.shard_keys


def test_admin_refresh_stats_reads_the_fake_store():
    folder, vectors, slot = _ready()
    admin = ShardingAdmin(folder, _cache_for(folder), lambda: NOW, lambda _seconds: None, vectors)
    doc = admin.refresh_stats("items")
    assert doc.features["books"].points_count == 1
    assert doc.slot_load[slot] == 1
    assert sum(doc.slot_load.values()) == 1


def test_async_admin_alias_sleeps_then_returns_the_rewrite_count():
    asyncio.run(_async_alias())


async def _async_alias() -> None:
    from qdrant_custom_sharding.async_sharded_client import AsyncShardingAdmin

    folder, vectors, slot = _ready()
    slept: list[int] = []

    async def sleep(seconds: int) -> None:
        slept.append(seconds)
        stored = vectors.read_ids("items", slot, [1])[0]
        assert stored.payload["feature"] == "books"

    admin = AsyncShardingAdmin(
        folder,
        _cache_for(folder),
        lambda: NOW,
        sleep,
        _AsyncVectors(vectors),
    )
    changed = await admin.alias_feature("items", "books", "libros", grace_seconds=4)
    assert slept == [4]
    assert changed == 1
    assert vectors.read_ids("items", slot, [1])[0].payload["feature"] == "libros"
    mapped = await admin.get_map("items")
    assert mapped.features["libros"].aliases == ["books"]
    vectors.upsert(
        "items",
        slot,
        [Point(id=2, vector=[1.0, 0.0], payload={"feature": "books"})],
    )
    dropped = await admin.drop_alias("items", "books")
    assert dropped == 1
    mapped = await admin.get_map("items")
    assert mapped.features["libros"].aliases == []
