import asyncio
from datetime import datetime, timezone

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_custom_sharding import services
from qdrant_custom_sharding.map_cache import MapCache
from qdrant_custom_sharding.models import FeatureInfo, MapDoc, Point
from qdrant_custom_sharding.qdrant_store import AsyncQdrantVectorStore, QdrantVectorStore
from qdrant_custom_sharding.services import create_sharded_collection
from qdrant_custom_sharding.sharded_client import ShardingAdmin
from tests.fakes import FakeMapFolder, FakeMapStore, FakeVectorStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def _collection(n_slots: int = 2) -> tuple[FakeMapStore, FakeVectorStore]:
    store = FakeMapStore()
    vectors = FakeVectorStore()
    create_sharded_collection(
        store,
        vectors,
        "items",
        "feature",
        "keyword",
        n_slots=n_slots,
        vectors_config={"size": 2},
    )
    return store, vectors


def _doc() -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0", "s1"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 3, "s1": 0},
    )


def test_map_add_slot_appends_without_changing_version():
    doc = _doc()
    doc.add_slot("s9")
    assert doc.slots == ["s0", "s1", "s9"]
    assert doc.slot_load["s9"] == 0
    assert doc.slot_load["s0"] == 3
    assert doc.version == 1
    doc.add_slot("s9")
    assert doc.slots == ["s0", "s1", "s9"]
    assert doc.version == 1


def test_add_slot_appends_name_and_bumps_version_once():
    store, vectors = _collection()
    before = store.load()
    doc = services.add_slot(store, vectors, "s9")
    assert doc.slots == [*before.slots, "s9"]
    assert doc.slot_load["s9"] == 0
    assert doc.slot_load["s0"] == before.slot_load["s0"]
    assert doc.version == before.version + 1
    vectors.upsert(
        "items",
        "s9",
        [Point(id=1, vector=[1.0, 0.0], payload={"feature": "books"})],
    )
    again = services.add_slot(store, vectors, "s9")
    assert again.slots == doc.slots
    assert again.version == doc.version
    assert store.load().slots == doc.slots
    assert store.load().version == doc.version
    assert vectors.read_ids("items", "s9", [1])[0].payload["feature"] == "books"


def test_add_slot_appends_when_the_vector_store_already_has_the_key():
    store, vectors = _collection()
    vectors.create_slot("items", "s9")
    vectors.upsert(
        "items",
        "s9",
        [Point(id=4, vector=[1.0, 0.0], payload={"feature": "books"})],
    )
    before = store.load()
    doc = services.add_slot(store, vectors, "s9")
    assert doc.slots == [*before.slots, "s9"]
    assert doc.slot_load["s9"] == 0
    assert doc.version == before.version + 1
    kept = vectors.read_ids("items", "s9", [4])
    assert len(kept) == 1
    assert kept[0].payload["feature"] == "books"


@pytest.mark.parametrize("name", ["", " ", "\t", "\n"])
def test_blank_slot_raises_and_leaves_the_map(name: str):
    store, vectors = _collection()
    before_slots = list(store.load().slots)
    before_version = store.load().version
    with pytest.raises(ValueError):
        services.add_slot(store, vectors, name)
    assert store.load().slots == before_slots
    assert store.load().version == before_version


def test_blank_slot_raises_before_the_map_is_read():
    with pytest.raises(ValueError):
        services.add_slot(FakeMapStore(), FakeVectorStore(), "  ")


def test_create_slot_keeps_points_already_on_that_slot():
    vectors = FakeVectorStore()
    vectors.create_collection("items")
    vectors.create_slot("items", "s0")
    vectors.upsert(
        "items",
        "s0",
        [Point(id=1, vector=[1.0, 0.0], payload={"feature": "books"})],
    )
    vectors.create_slot("items", "s0")
    found = vectors.read_ids("items", "s0", [1])
    assert [point.id for point in found] == [1]
    assert found[0].payload == {"feature": "books"}


def test_admin_add_slot_returns_the_map_with_the_new_slot():
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
    cache = MapCache(folder.store_for("items"), 30, lambda: NOW)
    cache.put(folder.store_for("items").load())
    admin = ShardingAdmin(
        folder, lambda _name: cache, lambda: NOW, lambda _seconds: None, vectors
    )
    before = folder.store_for("items").load()
    doc = admin.add_slot("items", "s9")
    assert doc.slots == [*before.slots, "s9"]
    assert doc.slot_load["s9"] == 0
    assert doc.version == before.version + 1
    seen, reason = cache.get()
    assert reason is None
    assert seen.slots == doc.slots
    assert seen.version == doc.version
    again = admin.add_slot("items", "s9")
    assert again.slots == doc.slots
    assert again.version == doc.version


def test_async_admin_add_slot_returns_the_map_with_the_new_slot():
    asyncio.run(_async_admin_add_slot())


async def _async_admin_add_slot() -> None:
    from qdrant_custom_sharding.async_sharded_client import AsyncShardingAdmin

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
    cache = MapCache(folder.store_for("items"), 30, lambda: NOW)
    cache.put(folder.store_for("items").load())
    admin = AsyncShardingAdmin(
        folder,
        lambda _name: cache,
        lambda: NOW,
        _asleep,
        _AsyncSlots(vectors),
    )
    before = folder.store_for("items").load()
    doc = await admin.add_slot("items", "s9")
    assert doc.slots == [*before.slots, "s9"]
    assert doc.slot_load["s9"] == 0
    assert doc.version == before.version + 1
    seen, reason = cache.get()
    assert reason is None
    assert seen.slots == doc.slots
    again = await admin.add_slot("items", "s9")
    assert again.slots == doc.slots
    assert again.version == doc.version


class _AsyncSlots:

    def __init__(self, inner: FakeVectorStore) -> None:
        self._inner = inner

    async def create_slot(self, collection: str, slot: str) -> None:
        self._inner.create_slot(collection, slot)


async def _asleep(_seconds: int) -> None:
    return None


class _Shard:

    def __init__(self, shard_key: object) -> None:
        self.shard_key = shard_key


class _Info:

    def __init__(self, local: list[str], remote: list[str]) -> None:
        self.local_shards = [_Shard(key) for key in local]
        self.remote_shards = [_Shard(key) for key in remote]


class _Client:

    def __init__(self, local: list[str] | None = None, remote: list[str] | None = None) -> None:
        self.local = [] if local is None else list(local)
        self.remote = [] if remote is None else list(remote)
        self.created: list[str] = []
        self.create_error: BaseException | None = None
        self.remote_after_error: list[str] | None = None

    def collection_cluster_info(self, collection_name: str) -> _Info:
        return _Info(self.local, self.remote)

    def create_shard_key(self, collection_name: str, shard_key: str) -> bool:
        if self.create_error is not None:
            error = self.create_error
            self.create_error = None
            if self.remote_after_error is not None:
                self.remote = list(self.remote_after_error)
            raise error
        self.created.append(shard_key)
        self.local.append(shard_key)
        return True


class _AsyncClient:

    def __init__(self, inner: _Client) -> None:
        self._inner = inner

    async def collection_cluster_info(self, collection_name: str) -> _Info:
        return self._inner.collection_cluster_info(collection_name)

    async def create_shard_key(self, collection_name: str, shard_key: str) -> bool:
        return self._inner.create_shard_key(collection_name, shard_key)


def test_create_slot_skips_a_key_the_cluster_already_has():
    client = _Client(remote=["s9"])
    QdrantVectorStore(client).create_slot("items", "s9")
    assert client.created == []


def test_create_slot_returns_when_the_key_appears_after_unexpected_response():
    client = _Client()
    client.create_error = UnexpectedResponse(409, "conflict", b"exists", {})
    client.remote_after_error = ["s9"]
    QdrantVectorStore(client).create_slot("items", "s9")
    assert client.created == []


def test_create_slot_reraises_when_the_key_is_still_absent():
    client = _Client()
    client.create_error = UnexpectedResponse(500, "missing", b"missing", {})
    with pytest.raises(UnexpectedResponse):
        QdrantVectorStore(client).create_slot("items", "s9")
    assert client.created == []


def test_create_slot_does_not_swallow_a_timeout():
    client = _Client()
    client.create_error = TimeoutError("timed out")
    with pytest.raises(TimeoutError):
        QdrantVectorStore(client).create_slot("items", "s9")


def test_async_create_slot_skips_a_key_the_cluster_already_has():
    asyncio.run(_async_create_slot_skips())


async def _async_create_slot_skips() -> None:
    client = _Client(local=["s9"])
    await AsyncQdrantVectorStore(_AsyncClient(client)).create_slot("items", "s9")
    assert client.created == []


def test_async_create_slot_returns_when_the_key_appears_after_unexpected_response():
    asyncio.run(_async_create_slot_conflict())


async def _async_create_slot_conflict() -> None:
    client = _Client()
    client.create_error = UnexpectedResponse(409, "conflict", b"exists", {})
    client.remote_after_error = ["s9"]
    await AsyncQdrantVectorStore(_AsyncClient(client)).create_slot("items", "s9")
    assert client.created == []
