import socket
import uuid
import asyncio

import pytest
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from qdrant_custom_sharding import AsyncShardedQdrantClient, ShardedQdrantClient, StopReason
from qdrant_custom_sharding.map_stores import FileMapFolder
from qdrant_custom_sharding.models import FeatureState
from qdrant_custom_sharding.qdrant_store import insert_only_supported
from tests.local_qdrant import local_kwargs

pytestmark = pytest.mark.integration

needs_insert_only = pytest.mark.skipif(
    not insert_only_supported(),
    reason="moves need qdrant-client and server >= 1.17 for insert-only upserts",
)
VECTOR = [0.1, 0.2, 0.3, 0.4]


def _qdrant_up() -> bool:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", 6333))
    except OSError:
        return False
    finally:
        sock.close()
    return True


@pytest.fixture(scope="module", autouse=True)
def qdrant_up():
    if not _qdrant_up():
        pytest.skip("Qdrant is not running on localhost:6333")


@pytest.fixture
def client(tmp_path):
    opened = ShardedQdrantClient(map_folder=FileMapFolder(tmp_path), **local_kwargs())
    yield opened
    opened.close()


def _point(point_id: int, feature: str) -> PointStruct:
    return PointStruct(id=point_id, vector=VECTOR, payload={"feature": feature})


def _ids(client, name: str, feature: str) -> set[int]:
    found = client.query_points(
        name,
        VECTOR,
        query_filter=Filter(
            must=[FieldCondition(key="feature", match=MatchValue(value=feature))]
        ),
        limit=100,
    )
    return {point.id for point in found.points}


def _count(client, name: str, feature: str) -> int:
    return client.count(
        name,
        count_filter=Filter(
            must=[FieldCondition(key="feature", match=MatchValue(value=feature))]
        ),
    ).count


@needs_insert_only
def test_live_admin_on_a_qdrant_collection(client):
    name = "live" + uuid.uuid4().hex[:10]
    try:
        client.sharding.create_collection(
            name,
            "feature",
            feature_type="keyword",
            n_slots=3,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        before = client.sharding.get_map(name)
        added = client.sharding.add_slot(name, "s3")
        assert added.slots == [*before.slots, "s3"]
        assert added.slot_load["s3"] == 0
        assert added.version == before.version + 1
        again = client.sharding.add_slot(name, "s3")
        assert again.version == added.version
        with pytest.raises(ValueError):
            client.sharding.add_slot(name, " ")

        client.upsert(name, [_point(i, "books") for i in range(10)], wait=True)
        client.upsert(name, [_point(i, "toys") for i in range(10, 14)], wait=True)
        placed = client.sharding.get_map(name)
        version = placed.version
        load = client.sharding.read_load(name)
        assert load.slots == tuple(placed.slots)
        assert sum(load.counts["s3"].values()) == 0
        assert client.sharding.get_map(name).version == version
        assert _count(client, name, "books") == 10
        assert _count(client, name, "toys") == 4

        books_slot = placed.features["books"].shard_keys[0]
        toys_slot = placed.features["toys"].shard_keys[0]
        if toys_slot != books_slot:
            client.sharding.migrate(name, "toys", books_slot, grace_seconds=0)
        shared = client.sharding.get_map(name).features["toys"].shard_keys
        assert shared == [books_slot]
        doubled = client.sharding.read_load(name)
        assert sum(doubled.counts[books_slot].values()) == 14

        version_before_plan = client.sharding.get_map(name).version
        plan = client.sharding.plan_rebalance(name)
        assert plan.reason in (StopReason.within_ratio, StopReason.no_legal_move)
        assert client.sharding.get_map(name).version == version_before_plan
        if plan.moves:
            applied = client.sharding.apply_rebalance(name, plan, grace_seconds=0)
            assert applied.features["books"].state == FeatureState.stable
            assert applied.features["toys"].state == FeatureState.stable
        assert _ids(client, name, "books") == set(range(10))
        assert _ids(client, name, "toys") == set(range(10, 14))

        free = next(
            slot
            for slot in client.sharding.get_map(name).slots
            if slot not in client.sharding.get_map(name).features["books"].shard_keys
        )
        client.sharding.start_move(name, "books", [free], grace_seconds=0)
        assert client.sharding.get_map(name).features["books"].state == FeatureState.copying
        copied = client.sharding.copy_feature(name, "books")
        assert copied == 10
        during = client.sharding.read_load(name)
        physical = sum(
            during.counts[slot].get("books", 0) for slot in during.slots
        )
        assert physical == 20
        finished = client.sharding.resume_move(name, "books", grace_seconds=0)
        assert finished.features["books"].state == FeatureState.stable
        assert finished.features["books"].shard_keys == [free]
        assert _count(client, name, "books") == 10
        after = client.sharding.read_load(name)
        assert sum(after.counts[slot].get("books", 0) for slot in after.slots) == 10

        client.delete(name, [1], wait=True)
        assert 1 not in _ids(client, name, "books")
        assert _count(client, name, "books") == 9
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)


@needs_insert_only
def test_live_admin_on_a_qdrant_collection_async(tmp_path):
    asyncio.run(_live_admin_async(tmp_path))


async def _live_admin_async(tmp_path) -> None:
    client = AsyncShardedQdrantClient(map_folder=FileMapFolder(tmp_path), **local_kwargs())
    name = "alive" + uuid.uuid4().hex[:10]
    try:
        await client.sharding.create_collection(
            name,
            "feature",
            feature_type="keyword",
            n_slots=3,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        before = await client.sharding.get_map(name)
        added = await client.sharding.add_slot(name, "s3")
        assert added.slots == [*before.slots, "s3"]
        assert (await client.sharding.add_slot(name, "s3")).version == added.version
        await client.upsert(name, [_point(i, "books") for i in range(6)], wait=True)
        await client.upsert(name, [_point(i, "toys") for i in range(6, 9)], wait=True)
        load = await client.sharding.read_load(name)
        assert sum(sum(counts.values()) for counts in load.counts.values()) == 9
        placed = await client.sharding.get_map(name)
        books_slot = placed.features["books"].shard_keys[0]
        toys_slot = placed.features["toys"].shard_keys[0]
        if toys_slot != books_slot:
            await client.sharding.migrate(name, "toys", books_slot, grace_seconds=0)
        plan = await client.sharding.plan_rebalance(name)
        if plan.moves:
            applied = await client.sharding.apply_rebalance(name, plan, grace_seconds=0)
            assert applied.features["books"].state == FeatureState.stable
            assert applied.features["toys"].state == FeatureState.stable
        current = await client.sharding.get_map(name)
        free = next(
            slot
            for slot in current.slots
            if slot not in current.features["books"].shard_keys
        )
        await client.sharding.start_move(name, "books", [free], grace_seconds=0)
        finished = await client.sharding.resume_move(name, "books", grace_seconds=0)
        assert finished.features["books"].state == FeatureState.stable
        assert finished.features["books"].shard_keys == [free]
        counted = await client.count(
            name,
            count_filter=Filter(
                must=[FieldCondition(key="feature", match=MatchValue(value="books"))]
            ),
        )
        assert counted.count == 6
    finally:
        if await client.collection_exists(name):
            await client.delete_collection(name)
        await client.close()
