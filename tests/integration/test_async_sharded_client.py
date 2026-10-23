import asyncio
import inspect
import socket
import uuid
import warnings
from pathlib import Path

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    DeleteOperation,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    PointStruct,
    PointVectors,
    PointsList,
    QueryRequest,
    ShardingMethod,
    UpsertOperation,
    VectorParams,
)

from qdrant_custom_sharding import AsyncShardedQdrantClient
from qdrant_custom_sharding.errors import MapNotFound, WriteDenied
from qdrant_custom_sharding.map_stores import FileMapFolder, _s3_client, open_map_folder
from qdrant_custom_sharding.models import FeatureState
from qdrant_custom_sharding.qdrant_store import (
    AsyncQdrantVectorStore,
    insert_only_supported,
)
from qdrant_custom_sharding.routing import pick_slot
from qdrant_custom_sharding.sharding_warnings import (
    FanOutWarning,
    FeatureMovedWarning,
    ShardingWarning,
    StaleMapWarning,
)
from tests.local_qdrant import local_kwargs

pytestmark = pytest.mark.integration

_INSERT_ONLY = "moves need qdrant-client and server >= 1.17 for insert-only upserts"
needs_insert_only = pytest.mark.skipif(not insert_only_supported(), reason=_INSERT_ONLY)

VECTOR = [0.1, 0.2, 0.3, 0.4]
_BUCKET = "sharding-maps"


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


def _open(tmp_path: Path, **kwargs: object) -> AsyncShardedQdrantClient:
    return AsyncShardedQdrantClient(map_folder=FileMapFolder(tmp_path), **local_kwargs(), **kwargs)


@pytest.fixture
def client(tmp_path: Path):
    opened = _open(tmp_path)
    yield opened
    asyncio.run(opened.close())


@pytest.fixture
def collection(client):
    name = "a" + uuid.uuid4().hex[:12]
    yield name
    asyncio.run(_drop(client, name))


async def _drop(client, name: str) -> None:
    if await client.collection_exists(name):
        await client.delete_collection(name)


def _point(point_id: int, feature: int | str, field: str = "feature") -> PointStruct:
    return PointStruct(id=point_id, vector=VECTOR, payload={field: feature})


def _filter(value: object, field: str = "feature") -> Filter:
    return Filter(must=FieldCondition(key=field, match=MatchValue(value=value)))


async def _create(
    client, name: str, field: str = "feature", feature_type: str = "keyword", n_slots: int = 3
):
    return await client.sharding.create_collection(
        name,
        field,
        feature_type=feature_type,
        n_slots=n_slots,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )


def _ids(response) -> set:
    return {point.id for point in response.points}


async def _slot_count(client, name: str, slot: str, flt: Filter) -> int:
    counted = await client.count(name, count_filter=flt, shard_key_selector=slot)
    return counted.count


async def _cluster_info(client, name: str):
    method = getattr(type(client), "collection_cluster_info", None)
    if method is not None and inspect.iscoroutinefunction(method):
        return await client.collection_cluster_info(name)
    http = client.http
    api = getattr(http, "cluster_api", None)
    if api is None:
        api = http.distributed_api
    body = api.collection_cluster_info(name)
    if inspect.isawaitable(body):
        body = await body
    if hasattr(body, "result"):
        return body.result
    return body


def test_unsharded_collection_passes_through(client, collection):
    asyncio.run(_unsharded(client, collection))


async def _unsharded(client, collection):
    await client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    await client.upsert(collection, [_point(1, "books")], wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = await client.query_points(collection, VECTOR, limit=10)
    assert _ids(found) == {1}
    assert caught == []


def test_create_collection_makes_slots_and_an_index(client):
    asyncio.run(_layout(client))


async def _layout(client):
    keyword = "k" + uuid.uuid4().hex[:12]
    integer = "i" + uuid.uuid4().hex[:12]
    try:
        await _create(client, keyword, "feature", "keyword", 3)
        await _create(client, integer, "subject_id", "integer", 3)
        await _assert_layout(client, keyword, "feature", "keyword")
        await _assert_layout(client, integer, "subject_id", "integer")
        await client.upsert(integer, [_point(1, 7, "subject_id")], wait=True)
        counted = await client.count(integer, count_filter=_filter(7, "subject_id"))
        assert counted.count == 1
    finally:
        await _drop(client, keyword)
        await _drop(client, integer)


def test_upsert_positional_args_and_a_new_value(client, collection):
    asyncio.run(_positional(client, collection))


async def _positional(client, collection):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books")])
    await client.upsert(collection, [_point(2, "fresh")], wait=True)
    doc = await client.sharding.get_map(collection)
    assert "books" in doc.features
    assert "fresh" in doc.features
    books = await client.count(collection, count_filter=_filter("books"))
    fresh = await client.count(collection, count_filter=_filter("fresh"))
    assert books.count == 1
    assert fresh.count == 1


def test_filtered_reads_hit_only_the_feature_slots(client, collection):
    asyncio.run(_filtered(client, collection))


async def _filtered(client, collection):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    doc = await client.sharding.get_map(collection)
    books = set(doc.features["books"].shard_keys)
    other = next(slot for slot in doc.slots if slot not in books)
    await client.upsert(collection, [_point(99, "books")], shard_key_selector=other, wait=True)
    flt = _filter("books")
    found = await client.query_points(collection, VECTOR, query_filter=flt, limit=10)
    assert _ids(found) == {1}
    explicit = await client.query_points(
        collection, VECTOR, query_filter=flt, shard_key_selector=other, limit=10
    )
    assert _ids(explicit) == {99}
    counted = await client.count(collection, count_filter=flt)
    assert counted.count == 1
    assert await _slot_count(client, collection, other, flt) == 1
    records, _offset = await client.scroll(collection, scroll_filter=flt, limit=10)
    assert {record.id for record in records} == {1}


def test_read_without_a_feature_fans_out(client, collection):
    asyncio.run(_fanout(client, collection))


async def _fanout(client, collection):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = await client.query_points(collection, VECTOR, limit=10)
    assert _ids(found) == {1, 2}
    fan = [item for item in caught if issubclass(item.category, FanOutWarning)]
    assert fan
    assert Path(fan[0].filename).name == "test_async_sharded_client.py"


def test_upsert_moves_a_point_when_the_feature_changes(client, collection):
    asyncio.run(_move(client, collection))


async def _move(client, collection):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books")], wait=True)
    old = list((await client.sharding.get_map(collection)).features["books"].shard_keys)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await client.upsert(collection, [_point(1, "toys")], wait=True)
    assert any(issubclass(item.category, FeatureMovedWarning) for item in caught)
    doc = await client.sharding.get_map(collection)
    assert set(old).isdisjoint(doc.features["toys"].shard_keys)
    for slot in old:
        records, _offset = await client.scroll(collection, shard_key_selector=slot, limit=10)
        assert all(record.id != 1 for record in records)


def test_set_payload_moves_a_point_when_the_feature_changes(client, collection):
    asyncio.run(_set_payload(client, collection))


async def _set_payload(client, collection):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books")], wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await client.set_payload(collection, {"feature": "toys"}, [1])
    assert any(issubclass(item.category, FeatureMovedWarning) for item in caught)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FanOutWarning)
        got = await client.retrieve(collection, [1])
    assert got[0].payload["feature"] == "toys"


def test_clear_payload_raises(client, collection):
    asyncio.run(_clear(client, collection))


async def _clear(client, collection):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books")], wait=True)
    with pytest.raises(ValueError, match="feature"):
        await client.clear_payload(collection, [1])


def test_batch_update_points(client, collection):
    asyncio.run(_batch(client, collection))


async def _batch(client, collection):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    results = await client.batch_update_points(
        collection,
        [
            UpsertOperation(upsert=PointsList(points=[_point(3, "books")])),
            DeleteOperation(delete=FilterSelector(filter=_filter("toys"))),
        ],
    )
    assert len(results) == 2
    books = await client.count(collection, count_filter=_filter("books"))
    toys = await client.count(collection, count_filter=_filter("toys"))
    assert books.count == 2
    assert toys.count == 0


def test_query_batch_points_uses_a_copy_of_the_request(client, collection):
    asyncio.run(_query_batch(client, collection))


async def _query_batch(client, collection):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    doc = await client.sharding.get_map(collection)
    other = next(slot for slot in doc.slots if slot not in doc.features["books"].shard_keys)
    await client.upsert(collection, [_point(99, "books")], shard_key_selector=other, wait=True)
    request = QueryRequest(query=VECTOR, filter=_filter("books"), limit=10)
    found = await client.query_batch_points(collection, [request])
    assert request.shard_key is None
    assert _ids(found[0]) == {1}


def test_upload_points_routes_each_batch(client, collection):
    asyncio.run(_upload(client, collection))


async def _upload(client, collection):
    await _create(client, collection)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await client.upload_points(
            collection,
            [_point(1, "books"), _point(2, "toys")],
            batch_size=1,
            parallel=2,
            wait=True,
        )
    assert any(issubclass(item.category, ShardingWarning) for item in caught)
    books = await client.count(collection, count_filter=_filter("books"))
    toys = await client.count(collection, count_filter=_filter("toys"))
    assert books.count == 1
    assert toys.count == 1


@needs_insert_only
def test_migrate_with_zero_grace_keeps_every_point(client, collection):
    asyncio.run(_migrate(client, collection))


async def _migrate(client, collection):
    await _create(client, collection, n_slots=3)
    await client.upsert(collection, [_point(i, "books") for i in range(6)], wait=True)
    current = set((await client.sharding.get_map(collection)).features["books"].shard_keys)
    target = next(slot for slot in ("s0", "s1", "s2") if slot not in current)
    moved = await client.sharding.migrate(collection, "books", target, grace_seconds=0)
    assert moved.features["books"].state == FeatureState.stable
    assert moved.features["books"].shard_keys == [target]
    total = 0
    for slot in moved.slots:
        total += await _slot_count(client, collection, slot, _filter("books"))
    assert total == 6


@needs_insert_only
def test_split_spreads_points_with_no_duplicates(client, collection):
    asyncio.run(_split(client, collection))


async def _split(client, collection):
    await _create(client, collection, n_slots=4)
    await client.upsert(collection, [_point(i, "books") for i in range(12)], wait=True)
    doc = await client.sharding.split(collection, "books", 3, grace_seconds=0)
    assert doc.features["books"].state == FeatureState.stable
    targets = doc.features["books"].shard_keys
    assert len(targets) == 3
    expected: dict[str, int] = {}
    for point_id in range(12):
        slot = pick_slot(targets, point_id)
        expected[slot] = expected.get(slot, 0) + 1
    for slot in doc.slots:
        assert await _slot_count(client, collection, slot, _filter("books")) == expected.get(slot, 0)
    assert len(expected) >= 2


def test_delete_collection_removes_the_map(client, collection, tmp_path: Path):
    asyncio.run(_delete_map(client, collection, tmp_path))


async def _delete_map(client, collection, tmp_path: Path):
    await _create(client, collection)
    path = tmp_path / f"{collection}.json"
    assert path.is_file()
    assert await client.delete_collection(collection) is True
    assert not path.exists()
    assert await client.sharding.get_map(collection) is None


def test_broken_map_fans_out_reads_and_denies_writes(client, collection, tmp_path: Path):
    asyncio.run(_broken_map(client, collection, tmp_path))


async def _broken_map(client, collection, tmp_path: Path):
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books")], wait=True)
    path = tmp_path / f"{collection}.json"
    path.unlink()
    path.mkdir()
    fresh = _open(tmp_path)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = await fresh.query_points(collection, VECTOR, limit=10)
        assert _ids(found) == {1}
        assert any(issubclass(item.category, StaleMapWarning) for item in caught)
        with pytest.raises(WriteDenied):
            await fresh.upsert(collection, [_point(2, "books")])
    finally:
        await fresh.close()
        if path.is_dir():
            path.rmdir()


def test_create_collection_rejects_sharding_overrides(client, collection):
    asyncio.run(_reject(client, collection))


async def _reject(client, collection):
    with pytest.raises(ValueError, match="sharding_method"):
        await client.sharding.create_collection(
            collection,
            "feature",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            sharding_method=ShardingMethod.AUTO,
        )


@needs_insert_only
def test_copy_keeps_a_concurrent_update_and_drops_a_concurrent_delete(
    client, collection, tmp_path: Path
):
    asyncio.run(_copy_race(client, collection, tmp_path))


async def _copy_race(client, collection, tmp_path: Path) -> None:
    await _create(client, collection, n_slots=2)
    await client.upsert(
        collection,
        [
            PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"}),
            PointStruct(id=2, vector=VECTOR, payload={"feature": "books", "note": "old"}),
        ],
        wait=True,
    )
    doc = await client.sharding.get_map(collection)
    current = doc.features["books"].shard_keys[0]
    target = next(slot for slot in doc.slots if slot != current)
    await client.sharding.start_move(collection, "books", [target], grace_seconds=0)
    writer = _open(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    original = AsyncQdrantVectorStore.scroll

    async def scroll(self, collection_name, slot, field, values, batch):
        first = True
        async for chunk in original(self, collection_name, slot, field, values, batch):
            if first:
                first = False
                ids = {point.id for point in chunk}
                if ids != {1, 2}:
                    raise AssertionError(f"scroll chunk ids {ids}")
                started.set()
                await release.wait()
            yield chunk

    AsyncQdrantVectorStore.scroll = scroll
    try:
        task = asyncio.create_task(client.sharding.copy_feature(collection, "books", batch=10))
        await asyncio.wait_for(started.wait(), 30)
        await writer.upsert(
            collection,
            [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "new"})],
            wait=True,
        )
        await writer.delete(
            collection,
            Filter(
                must=[
                    FieldCondition(key="feature", match=MatchValue(value="books")),
                    HasIdCondition(has_id=[2]),
                ]
            ),
            wait=True,
        )
        release.set()
        await task
        await client.sharding.open_dual(collection, "books", grace_seconds=0)
        await client.sharding.cutover(collection, "books", grace_seconds=0)
        await client.sharding.finish_move(collection, "books")
        found = await client.retrieve(collection, [1])
        assert found[0].payload["note"] == "new"
        assert await client.retrieve(collection, [2]) == []
    finally:
        AsyncQdrantVectorStore.scroll = original
        release.set()
        await writer.close()


def test_unknown_filter_reloads_and_routes_when_another_client_added_it(
    client, collection, tmp_path: Path
):
    asyncio.run(_unknown_routes(client, collection, tmp_path))


async def _unknown_routes(client, collection, tmp_path: Path) -> None:
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books")], wait=True)
    await client.query_points(collection, VECTOR, query_filter=_filter("books"), limit=10)
    other = _open(tmp_path)
    try:
        await other.upsert(collection, [_point(2, "toys")], wait=True)
        doc = await other.sharding.get_map(collection)
        toy_slots = set(doc.features["toys"].shard_keys)
        decoy = next(slot for slot in doc.slots if slot not in toy_slots)
        await other.upsert(collection, [_point(99, "toys")], shard_key_selector=decoy, wait=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found = await client.query_points(
                collection, VECTOR, query_filter=_filter("toys"), limit=10
            )
        assert _ids(found) == {2}
        assert not any(issubclass(item.category, FanOutWarning) for item in caught)
    finally:
        await other.close()


def test_unknown_filter_fans_out_when_the_value_stays_unknown(client, collection):
    asyncio.run(_unknown_fans_out(client, collection))


async def _unknown_fans_out(client, collection) -> None:
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books")], wait=True)
    await client.query_points(collection, VECTOR, query_filter=_filter("books"), limit=10)
    slot = (await client.sharding.get_map(collection)).slots[0]
    await client.upsert(collection, [_point(7, "ghost")], shard_key_selector=slot, wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = await client.query_points(
            collection, VECTOR, query_filter=_filter("ghost"), limit=10
        )
    assert _ids(found) == {7}
    assert any(issubclass(item.category, FanOutWarning) for item in caught)


@needs_insert_only
def test_feature_filter_mutations_hit_both_copies_in_dual_and_cutover(client, collection):
    asyncio.run(_both_copies(client, collection))


async def _both_copies(client, collection) -> None:
    await client.sharding.create_collection(
        collection,
        "feature",
        n_slots=2,
        vectors_config={"text": VectorParams(size=4, distance=Distance.COSINE)},
    )
    await client.upsert(
        collection,
        [
            PointStruct(id=point_id, vector={"text": VECTOR}, payload={"feature": "books"})
            for point_id in (1, 2, 3)
        ],
        wait=True,
    )
    doc = await client.sharding.get_map(collection)
    old = doc.features["books"].shard_keys[0]
    new = next(slot for slot in doc.slots if slot != old)
    await client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    await client.sharding.copy_feature(collection, "books")
    await client.sharding.open_dual(collection, "books", grace_seconds=0)
    await client.set_payload(collection, {"note": "dual"}, _filter("books"), wait=True)
    await client.delete_vectors(collection, ["text"], _filter("books"), wait=True)
    assert await _notes(client, collection, old) == {"dual"}
    assert await _notes(client, collection, new) == {"dual"}
    assert await _text_vectors(client, collection, old) == [None, None, None]
    assert await _text_vectors(client, collection, new) == [None, None, None]
    await client.upsert(
        collection,
        [
            PointStruct(
                id=point_id,
                vector={"text": VECTOR},
                payload={"feature": "books", "note": "dual"},
            )
            for point_id in (1, 2, 3)
        ],
        wait=True,
    )
    await client.sharding.cutover(collection, "books", grace_seconds=0)
    await client.set_payload(collection, {"note": "cut"}, _filter("books"), wait=True)
    await client.delete_vectors(collection, ["text"], _filter("books"), wait=True)
    assert await _notes(client, collection, old) == {"cut"}
    assert await _notes(client, collection, new) == {"cut"}
    assert await _text_vectors(client, collection, old) == [None, None, None]
    assert await _text_vectors(client, collection, new) == [None, None, None]
    await client.delete(collection, _filter("books"), wait=True)
    assert await _slot_count(client, collection, old, _filter("books")) == 0
    assert await _slot_count(client, collection, new, _filter("books")) == 0


async def _slot_records(client, collection, slot):
    return await client.retrieve(
        collection,
        [1, 2, 3],
        with_payload=True,
        with_vectors=True,
        shard_key_selector=slot,
    )


async def _notes(client, collection, slot) -> set:
    records = await _slot_records(client, collection, slot)
    return {record.payload.get("note") for record in records}


async def _text_vectors(client, collection, slot) -> list:
    vectors = []
    for record in await _slot_records(client, collection, slot):
        vector = record.vector
        if isinstance(vector, dict):
            vectors.append(vector.get("text"))
        else:
            vectors.append(vector)
    return vectors


def test_alias_rewrites_old_points_and_drop_alias_removes_the_old_id(client, collection):
    asyncio.run(_alias(client, collection))


async def _alias(client, collection) -> None:
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books"), _point(2, "books")], wait=True)
    changed = await client.sharding.alias_feature(
        collection, "books", "libros", grace_seconds=0
    )
    assert changed == 2
    await client.upsert(collection, [_point(3, "libros")], wait=True)
    libros = await client.query_points(
        collection, VECTOR, query_filter=_filter("libros"), limit=10
    )
    assert _ids(libros) == {1, 2, 3}
    payloads = {point.id: point.payload["feature"] for point in libros.points}
    assert payloads[1] == "libros"
    assert payloads[2] == "libros"
    books = await client.query_points(
        collection, VECTOR, query_filter=_filter("books"), limit=10
    )
    assert _ids(books) == set()
    await client.upsert(collection, [_point(4, "books")], wait=True)
    late = await client.query_points(
        collection, VECTOR, query_filter=_filter("books"), limit=10
    )
    assert _ids(late) == {4}
    dropped = await client.sharding.drop_alias(collection, "books")
    assert dropped == 1
    after = await client.query_points(
        collection, VECTOR, query_filter=_filter("libros"), limit=10
    )
    assert _ids(after) == {1, 2, 3, 4}
    assert {point.payload["feature"] for point in after.points} == {"libros"}
    doc = await client.sharding.get_map(collection)
    assert "books" not in doc.features["libros"].aliases
    gone = await client.query_points(
        collection, VECTOR, query_filter=_filter("books"), limit=10
    )
    assert _ids(gone) == set()


def test_delete_by_feature_filter(client, collection):
    asyncio.run(_delete_filter(client, collection))


async def _delete_filter(client, collection) -> None:
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    await client.delete(collection, _filter("books"))
    books = await client.count(collection, count_filter=_filter("books"))
    toys = await client.count(collection, count_filter=_filter("toys"))
    assert books.count == 0
    assert toys.count == 1


def test_skipped_feature_check_leaves_the_old_copy(client, collection, tmp_path: Path):
    asyncio.run(_skipped_check(client, collection, tmp_path))


async def _skipped_check(client, collection, tmp_path: Path) -> None:
    await _create(client, collection)
    blind = _open(tmp_path, detect_feature_changes=False)
    try:
        await blind.upsert(collection, [_point(1, "books")], wait=True)
        old = list((await blind.sharding.get_map(collection)).features["books"].shard_keys)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await blind.upsert(collection, [_point(1, "toys")], wait=True)
        assert not any(issubclass(item.category, FeatureMovedWarning) for item in caught)
        doc = await blind.sharding.get_map(collection)
        assert set(old).isdisjoint(doc.features["toys"].shard_keys)
        total = 0
        for slot in doc.slots:
            total += (await client.count(collection, shard_key_selector=slot)).count
        assert total == 2
    finally:
        await blind.close()


def test_s3_map_folder_routes_a_collection():
    asyncio.run(_s3_routes())


async def _s3_routes() -> None:
    prefix = f"tests/{uuid.uuid4().hex}/"
    folder = open_map_folder(f"s3://{_BUCKET}/{prefix}")
    routed = AsyncShardedQdrantClient(map_folder=folder, **local_kwargs())
    name = "a" + uuid.uuid4().hex[:12]
    key = prefix + name + ".json"
    try:
        await _create(routed, name, n_slots=2)
        await routed.upsert(name, [_point(1, "books")], wait=True)
        counted = await routed.count(name, count_filter=_filter("books"))
        assert counted.count == 1
        doc = await routed.sharding.get_map(name)
        assert doc.features["books"].shard_keys
        await routed.delete_collection(name)
        with pytest.raises(MapNotFound):
            folder.store_for(name).load()
    finally:
        if await routed.collection_exists(name):
            await routed.delete_collection(name)
        await routed.close()
        _s3_client().delete_object(Bucket=_BUCKET, Key=key)


async def _assert_layout(client, name: str, field: str, kind: str) -> None:
    info = await client.get_collection(name)
    method = info.config.params.sharding_method
    assert getattr(method, "value", method) == "custom"
    schema = info.payload_schema[field]
    assert getattr(schema.data_type, "value", schema.data_type) == kind
    cluster = await _cluster_info(client, name)
    keys = set()
    for shard in list(cluster.local_shards) + list(cluster.remote_shards):
        if shard.shard_key is not None:
            keys.add(str(shard.shard_key))
    assert keys == {"s0", "s1", "s2"}


def test_writes_that_omit_the_feature(client, collection):
    asyncio.run(_writes_that_omit_the_feature(client, collection))


def _tag(value: str) -> Filter:
    return Filter(must=FieldCondition(key="tag", match=MatchValue(value=value)))


async def _fan(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await fn()
    fan = [item for item in caught if issubclass(item.category, FanOutWarning)]
    assert fan
    assert Path(fan[0].filename).name == "test_async_sharded_client.py"
    return result


async def _got(client, collection, point_id):
    rows = await client.retrieve(collection, [point_id], with_payload=True, with_vectors=True)
    if not rows:
        return None
    return rows[0]


def _vec(record):
    if record is None:
        return None
    vector = record.vector
    if isinstance(vector, dict):
        if not vector:
            return None
        vector = vector.get("") if "" in vector else next(iter(vector.values()))
    return vector


async def _writes_that_omit_the_feature(client, collection):
    await _create(client, collection)
    await client.upsert(
        collection,
        [
            PointStruct(
                id=point_id,
                vector=VECTOR,
                payload={"feature": feature, "tag": tag},
            )
            for point_id, feature, tag in (
                (1, "books", "red"),
                (2, "toys", "red"),
                (3, "books", "red"),
                (4, "toys", "red"),
                (5, "books", "red"),
                (6, "books", "red"),
                (7, "books", "red"),
                (8, "books", "del"),
                (9, "toys", "keep"),
                (10, "books", "pay"),
                (11, "books", "trim"),
                (12, "books", "novec"),
                (13, "books", "red"),
            )
        ],
        wait=True,
    )
    await _fan(lambda: client.delete(collection, [1, 99], wait=True))
    assert await _got(client, collection, 1) is None
    assert await _got(client, collection, 2) is not None

    await _fan(lambda: client.set_payload(collection, {"tag": "blue"}, [3, 4], wait=True))
    third = await _got(client, collection, 3)
    fourth = await _got(client, collection, 4)
    assert third.payload == {"feature": "books", "tag": "blue"}
    assert fourth.payload == {"feature": "toys", "tag": "blue"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(UnexpectedResponse) as exc:
            await client.set_payload(collection, {"tag": "blue"}, [98], wait=True)
    assert exc.value.status_code == 404
    assert any(issubclass(item.category, FanOutWarning) for item in caught)
    assert Path(caught[0].filename).name == "test_async_sharded_client.py"

    await _fan(lambda: client.delete_payload(collection, ["tag"], [5], wait=True))
    assert (await _got(client, collection, 5)).payload == {"feature": "books"}

    before = list(_vec(await _got(client, collection, 6)))
    await _fan(
        lambda: client.update_vectors(
            collection, [PointVectors(id=6, vector=[0.9, 0.1, 0.2, 0.3])], wait=True
        )
    )
    assert list(_vec(await _got(client, collection, 6))) != before

    await _fan(lambda: client.delete_vectors(collection, [""], [7], wait=True))
    assert not _vec(await _got(client, collection, 7))
    assert (await _got(client, collection, 7)).payload["feature"] == "books"

    await _fan(lambda: client.delete(collection, _tag("del"), wait=True))
    assert await _got(client, collection, 8) is None
    assert (await _got(client, collection, 9)).payload["feature"] == "toys"

    await _fan(lambda: client.set_payload(collection, {"note": "yes"}, _tag("pay"), wait=True))
    paid = (await _got(client, collection, 10)).payload
    assert paid["note"] == "yes"
    assert paid["feature"] == "books"

    await _fan(lambda: client.delete_payload(collection, ["tag"], _tag("trim"), wait=True))
    assert (await _got(client, collection, 11)).payload == {"feature": "books"}

    await _fan(lambda: client.delete_vectors(collection, [""], _tag("novec"), wait=True))
    assert not _vec(await _got(client, collection, 12))
    assert (await _got(client, collection, 12)).payload["feature"] == "books"

    await client.overwrite_payload(
        collection, {"feature": "books", "tag": "green"}, [13], wait=True
    )
    assert (await _got(client, collection, 13)).payload == {"feature": "books", "tag": "green"}
    with pytest.raises(ValueError, match="overwrite_payload"):
        await client.overwrite_payload(collection, {"tag": "x"}, [13])
    with pytest.raises(ValueError, match="clear_payload"):
        await client.clear_payload(collection, [2])
    with pytest.raises(UnexpectedResponse) as mixed:
        await client.set_payload(collection, {"tag": "mix"}, [2, 97], wait=True)
    assert mixed.value.status_code == 404
    assert (await _got(client, collection, 2)).payload["tag"] == "mix"


def _other_slot(doc):
    current = doc.features["books"].shard_keys[0]
    return next(slot for slot in doc.slots if slot != current)


async def _note_on(client, collection, slot, point_id=1):
    rows = await client.retrieve(
        collection, [point_id], with_payload=True, shard_key_selector=slot
    )
    assert len(rows) == 1
    return rows[0].payload["note"]


@needs_insert_only
def test_set_payload_by_id_during_copy_reaches_the_copied_point(client, collection):
    asyncio.run(_id_update_after_copy(client, collection))


async def _id_update_after_copy(client, collection) -> None:
    await _create(client, collection, n_slots=2)
    await client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"})],
        wait=True,
    )
    new = _other_slot(await client.sharding.get_map(collection))
    await client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    await client.sharding.copy_feature(collection, "books")
    await client.set_payload(collection, {"note": "new"}, [1], wait=True)
    assert await _note_on(client, collection, new) == "new"
    await client.sharding.open_dual(collection, "books", grace_seconds=0)
    await client.sharding.cutover(collection, "books", grace_seconds=0)
    await client.sharding.finish_move(collection, "books")
    found = await client.retrieve(collection, [1])
    assert found[0].payload["note"] == "new"


@needs_insert_only
def test_set_payload_by_id_before_the_copy_survives_it(client, collection):
    asyncio.run(_id_update_before_copy(client, collection))


async def _id_update_before_copy(client, collection) -> None:
    await _create(client, collection, n_slots=2)
    await client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"})],
        wait=True,
    )
    new = _other_slot(await client.sharding.get_map(collection))
    await client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    await client.set_payload(collection, {"note": "new"}, [1], wait=True)
    await client.sharding.copy_feature(collection, "books")
    assert await _note_on(client, collection, new) == "new"


@needs_insert_only
def test_filter_set_payload_before_the_copy_survives_it(client, collection):
    asyncio.run(_filter_update_before_copy(client, collection))


async def _filter_update_before_copy(client, collection) -> None:
    await _create(client, collection, n_slots=2)
    await client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"})],
        wait=True,
    )
    new = _other_slot(await client.sharding.get_map(collection))
    await client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    await client.set_payload(collection, {"note": "new"}, _filter("books"), wait=True)
    await client.sharding.copy_feature(collection, "books")
    assert await _note_on(client, collection, new) == "new"


@needs_insert_only
def test_filter_set_payload_during_copy_keeps_the_new_value(
    client, collection, tmp_path: Path
):
    asyncio.run(_filter_update_during_copy(client, collection, tmp_path))


async def _filter_update_during_copy(client, collection, tmp_path: Path) -> None:
    await _create(client, collection, n_slots=2)
    await client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"})],
        wait=True,
    )
    new = _other_slot(await client.sharding.get_map(collection))
    await client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    writer = _open(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    original = AsyncQdrantVectorStore.scroll

    async def scroll(self, collection_name, slot, field, values, batch):
        first = True
        async for chunk in original(self, collection_name, slot, field, values, batch):
            if first:
                first = False
                started.set()
                await release.wait()
            yield chunk

    AsyncQdrantVectorStore.scroll = scroll
    try:
        task = asyncio.create_task(client.sharding.copy_feature(collection, "books"))
        await asyncio.wait_for(started.wait(), 30)
        await writer.set_payload(collection, {"note": "new"}, _filter("books"), wait=True)
        release.set()
        await task
        assert await _note_on(client, collection, new) == "new"
    finally:
        AsyncQdrantVectorStore.scroll = original
        release.set()
        await writer.close()


@needs_insert_only
def test_set_payload_by_id_during_cutover_hits_the_new_slot(client, collection):
    asyncio.run(_id_update_during_cutover(client, collection))


async def _id_update_during_cutover(client, collection) -> None:
    await _create(client, collection, n_slots=2)
    await client.upsert(collection, [_point(1, "books")], wait=True)
    new = _other_slot(await client.sharding.get_map(collection))
    await client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    await client.sharding.copy_feature(collection, "books")
    await client.sharding.open_dual(collection, "books", grace_seconds=0)
    await client.sharding.cutover(collection, "books", grace_seconds=0)
    await client.upsert(
        collection,
        [PointStruct(id=2, vector=VECTOR, payload={"feature": "books"})],
        wait=True,
    )
    await client.set_payload(collection, {"note": "late"}, [2], wait=True)
    found = await client.retrieve(collection, [2])
    assert found[0].payload["note"] == "late"


def test_set_payload_by_id_reloads_a_feature_added_by_another_client(
    client, collection, tmp_path: Path
):
    asyncio.run(_reload_id_write(client, collection, tmp_path))


async def _reload_id_write(client, collection, tmp_path: Path) -> None:
    await _create(client, collection)
    await client.upsert(collection, [_point(1, "books")], wait=True)
    other = _open(tmp_path)
    try:
        await other.upsert(collection, [_point(2, "toys")], wait=True)
        await client.set_payload(collection, {"tag": "yes"}, [2], wait=True)
        got = await _got(client, collection, 2)
        assert got.payload["feature"] == "toys"
        assert got.payload["tag"] == "yes"
    finally:
        await other.close()


@needs_insert_only
def test_cutover_repairs_a_copy_that_lost_a_race(client, collection):
    asyncio.run(_cutover_repairs(client, collection))


async def _cutover_repairs(client, collection) -> None:
    await _create(client, collection, n_slots=2)
    await client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "kept"})],
        wait=True,
    )
    doc = await client.sharding.get_map(collection)
    old = doc.features["books"].shard_keys[0]
    new = next(slot for slot in doc.slots if slot != old)
    await client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    await client.sharding.copy_feature(collection, "books")
    await client.sharding.open_dual(collection, "books", grace_seconds=0)
    await client.delete(collection, [1], shard_key_selector=new, wait=True)
    await client.upsert(
        collection,
        [PointStruct(id=2, vector=VECTOR, payload={"feature": "books", "note": "added"})],
        shard_key_selector=old,
        wait=True,
    )
    result = await client.sharding.cutover(collection, "books", grace_seconds=0)
    assert result.copied == 2
    assert result.removed == 0
    rows = await client.retrieve(
        collection, [1, 2], shard_key_selector=new, with_payload=True
    )
    assert {row.id: row.payload["note"] for row in rows} == {1: "kept", 2: "added"}
