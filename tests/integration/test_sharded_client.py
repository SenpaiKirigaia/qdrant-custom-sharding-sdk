import socket
import threading
import uuid
import warnings
from pathlib import Path

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    DeleteOperation,
    PointStruct,
    PointVectors,
    PointsList,
    QueryRequest,
    ShardingMethod,
    UpsertOperation,
    VectorParams,
)

from qdrant_custom_sharding import ShardedQdrantClient
from qdrant_custom_sharding.errors import MapNotFound, WriteDenied
from qdrant_custom_sharding.map_stores import FileMapFolder, _s3_client, open_map_folder
from qdrant_custom_sharding.qdrant_store import QdrantVectorStore, insert_only_supported
from tests.local_qdrant import local_kwargs
from qdrant_custom_sharding.models import FeatureState
from qdrant_custom_sharding.routing import pick_slot
from qdrant_custom_sharding.sharding_warnings import (
    FanOutWarning,
    FeatureMovedWarning,
    ShardingWarning,
    StaleMapWarning,
)

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


def _open(tmp_path: Path, **kwargs: object) -> ShardedQdrantClient:
    return ShardedQdrantClient(map_folder=FileMapFolder(tmp_path), **local_kwargs(), **kwargs)


@pytest.fixture
def client(tmp_path: Path) -> ShardedQdrantClient:
    return _open(tmp_path)


@pytest.fixture
def collection(client):
    name = "t" + uuid.uuid4().hex[:12]
    yield name
    if client.collection_exists(name):
        client.delete_collection(name)


def _point(point_id: int, feature: int | str, field: str = "feature") -> PointStruct:
    return PointStruct(id=point_id, vector=VECTOR, payload={field: feature})


def _filter(value: object, field: str = "feature") -> Filter:
    return Filter(must=FieldCondition(key=field, match=MatchValue(value=value)))


def _create(client, name: str, field: str = "feature", feature_type: str = "keyword", n_slots: int = 3):
    return client.sharding.create_collection(
        name,
        field,
        feature_type=feature_type,
        n_slots=n_slots,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )


def _ids(response) -> set:
    return {point.id for point in response.points}


def _slot_count(client, name: str, slot: str, flt: Filter) -> int:
    return client.count(name, count_filter=flt, shard_key_selector=slot).count


def test_unsharded_collection_passes_through(client, collection):
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    client.upsert(collection, [_point(1, "books")], wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = client.query_points(collection, VECTOR, limit=10)
    assert _ids(found) == {1}
    assert caught == []


def test_create_collection_makes_slots_and_an_index(client):
    keyword = "k" + uuid.uuid4().hex[:12]
    integer = "i" + uuid.uuid4().hex[:12]
    try:
        _create(client, keyword, "feature", "keyword", 3)
        _create(client, integer, "subject_id", "integer", 3)
        _assert_layout(client, keyword, "feature", "keyword")
        _assert_layout(client, integer, "subject_id", "integer")
        client.upsert(integer, [_point(1, 7, "subject_id")], wait=True)
        counted = client.count(integer, count_filter=_filter(7, "subject_id"))
        assert counted.count == 1
    finally:
        for name in (keyword, integer):
            if client.collection_exists(name):
                client.delete_collection(name)


def test_create_collection_rejects_sharding_overrides(client, collection):
    with pytest.raises(ValueError, match="sharding_method"):
        client.sharding.create_collection(
            collection,
            "feature",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            sharding_method=ShardingMethod.AUTO,
        )
    assert not client.collection_exists(collection)


def test_upsert_positional_args_and_a_new_value(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")])
    client.upsert(collection, [_point(2, "fresh")], wait=True)
    doc = client.sharding.get_map(collection)
    assert "books" in doc.features
    assert "fresh" in doc.features
    assert client.count(collection, count_filter=_filter("books")).count == 1
    assert client.count(collection, count_filter=_filter("fresh")).count == 1


def test_filtered_reads_hit_only_the_feature_slots(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    doc = client.sharding.get_map(collection)
    books = set(doc.features["books"].shard_keys)
    other = next(slot for slot in doc.slots if slot not in books)
    client.upsert(collection, [_point(99, "books")], shard_key_selector=other, wait=True)
    flt = _filter("books")
    assert _ids(client.query_points(collection, VECTOR, query_filter=flt, limit=10)) == {1}
    explicit = client.query_points(
        collection, VECTOR, query_filter=flt, shard_key_selector=other, limit=10
    )
    assert _ids(explicit) == {99}
    assert client.count(collection, count_filter=flt).count == 1
    assert _slot_count(client, collection, other, flt) == 1
    records, _offset = client.scroll(collection, scroll_filter=flt, limit=10)
    assert {record.id for record in records} == {1}
    facet = client.facet(
        collection, key="feature", facet_filter=flt, exact=True, limit=10
    )
    counts = {hit.value: hit.count for hit in facet.hits}
    assert counts.get("books") == 1


def test_read_without_a_feature_fans_out(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = client.query_points(collection, VECTOR, limit=10)
    assert _ids(found) == {1, 2}
    fan = [item for item in caught if issubclass(item.category, FanOutWarning)]
    assert fan
    assert Path(fan[0].filename).name == "test_sharded_client.py"


def test_sharding_warnings_can_be_silenced(client, collection, tmp_path: Path):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    quiet = _open(tmp_path, sharding_warnings=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert quiet.retrieve(collection, [1])[0].id == 1
    assert caught == []


def test_retrieve_warns_and_names_the_caller_file(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = client.retrieve(collection, [1])
    assert records[0].id == 1
    fan = [item for item in caught if issubclass(item.category, FanOutWarning)]
    assert len(fan) == 1
    assert Path(fan[0].filename).name == "test_sharded_client.py"


def test_upsert_moves_a_point_when_the_feature_changes(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    old = list(client.sharding.get_map(collection).features["books"].shard_keys)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.upsert(collection, [_point(1, "toys")], wait=True)
    assert any(issubclass(item.category, FeatureMovedWarning) for item in caught)
    doc = client.sharding.get_map(collection)
    assert set(old).isdisjoint(doc.features["toys"].shard_keys)
    for slot in old:
        records, _offset = client.scroll(collection, shard_key_selector=slot, limit=10)
        assert all(record.id != 1 for record in records)
    total = sum(_slot_count(client, collection, slot, _filter("toys")) for slot in doc.slots)
    assert total == 1


def test_set_payload_moves_a_point_when_the_feature_changes(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    old = list(client.sharding.get_map(collection).features["books"].shard_keys)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.set_payload(collection, {"feature": "toys"}, [1])
    assert any(issubclass(item.category, FeatureMovedWarning) for item in caught)
    for slot in old:
        records, _offset = client.scroll(collection, shard_key_selector=slot, limit=10)
        assert all(record.id != 1 for record in records)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        got = client.retrieve(collection, [1])
    assert got[0].payload["feature"] == "toys"


def test_clear_payload_raises(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    with pytest.raises(ValueError, match="feature"):
        client.clear_payload(collection, [1])


def test_delete_by_feature_filter(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    client.delete(collection, _filter("books"))
    assert client.count(collection, count_filter=_filter("books")).count == 0
    assert client.count(collection, count_filter=_filter("toys")).count == 1


def test_batch_update_points(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    results = client.batch_update_points(
        collection,
        [
            UpsertOperation(upsert=PointsList(points=[_point(3, "books")])),
            DeleteOperation(delete=FilterSelector(filter=_filter("toys"))),
        ],
    )
    assert len(results) == 2
    assert client.count(collection, count_filter=_filter("books")).count == 2
    assert client.count(collection, count_filter=_filter("toys")).count == 0


def test_query_batch_points_uses_a_copy_of_the_request(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books"), _point(2, "toys")], wait=True)
    doc = client.sharding.get_map(collection)
    other = next(slot for slot in doc.slots if slot not in doc.features["books"].shard_keys)
    client.upsert(collection, [_point(99, "books")], shard_key_selector=other, wait=True)
    request = QueryRequest(query=VECTOR, filter=_filter("books"), limit=10)
    found = client.query_batch_points(collection, [request])
    assert request.shard_key is None
    assert _ids(found[0]) == {1}


def test_upload_points_routes_each_batch(client, collection):
    _create(client, collection)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.upload_points(
            collection,
            [_point(1, "books"), _point(2, "toys")],
            batch_size=1,
            parallel=2,
            wait=True,
        )
    assert any(issubclass(item.category, ShardingWarning) for item in caught)
    assert client.count(collection, count_filter=_filter("books")).count == 1
    assert client.count(collection, count_filter=_filter("toys")).count == 1


@needs_insert_only
def test_migrate_with_zero_grace_keeps_every_point(client, collection):
    _create(client, collection, n_slots=3)
    client.upsert(collection, [_point(i, "books") for i in range(6)], wait=True)
    current = set(client.sharding.get_map(collection).features["books"].shard_keys)
    target = next(slot for slot in ("s0", "s1", "s2") if slot not in current)
    moved = client.sharding.migrate(collection, "books", target, grace_seconds=0)
    assert moved.features["books"].state == FeatureState.stable
    assert moved.features["books"].shard_keys == [target]
    assert moved.features["books"].write_to is None
    total = sum(_slot_count(client, collection, slot, _filter("books")) for slot in moved.slots)
    assert total == 6
    assert client.count(collection, count_filter=_filter("books")).count == 6


@needs_insert_only
def test_filtered_reads_during_migrate_hit_one_placement(client, collection, tmp_path: Path):
    _create(client, collection, n_slots=3)
    client.upsert(collection, [_point(i, "books") for i in range(6)], wait=True)
    current = set(client.sharding.get_map(collection).features["books"].shard_keys)
    target = next(slot for slot in ("s0", "s1", "s2") if slot not in current)
    reader = _open(tmp_path, map_ttl_seconds=0)
    seen: list[int] = []

    def pause(_seconds: int) -> None:
        reader.query_points(collection, VECTOR, query_filter=_filter("books"), limit=10)
        counted = reader.count(collection, count_filter=_filter("books"))
        assert counted.count == 6
        seen.append(counted.count)

    migrator = _open(tmp_path, sleep_fn=pause)
    try:
        moved = migrator.sharding.migrate(collection, "books", target, grace_seconds=0)
        assert seen
        assert reader.count(collection, count_filter=_filter("books")).count == 6
        assert moved.features["books"].state == FeatureState.stable
        assert moved.features["books"].shard_keys == [target]
    finally:
        reader.close()
        migrator.close()


@needs_insert_only
def test_split_spreads_points_with_no_duplicates(client, collection):
    _create(client, collection, n_slots=4)
    client.upsert(collection, [_point(i, "books") for i in range(12)], wait=True)
    doc = client.sharding.split(collection, "books", 3, grace_seconds=0)
    assert doc.features["books"].state == FeatureState.stable
    targets = doc.features["books"].shard_keys
    assert len(targets) == 3
    expected: dict[str, int] = {}
    for point_id in range(12):
        slot = pick_slot(targets, point_id)
        expected[slot] = expected.get(slot, 0) + 1
    for slot in doc.slots:
        assert _slot_count(client, collection, slot, _filter("books")) == expected.get(slot, 0)
    assert len(expected) >= 2


def test_alias_rewrites_old_points_and_drop_alias_removes_the_old_id(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books"), _point(2, "books")], wait=True)
    changed = client.sharding.alias_feature(collection, "books", "libros", grace_seconds=0)
    assert changed == 2
    client.upsert(collection, [_point(3, "libros")], wait=True)
    libros = client.query_points(collection, VECTOR, query_filter=_filter("libros"), limit=10)
    assert _ids(libros) == {1, 2, 3}
    payloads = {point.id: point.payload["feature"] for point in libros.points}
    assert payloads[1] == "libros"
    assert payloads[2] == "libros"
    books = client.query_points(collection, VECTOR, query_filter=_filter("books"), limit=10)
    assert _ids(books) == set()
    client.upsert(collection, [_point(4, "books")], wait=True)
    late = client.query_points(collection, VECTOR, query_filter=_filter("books"), limit=10)
    assert _ids(late) == {4}
    dropped = client.sharding.drop_alias(collection, "books")
    assert dropped == 1
    after = client.query_points(collection, VECTOR, query_filter=_filter("libros"), limit=10)
    assert _ids(after) == {1, 2, 3, 4}
    assert {point.payload["feature"] for point in after.points} == {"libros"}
    doc = client.sharding.get_map(collection)
    assert "books" not in doc.features["libros"].aliases
    gone = client.query_points(collection, VECTOR, query_filter=_filter("books"), limit=10)
    assert _ids(gone) == set()


def test_delete_collection_removes_the_map(client, collection, tmp_path: Path):
    _create(client, collection)
    path = tmp_path / f"{collection}.json"
    assert path.is_file()
    assert client.delete_collection(collection) is True
    assert not path.exists()
    assert client.sharding.get_map(collection) is None


def test_skipped_feature_check_leaves_the_old_copy(client, collection, tmp_path: Path):
    _create(client, collection)
    blind = _open(tmp_path, detect_feature_changes=False)
    blind.upsert(collection, [_point(1, "books")], wait=True)
    old = list(blind.sharding.get_map(collection).features["books"].shard_keys)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blind.upsert(collection, [_point(1, "toys")], wait=True)
    assert not any(issubclass(item.category, FeatureMovedWarning) for item in caught)
    doc = blind.sharding.get_map(collection)
    assert set(old).isdisjoint(doc.features["toys"].shard_keys)
    total = sum(client.count(collection, shard_key_selector=slot).count for slot in doc.slots)
    assert total == 2


def test_broken_map_fans_out_reads_and_denies_writes(client, collection, tmp_path: Path):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    path = tmp_path / f"{collection}.json"
    path.unlink()
    path.mkdir()
    try:
        _broken_map_check(client, collection, tmp_path)
    finally:
        if path.is_dir():
            path.rmdir()


def _broken_map_check(client, collection, tmp_path: Path) -> None:
    fresh = _open(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = fresh.query_points(collection, VECTOR, limit=10)
    assert _ids(found) == {1}
    assert any(issubclass(item.category, StaleMapWarning) for item in caught)
    with pytest.raises(WriteDenied):
        fresh.upsert(collection, [_point(2, "books")])


def test_s3_map_folder_routes_a_collection():
    prefix = f"tests/{uuid.uuid4().hex}/"
    folder = open_map_folder(f"s3://{_BUCKET}/{prefix}")
    routed = ShardedQdrantClient(map_folder=folder, **local_kwargs())
    name = "t" + uuid.uuid4().hex[:12]
    key = prefix + name + ".json"
    try:
        _create(routed, name, n_slots=2)
        routed.upsert(name, [_point(1, "books")], wait=True)
        assert routed.count(name, count_filter=_filter("books")).count == 1
        assert routed.sharding.get_map(name).features["books"].shard_keys
        routed.delete_collection(name)
        with pytest.raises(MapNotFound):
            folder.store_for(name).load()
    finally:
        if routed.collection_exists(name):
            routed.delete_collection(name)
        _s3_client().delete_object(Bucket=_BUCKET, Key=key)


@needs_insert_only
def test_copy_keeps_a_concurrent_update_and_drops_a_concurrent_delete(
    client, collection, tmp_path: Path
):
    _create(client, collection, n_slots=2)
    client.upsert(
        collection,
        [
            PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"}),
            PointStruct(id=2, vector=VECTOR, payload={"feature": "books", "note": "old"}),
        ],
        wait=True,
    )
    doc = client.sharding.get_map(collection)
    current = doc.features["books"].shard_keys[0]
    target = next(slot for slot in doc.slots if slot != current)
    client.sharding.start_move(collection, "books", [target], grace_seconds=0)
    writer = _open(tmp_path)
    started = threading.Event()
    release = threading.Event()
    original = QdrantVectorStore.scroll

    def scroll(self, collection_name, slot, field, values, batch):
        first = True
        for chunk in original(self, collection_name, slot, field, values, batch):
            if first:
                first = False
                ids = {point.id for point in chunk}
                if ids != {1, 2}:
                    raise AssertionError(f"scroll chunk ids {ids}")
                started.set()
                if not release.wait(30):
                    raise TimeoutError("copy hook was not released")
            yield chunk

    QdrantVectorStore.scroll = scroll
    errors: list[BaseException] = []

    def run_copy() -> None:
        try:
            client.sharding.copy_feature(collection, "books", batch=10)
        except BaseException as exc:
            errors.append(exc)
        finally:
            started.set()

    worker = threading.Thread(target=run_copy)
    worker.start()
    try:
        assert started.wait(30)
        if errors:
            raise errors[0]
        writer.upsert(
            collection,
            [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "new"})],
            wait=True,
        )
        writer.delete(
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
        worker.join(60)
    finally:
        QdrantVectorStore.scroll = original
        release.set()
        if worker.is_alive():
            worker.join(10)
    assert not worker.is_alive()
    assert errors == []
    client.sharding.open_dual(collection, "books", grace_seconds=0)
    client.sharding.cutover(collection, "books", grace_seconds=0)
    client.sharding.finish_move(collection, "books")
    found = client.retrieve(collection, [1])
    assert found[0].payload["note"] == "new"
    assert client.retrieve(collection, [2]) == []


def test_unknown_filter_reloads_and_routes_when_another_client_added_it(
    client, collection, tmp_path: Path
):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    client.query_points(collection, VECTOR, query_filter=_filter("books"), limit=10)
    other = _open(tmp_path)
    other.upsert(collection, [_point(2, "toys")], wait=True)
    doc = other.sharding.get_map(collection)
    toy_slots = set(doc.features["toys"].shard_keys)
    decoy = next(slot for slot in doc.slots if slot not in toy_slots)
    other.upsert(collection, [_point(99, "toys")], shard_key_selector=decoy, wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = client.query_points(collection, VECTOR, query_filter=_filter("toys"), limit=10)
    assert _ids(found) == {2}
    assert not any(issubclass(item.category, FanOutWarning) for item in caught)


def test_unknown_filter_fans_out_when_the_value_stays_unknown(client, collection):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    client.query_points(collection, VECTOR, query_filter=_filter("books"), limit=10)
    slot = client.sharding.get_map(collection).slots[0]
    client.upsert(collection, [_point(7, "ghost")], shard_key_selector=slot, wait=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = client.query_points(collection, VECTOR, query_filter=_filter("ghost"), limit=10)
    assert _ids(found) == {7}
    assert any(issubclass(item.category, FanOutWarning) for item in caught)


@needs_insert_only
def test_feature_filter_mutations_hit_both_copies_in_dual_and_cutover(client, collection):
    client.sharding.create_collection(
        collection,
        "feature",
        n_slots=2,
        vectors_config={"text": VectorParams(size=4, distance=Distance.COSINE)},
    )
    books = [
        PointStruct(id=point_id, vector={"text": VECTOR}, payload={"feature": "books"})
        for point_id in (1, 2, 3)
    ]
    client.upsert(collection, books, wait=True)
    doc = client.sharding.get_map(collection)
    old = doc.features["books"].shard_keys[0]
    new = next(slot for slot in doc.slots if slot != old)
    client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    client.sharding.copy_feature(collection, "books")
    client.sharding.open_dual(collection, "books", grace_seconds=0)
    client.set_payload(collection, {"note": "dual"}, _filter("books"), wait=True)
    client.delete_vectors(collection, ["text"], _filter("books"), wait=True)
    assert _notes(client, collection, old) == {"dual"}
    assert _notes(client, collection, new) == {"dual"}
    assert _text_vectors(client, collection, old) == [None, None, None]
    assert _text_vectors(client, collection, new) == [None, None, None]
    client.upsert(
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
    client.sharding.cutover(collection, "books", grace_seconds=0)
    client.set_payload(collection, {"note": "cut"}, _filter("books"), wait=True)
    client.delete_vectors(collection, ["text"], _filter("books"), wait=True)
    assert _notes(client, collection, old) == {"cut"}
    assert _notes(client, collection, new) == {"cut"}
    assert _text_vectors(client, collection, old) == [None, None, None]
    assert _text_vectors(client, collection, new) == [None, None, None]
    client.delete(collection, _filter("books"), wait=True)
    assert _slot_count(client, collection, old, _filter("books")) == 0
    assert _slot_count(client, collection, new, _filter("books")) == 0


def _slot_records(client, collection, slot):
    return client.retrieve(
        collection,
        [1, 2, 3],
        with_payload=True,
        with_vectors=True,
        shard_key_selector=slot,
    )


def _notes(client, collection, slot) -> set:
    return {record.payload.get("note") for record in _slot_records(client, collection, slot)}


def _text_vectors(client, collection, slot) -> list:
    vectors = []
    for record in _slot_records(client, collection, slot):
        vector = record.vector
        if isinstance(vector, dict):
            vectors.append(vector.get("text"))
        else:
            vectors.append(vector)
    return vectors


def _assert_layout(client, name: str, field: str, kind: str) -> None:
    info = client.get_collection(name)
    method = info.config.params.sharding_method
    assert getattr(method, "value", method) == "custom"
    schema = info.payload_schema[field]
    assert getattr(schema.data_type, "value", schema.data_type) == kind
    http = client.http
    api = getattr(http, "cluster_api", None)
    if api is None:
        api = http.distributed_api
    body = api.collection_cluster_info(name)
    cluster = body.result if hasattr(body, "result") else body
    keys = set()
    for shard in list(cluster.local_shards) + list(cluster.remote_shards):
        if shard.shard_key is not None:
            keys.add(str(shard.shard_key))
    assert keys == {"s0", "s1", "s2"}


def _fan(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    fan = [item for item in caught if issubclass(item.category, FanOutWarning)]
    assert fan
    assert Path(fan[0].filename).name == "test_sharded_client.py"
    return result


def _got(client, collection, point_id):
    rows = client.retrieve(collection, [point_id], with_payload=True, with_vectors=True)
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


def _tag(value: str) -> Filter:
    return Filter(must=FieldCondition(key="tag", match=MatchValue(value=value)))


def test_writes_that_omit_the_feature(client, collection):
    _create(client, collection)
    client.upsert(
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
    _fan(lambda: client.delete(collection, [1, 99], wait=True))
    assert _got(client, collection, 1) is None
    assert _got(client, collection, 2) is not None

    _fan(lambda: client.set_payload(collection, {"tag": "blue"}, [3, 4], wait=True))
    assert _got(client, collection, 3).payload == {"feature": "books", "tag": "blue"}
    assert _got(client, collection, 4).payload == {"feature": "toys", "tag": "blue"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(UnexpectedResponse) as exc:
            client.set_payload(collection, {"tag": "blue"}, [98], wait=True)
    assert exc.value.status_code == 404
    assert any(issubclass(item.category, FanOutWarning) for item in caught)
    assert Path(caught[0].filename).name == "test_sharded_client.py"

    _fan(lambda: client.delete_payload(collection, ["tag"], [5], wait=True))
    assert _got(client, collection, 5).payload == {"feature": "books"}

    before = list(_vec(_got(client, collection, 6)))
    _fan(
        lambda: client.update_vectors(
            collection, [PointVectors(id=6, vector=[0.9, 0.1, 0.2, 0.3])], wait=True
        )
    )
    assert list(_vec(_got(client, collection, 6))) != before

    _fan(lambda: client.delete_vectors(collection, [""], [7], wait=True))
    assert not _vec(_got(client, collection, 7))
    assert _got(client, collection, 7).payload["feature"] == "books"

    _fan(lambda: client.delete(collection, _tag("del"), wait=True))
    assert _got(client, collection, 8) is None
    assert _got(client, collection, 9).payload["feature"] == "toys"

    _fan(lambda: client.set_payload(collection, {"note": "yes"}, _tag("pay"), wait=True))
    paid = _got(client, collection, 10).payload
    assert paid["note"] == "yes"
    assert paid["feature"] == "books"

    _fan(lambda: client.delete_payload(collection, ["tag"], _tag("trim"), wait=True))
    assert _got(client, collection, 11).payload == {"feature": "books"}

    _fan(lambda: client.delete_vectors(collection, [""], _tag("novec"), wait=True))
    assert not _vec(_got(client, collection, 12))
    assert _got(client, collection, 12).payload["feature"] == "books"

    client.overwrite_payload(
        collection, {"feature": "books", "tag": "green"}, [13], wait=True
    )
    assert _got(client, collection, 13).payload == {"feature": "books", "tag": "green"}
    with pytest.raises(ValueError, match="overwrite_payload"):
        client.overwrite_payload(collection, {"tag": "x"}, [13])
    with pytest.raises(ValueError, match="clear_payload"):
        client.clear_payload(collection, [2])
    with pytest.raises(UnexpectedResponse) as mixed:
        client.set_payload(collection, {"tag": "mix"}, [2, 97], wait=True)
    assert mixed.value.status_code == 404
    assert _got(client, collection, 2).payload["tag"] == "mix"


def _other_slot(doc):
    current = doc.features["books"].shard_keys[0]
    return next(slot for slot in doc.slots if slot != current)


def _note_on(client, collection, slot, point_id=1):
    rows = client.retrieve(
        collection, [point_id], with_payload=True, shard_key_selector=slot
    )
    assert len(rows) == 1
    return rows[0].payload["note"]


@needs_insert_only
def test_set_payload_by_id_during_copy_reaches_the_copied_point(client, collection):
    _create(client, collection, n_slots=2)
    client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"})],
        wait=True,
    )
    new = _other_slot(client.sharding.get_map(collection))
    client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    client.sharding.copy_feature(collection, "books")
    client.set_payload(collection, {"note": "new"}, [1], wait=True)
    assert _note_on(client, collection, new) == "new"
    client.sharding.open_dual(collection, "books", grace_seconds=0)
    client.sharding.cutover(collection, "books", grace_seconds=0)
    client.sharding.finish_move(collection, "books")
    assert client.retrieve(collection, [1])[0].payload["note"] == "new"


@needs_insert_only
def test_set_payload_by_id_before_the_copy_survives_it(client, collection):
    _create(client, collection, n_slots=2)
    client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"})],
        wait=True,
    )
    new = _other_slot(client.sharding.get_map(collection))
    client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    client.set_payload(collection, {"note": "new"}, [1], wait=True)
    client.sharding.copy_feature(collection, "books")
    assert _note_on(client, collection, new) == "new"


@needs_insert_only
def test_filter_set_payload_before_the_copy_survives_it(client, collection):
    _create(client, collection, n_slots=2)
    client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"})],
        wait=True,
    )
    new = _other_slot(client.sharding.get_map(collection))
    client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    client.set_payload(collection, {"note": "new"}, _filter("books"), wait=True)
    client.sharding.copy_feature(collection, "books")
    assert _note_on(client, collection, new) == "new"


@needs_insert_only
def test_filter_set_payload_during_copy_keeps_the_new_value(
    client, collection, tmp_path: Path
):
    _create(client, collection, n_slots=2)
    client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "old"})],
        wait=True,
    )
    new = _other_slot(client.sharding.get_map(collection))
    client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    writer = _open(tmp_path)
    started = threading.Event()
    release = threading.Event()
    original = QdrantVectorStore.scroll

    def scroll(self, collection_name, slot, field, values, batch):
        first = True
        for chunk in original(self, collection_name, slot, field, values, batch):
            if first:
                first = False
                started.set()
                if not release.wait(30):
                    raise TimeoutError("copy hook was not released")
            yield chunk

    QdrantVectorStore.scroll = scroll
    errors: list[BaseException] = []

    def run_copy() -> None:
        try:
            client.sharding.copy_feature(collection, "books")
        except BaseException as exc:
            errors.append(exc)
        finally:
            started.set()

    worker = threading.Thread(target=run_copy)
    worker.start()
    try:
        assert started.wait(30)
        if errors:
            raise errors[0]
        writer.set_payload(collection, {"note": "new"}, _filter("books"), wait=True)
        release.set()
        worker.join(60)
    finally:
        QdrantVectorStore.scroll = original
        release.set()
        if worker.is_alive():
            worker.join(10)
    assert not worker.is_alive()
    assert errors == []
    assert _note_on(client, collection, new) == "new"


@needs_insert_only
def test_set_payload_by_id_during_cutover_hits_the_new_slot(client, collection):
    _create(client, collection, n_slots=2)
    client.upsert(collection, [_point(1, "books")], wait=True)
    new = _other_slot(client.sharding.get_map(collection))
    client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    client.sharding.copy_feature(collection, "books")
    client.sharding.open_dual(collection, "books", grace_seconds=0)
    client.sharding.cutover(collection, "books", grace_seconds=0)
    client.upsert(
        collection,
        [PointStruct(id=2, vector=VECTOR, payload={"feature": "books"})],
        wait=True,
    )
    client.set_payload(collection, {"note": "late"}, [2], wait=True)
    assert client.retrieve(collection, [2])[0].payload["note"] == "late"


def test_set_payload_by_id_reloads_a_feature_added_by_another_client(
    client, collection, tmp_path: Path
):
    _create(client, collection)
    client.upsert(collection, [_point(1, "books")], wait=True)
    other = _open(tmp_path)
    other.upsert(collection, [_point(2, "toys")], wait=True)
    client.set_payload(collection, {"tag": "yes"}, [2], wait=True)
    got = _got(client, collection, 2)
    assert got.payload["feature"] == "toys"
    assert got.payload["tag"] == "yes"
    other.close()


@needs_insert_only
def test_cutover_repairs_a_copy_that_lost_a_race(client, collection):
    _create(client, collection, n_slots=2)
    client.upsert(
        collection,
        [PointStruct(id=1, vector=VECTOR, payload={"feature": "books", "note": "kept"})],
        wait=True,
    )
    doc = client.sharding.get_map(collection)
    old = doc.features["books"].shard_keys[0]
    new = next(slot for slot in doc.slots if slot != old)
    client.sharding.start_move(collection, "books", [new], grace_seconds=0)
    client.sharding.copy_feature(collection, "books")
    client.sharding.open_dual(collection, "books", grace_seconds=0)
    client.delete(collection, [1], shard_key_selector=new, wait=True)
    client.upsert(
        collection,
        [PointStruct(id=2, vector=VECTOR, payload={"feature": "books", "note": "added"})],
        shard_key_selector=old,
        wait=True,
    )
    result = client.sharding.cutover(collection, "books", grace_seconds=0)
    assert result.copied == 2
    assert result.removed == 0
    rows = client.retrieve(collection, [1, 2], shard_key_selector=new, with_payload=True)
    assert {row.id: row.payload["note"] for row in rows} == {1: "kept", 2: "added"}

