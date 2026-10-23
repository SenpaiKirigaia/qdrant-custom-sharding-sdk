from datetime import datetime, timedelta, timezone

import pytest

from qdrant_custom_sharding.errors import (
    InvalidMove,
    MapNotFound,
    MoveNotReady,
    VersionConflict,
)
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc, Point
from qdrant_custom_sharding.planning import plan_placement
from qdrant_custom_sharding.services import (
    align_actions,
    align_ids,
    add_feature,
    add_features,
    alias_feature,
    apply_stats,
    build_collection_map,
    copy_feature,
    copy_job,
    create_sharded_collection,
    cutover,
    finish_job,
    finish_move,
    group_by_slot,
    reconcile_pairs,
    reconcile_tally,
    migrate,
    open_dual,
    refresh_stats,
    split_targets,
    start_move,
    update_map,
)
from tests.fakes import FakeMapStore, FakeVectorStore

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


class RacingMapStore(FakeMapStore):

    def __init__(self, conflicts: int) -> None:
        super().__init__()
        self.conflicts = conflicts

    def save(self, doc: MapDoc) -> None:
        if self.conflicts:
            self.conflicts -= 1
            current = FakeMapStore.load(self)
            current.version += 1
            FakeMapStore.save(self, current)
        FakeMapStore.save(self, doc)


class ConflictOnce:

    def __init__(self, doc: MapDoc) -> None:
        self.doc = doc
        self._failed = False

    def load(self) -> MapDoc:
        return self.doc

    def save(self, doc: MapDoc) -> None:
        if not self._failed:
            self._failed = True
            raise VersionConflict("race")
        if self.doc.version != doc.version - 1:
            raise VersionConflict(
                f"stored version is {self.doc.version}, got {doc.version}"
            )
        self.doc = doc


class RecordingVectors(FakeVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[object, ...]] = []

    def create_collection(self, collection: str, **kwargs: object) -> None:
        self.events.append(("collection", collection, dict(kwargs)))
        super().create_collection(collection, **kwargs)

    def create_slot(self, collection: str, slot: str) -> None:
        self.events.append(("slot", slot))
        super().create_slot(collection, slot)

    def create_feature_index(
        self, collection: str, field: str, feature_type: str
    ) -> None:
        self.events.append(("index", collection, field, feature_type))
        super().create_feature_index(collection, field, feature_type)


def _map() -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0", "s1", "s2"],
        features={},
        slot_load={"s0": 0, "s1": 0, "s2": 0},
    )


def _ready(
    n_slots: int = 3,
    feature_field: str = "feature",
    feature_type: str = "keyword",
) -> tuple[FakeMapStore, FakeVectorStore]:
    store = FakeMapStore()
    vectors = FakeVectorStore()
    create_sharded_collection(
        store,
        vectors,
        "items",
        feature_field,
        feature_type,
        n_slots=n_slots,
        vectors_config={"size": 2},
    )
    return store, vectors


def _put(
    vectors: FakeVectorStore,
    slot: str,
    point_id: int,
    field: str,
    value: int | str,
) -> None:
    vectors.upsert(
        "items",
        slot,
        [Point(id=point_id, vector=[1.0, 0.0], payload={field: value})],
    )


def _ids(
    vectors: FakeVectorStore,
    slot: str,
    field: str,
    values: list[int | str],
) -> list[int | str]:
    found: list[int | str] = []
    for batch in vectors.scroll("items", slot, field, values, 10):
        found.extend(point.id for point in batch)
    return found


def test_update_map_deep_copies_before_a_conflict():
    original = _map()
    store = ConflictOnce(original)
    update_map(store, lambda doc: doc.add_feature("books"))
    assert original.version == 1
    assert "books" not in original.features
    assert store.doc.version == 2
    assert store.doc.features["books"].shard_keys == ["s0"]


def test_update_map_retries_then_gives_up():
    store = RacingMapStore(0)
    store.save(_map())
    store.conflicts = 1
    update_map(store, lambda doc: doc.add_feature("toys"))
    loaded = store.load()
    assert loaded.version == 3
    assert "toys" in loaded.features

    stubborn = RacingMapStore(0)
    stubborn.save(_map())
    stubborn.conflicts = 5
    with pytest.raises(VersionConflict):
        update_map(stubborn, lambda doc: doc.add_feature("toys"), retries=2)
    assert "toys" not in stubborn.load().features


def test_create_sharded_collection_creates_slots_an_index_and_version_one():
    store = FakeMapStore()
    vectors = RecordingVectors()
    doc = create_sharded_collection(
        store,
        vectors,
        "items",
        "subject_id",
        "integer",
        n_slots=3,
        vectors_config={"size": 4, "distance": "Dot"},
        replication_factor=2,
    )
    assert doc.version == 1
    assert doc.collection == "items"
    assert doc.feature_field == "subject_id"
    assert doc.feature_type == "integer"
    assert doc.slots == ["s0", "s1", "s2"]
    assert doc.features == {}
    assert doc.slot_load == {"s0": 0, "s1": 0, "s2": 0}
    assert vectors.events == [
        (
            "collection",
            "items",
            {
                "vectors_config": {"size": 4, "distance": "Dot"},
                "replication_factor": 2,
            },
        ),
        ("slot", "s0"),
        ("slot", "s1"),
        ("slot", "s2"),
        ("index", "items", "subject_id", "integer"),
    ]
    assert store.load() == doc


def test_create_sharded_collection_uses_plan_placement_when_counts_are_given():
    counts = {"big": 5, "small": 1}
    slots = ["s0", "s1"]
    store = FakeMapStore()
    vectors = FakeVectorStore()
    doc = create_sharded_collection(
        store,
        vectors,
        "items",
        "feature",
        "keyword",
        n_slots=2,
        feature_counts=counts,
        vectors_config={"size": 2},
    )
    assert {
        name: info.shard_keys for name, info in doc.features.items()
    } == plan_placement(counts, slots)
    assert doc.version == 1
    assert doc.features["big"].points_count == 5
    assert doc.features["small"].points_count == 1
    assert sum(doc.slot_load.values()) == sum(counts.values())


def test_add_feature_picks_the_least_loaded_slot_and_is_idempotent():
    store, vectors = _ready(3)
    assert add_feature(store, "books").shard_keys == ["s0"]
    assert add_feature(store, "books").shard_keys == ["s0"]
    loaded = store.load()
    loaded.features["kept"] = FeatureInfo(shard_keys=["s0"])
    loaded.slot_load["s0"] = 5
    loaded.version += 1
    store.save(loaded)
    assert add_feature(store, "toys").shard_keys == ["s1"]
    alias_feature(store, vectors, "toys", "juguetes", lambda _seconds: None, 0)
    assert add_feature(store, "toys").shard_keys == ["s1"]
    assert "juguetes" in store.load().features
    assert "toys" not in store.load().features


def test_create_sharded_collection_rejects_a_bad_feature_type_before_io():
    store = FakeMapStore()
    vectors = RecordingVectors()
    with pytest.raises(ValueError, match="feature_type"):
        create_sharded_collection(
            store,
            vectors,
            "items",
            "feature",
            "float",
            vectors_config={"size": 2},
        )
    assert vectors.events == []
    with pytest.raises(MapNotFound):
        store.load()


def test_add_features_places_every_value_in_one_update():
    store, _vectors = _ready(3)
    doc = add_features(store, ["books", "toys", "books"])
    assert list(doc.features) == ["books", "toys"]
    assert doc.features["books"].shard_keys == ["s0"]
    assert doc.features["toys"].shard_keys == ["s1"]
    assert doc.version == 2


def test_add_feature_accepts_an_int():
    store, _vectors = _ready(feature_type="integer", feature_field="subject_id")
    assert add_feature(store, 7).shard_keys == ["s0"]
    assert add_feature(store, "7") is not None
    assert list(store.load().features) == ["7"]


def test_copy_feature_rejects_a_stable_feature():
    store, vectors = _ready(2)
    add_feature(store, "books")
    with pytest.raises(MoveNotReady):
        copy_feature(store, vectors, "books", NOW)
    assert store.load().features["books"].copied is False


def test_align_actions_insert_present_ids_and_retry_presence_changes():
    assert align_actions([1, 2, 3], {1, 3}).insert_ids == [1, 3]
    assert align_actions([1, 2, 3], {1, 3}).delete_ids == [2]
    second = align_actions([1, 2, 3], {1, 3}, {1, 2})
    assert second.insert_ids == [2]
    assert second.delete_ids == [3]
    same = align_actions([1, 2], {1}, {1})
    assert same.insert_ids == []
    assert same.delete_ids == []


def test_reconcile_pairs_are_the_symmetric_difference():
    pairs = reconcile_pairs({1, 2}, {2, 3}, ["s0"], ["s1"])
    assert [(pair.old_slot, pair.new_slot, pair.ids) for pair in pairs] == [
        ("s0", "s1", [1, 3])
    ]
    assert reconcile_pairs({1}, {1}, ["s0"], ["s1"]) == []
    assert reconcile_tally({1}, {3}, [1, 3], {1}) == (1, 1)
    assert reconcile_tally({1}, {3}, [1, 3], set()) == (0, 1)


def test_align_drops_a_point_deleted_after_the_first_read():
    _store, vectors = _ready(2)
    vectors.upsert(
        "items",
        "s0",
        [Point(id=1, vector=[1.0], payload={"feature": "books"})],
    )
    real = vectors.read_ids
    calls = {"n": 0}

    def read_ids(collection, slot, ids):
        calls["n"] += 1
        points = real(collection, slot, ids)
        if calls["n"] == 1:
            vectors.delete_ids(collection, slot, list(ids))
        return points

    vectors.read_ids = read_ids
    assert align_ids(vectors, "items", "s0", "s1", [1]) == set()
    assert _ids(vectors, "s1", "feature", ["books"]) == []


def test_align_inserts_a_point_that_appears_after_the_first_read():
    _store, vectors = _ready(2)
    vectors.upsert(
        "items",
        "s0",
        [Point(id=1, vector=[1.0], payload={"feature": "books", "note": "new"})],
    )
    real = vectors.read_ids
    calls = {"n": 0}

    def read_ids(collection, slot, ids):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        return real(collection, slot, ids)

    vectors.read_ids = read_ids
    assert align_ids(vectors, "items", "s0", "s1", [1]) == {1}
    found = []
    for batch in vectors.scroll("items", "s1", "feature", ["books"], 10):
        found.extend(batch)
    assert found[0].payload["note"] == "new"


def test_cutover_repairs_ids_that_exist_on_only_one_side():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    _put(vectors, "s1", 2, "feature", "books")
    start_move(store, "books", ["s1"], NOW, 0)
    copy_feature(store, vectors, "books", NOW)
    vectors.delete_ids("items", "s1", [1])
    open_dual(store, "books", NOW, 0)
    result = cutover(store, vectors, "books", NOW, 0)
    assert result.old_slots == ["s0"]
    assert result.copied == 1
    assert result.removed == 1
    assert _ids(vectors, "s1", "feature", ["books"]) == [1]
    assert _ids(vectors, "s0", "feature", ["books"]) == [1]


def test_copy_keeps_a_newer_payload_and_drops_a_deleted_point():
    store, vectors = _ready(2)
    add_feature(store, "books")
    vectors.upsert(
        "items",
        "s0",
        [
            Point(id=1, vector=[1.0, 0.0], payload={"feature": "books", "note": "old"}),
            Point(id=2, vector=[1.0, 0.0], payload={"feature": "books", "note": "old"}),
        ],
    )
    start_move(store, "books", ["s1"], NOW, 0)
    vectors.upsert(
        "items",
        "s1",
        [Point(id=1, vector=[1.0, 0.0], payload={"feature": "books", "note": "new"})],
    )
    original = vectors.scroll

    def scroll(collection, slot, field, values, batch):
        for chunk in original(collection, slot, field, values, batch):
            vectors.delete_ids(collection, slot, [2])
            yield chunk

    vectors.scroll = scroll
    assert copy_feature(store, vectors, "books", NOW) == 2
    found = []
    for batch in original("items", "s1", "feature", ["books"], 10):
        found.extend(batch)
    assert {point.id: point.payload["note"] for point in found} == {1: "new"}


def test_copy_feature_copies_aliases_and_integer_values():
    store, vectors = _ready(2)
    add_feature(store, "books")
    alias_feature(store, vectors, "books", "libros", lambda _seconds: None, 0)
    _put(vectors, "s0", 1, "feature", "books")
    _put(vectors, "s0", 2, "feature", "libros")
    start_move(store, "libros", ["s1"], NOW, 0)
    assert copy_feature(store, vectors, "books", NOW) == 2
    assert _ids(vectors, "s1", "feature", ["books", "libros"]) == [1, 2]

    numbers, number_vectors = _ready(
        2, feature_field="subject_id", feature_type="integer"
    )
    add_feature(numbers, 7)
    alias_feature(numbers, number_vectors, "7", "8", lambda _seconds: None, 0)
    _put(number_vectors, "s0", 3, "subject_id", 7)
    start_move(numbers, 8, ["s1"], NOW, 0)
    assert copy_feature(numbers, number_vectors, 7, NOW) == 1
    assert _ids(number_vectors, "s1", "subject_id", [7, 8]) == [3]


def test_move_steps_wait_for_grace_and_cutover_keeps_shard_keys():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    _put(vectors, "s1", 1, "feature", "books")
    start_move(store, "books", ["s1"], NOW, 10)
    info = store.load().features["books"]
    assert info.state == FeatureState.copying
    assert info.grace_until == NOW + timedelta(seconds=10)
    with pytest.raises(MoveNotReady):
        open_dual(store, "books", NOW, 10)
    ready_copy = NOW + timedelta(seconds=10)
    with pytest.raises(MoveNotReady):
        copy_feature(store, vectors, "books", NOW)
    with pytest.raises(MoveNotReady):
        open_dual(store, "books", ready_copy, 10)
    assert copy_feature(store, vectors, "books", ready_copy) == 1
    assert store.load().features["books"].copied is True
    open_dual(store, "books", ready_copy, 10)
    info = store.load().features["books"]
    assert info.state == FeatureState.dual
    assert info.shard_keys == ["s0"]
    with pytest.raises(MoveNotReady):
        cutover(store, vectors, "books", NOW + timedelta(seconds=10), 10)
    ready = NOW + timedelta(seconds=20)
    assert cutover(store, vectors, "books", ready, 10).old_slots == ["s0"]
    info = store.load().features["books"]
    assert info.state == FeatureState.cutover
    assert info.shard_keys == ["s0"]
    assert info.write_to == ["s1"]
    with pytest.raises(MoveNotReady):
        finish_move(store, vectors, "books", ready)
    assert _ids(vectors, "s0", "feature", ["books"]) == [1]
    assert store.load().features["books"].state == FeatureState.cutover
    finish_move(store, vectors, "books", ready + timedelta(seconds=10))
    info = store.load().features["books"]
    assert info.state == FeatureState.stable
    assert info.shard_keys == ["s1"]
    assert info.write_to is None
    assert info.grace_until is None
    assert info.copied is False
    assert _ids(vectors, "s0", "feature", ["books"]) == []
    assert _ids(vectors, "s1", "feature", ["books"]) == [1]


def test_finish_move_deletes_old_slots_then_updates_the_map():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    start_move(store, "books", ["s1"], NOW, 0)
    copy_feature(store, vectors, "books", NOW)
    open_dual(store, "books", NOW, 0)
    cutover(store, vectors, "books", NOW, 0)
    events: list[tuple[object, ...]] = []
    original_delete = vectors.delete_feature
    original_save = store.save

    def delete_feature(
        collection: str, slot: str, field: str, values: list[int | str]
    ) -> None:
        events.append(("delete", slot, store.load().features["books"].state))
        original_delete(collection, slot, field, values)

    def save(doc: MapDoc) -> None:
        events.append(("save", doc.features["books"].state))
        original_save(doc)

    vectors.delete_feature = delete_feature
    store.save = save
    finish_move(store, vectors, "books", NOW)
    assert events == [
        ("delete", "s0", FeatureState.cutover),
        ("save", FeatureState.stable),
    ]


def test_finish_move_rejects_the_wrong_state_before_deleting():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")

    def delete_feature(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("delete must not run")

    vectors.delete_feature = delete_feature
    with pytest.raises(InvalidMove):
        finish_move(store, vectors, "books", NOW)
    assert _ids(vectors, "s0", "feature", ["books"]) == [1]


def test_migrate_sleeps_before_copy_switch_and_delete():
    store, vectors = _ready(3)
    add_feature(store, "books")
    add_feature(store, "toys")
    _put(vectors, "s0", 1, "feature", "books")
    _put(vectors, "s0", 2, "feature", "books")
    _put(vectors, "s0", 3, "feature", "books")
    _put(vectors, "s1", 9, "feature", "toys")
    clock = [NOW]
    slept: list[int] = []
    states: list[FeatureState] = []

    def now_fn() -> datetime:
        return clock[0]

    def sleep_fn(seconds: int) -> None:
        slept.append(seconds)
        states.append(store.load().features["books"].state)
        clock[0] = clock[0] + timedelta(seconds=seconds)

    migrate(store, vectors, "books", ["s2"], now_fn, sleep_fn, 60)
    assert slept == [60, 60, 60]
    assert states == [
        FeatureState.copying,
        FeatureState.dual,
        FeatureState.cutover,
    ]
    moved = store.load().features["books"]
    assert moved.state == FeatureState.stable
    assert moved.shard_keys == ["s2"]
    assert moved.write_to is None
    assert moved.grace_until is None
    assert _ids(vectors, "s0", "feature", ["books"]) == []
    assert _ids(vectors, "s2", "feature", ["books"]) == [1, 2, 3]
    assert _ids(vectors, "s1", "feature", ["toys"]) == [9]


def test_migrate_with_zero_grace_still_sleeps_three_times():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    slept: list[int] = []
    migrate(store, vectors, "books", ["s1"], lambda: NOW, slept.append, 0)
    assert slept == [0, 0, 0]
    assert store.load().features["books"].shard_keys == ["s1"]
    assert _ids(vectors, "s1", "feature", ["books"]) == [1]


def test_a_point_written_to_both_slots_while_copying_survives():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    start_move(store, "books", ["s1"], NOW, 0)
    _put(vectors, "s0", 7, "feature", "books")
    _put(vectors, "s1", 7, "feature", "books")
    assert copy_feature(store, vectors, "books", NOW) == 2
    open_dual(store, "books", NOW, 0)
    cutover(store, vectors, "books", NOW, 0)
    finish_move(store, vectors, "books", NOW)
    assert _ids(vectors, "s0", "feature", ["books"]) == []
    assert _ids(vectors, "s1", "feature", ["books"]) == [1, 7]


def test_alias_feature_accepts_an_int():
    store, vectors = _ready(feature_type="integer", feature_field="subject_id")
    add_feature(store, 7)
    alias_feature(store, vectors, 7, 8, lambda _seconds: None, 0)
    assert list(store.load().features) == ["8"]
    assert store.load().features["8"].aliases == ["7"]


def test_split_targets_skips_busy_slots_and_the_feature_slot():
    store, _vectors = _ready(4)
    add_feature(store, "books")
    doc = store.load()
    doc.features["books"] = FeatureInfo(shard_keys=["s1"])
    doc.slot_load = {"s0": 100, "s1": 0, "s2": 50, "s3": 40}
    doc.features["other"] = FeatureInfo(
        shard_keys=["s2"],
        state=FeatureState.copying,
        write_to=["s0"],
    )
    assert split_targets(doc, "books", 1) == ["s3"]


def test_split_spreads_one_feature_and_every_point_stays_on_the_new_slots():
    store, vectors = _ready(4)
    add_feature(store, "books")
    for point_id in range(8):
        _put(vectors, "s0", point_id, "feature", "books")
    targets = split_targets(store.load(), "books", 2)
    assert len(targets) == 2
    assert targets != store.load().features["books"].shard_keys
    migrate(store, vectors, "books", targets, lambda: NOW, lambda _seconds: None, 0)
    doc = store.load()
    assert doc.features["books"].shard_keys == targets
    assert doc.features["books"].state == FeatureState.stable
    moved: list[int | str] = []
    for slot in targets:
        moved.extend(_ids(vectors, slot, "feature", ["books"]))
    assert sorted(moved) == list(range(8))
    for slot in doc.slots:
        if slot not in targets:
            assert _ids(vectors, slot, "feature", ["books"]) == []


def test_refresh_stats_counts_aliases_and_integer_payloads():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    _put(vectors, "s0", 2, "feature", "books")
    alias_feature(store, vectors, "books", "libros", lambda _seconds: None, 0)
    _put(vectors, "s0", 3, "feature", "libros")
    _put(vectors, "s0", 4, "feature", "books")
    doc = refresh_stats(store, vectors)
    assert doc.features["libros"].points_count == 4
    assert doc.slot_load == {"s0": 4, "s1": 0}

    numbers, number_vectors = _ready(
        1, feature_field="subject_id", feature_type="integer"
    )
    add_feature(numbers, 7)
    _put(number_vectors, "s0", 4, "subject_id", 7)
    counted = refresh_stats(numbers, number_vectors)
    assert counted.features["7"].points_count == 1
    assert counted.slot_load == {"s0": 1}


def test_build_collection_map_names_slots_and_starts_at_version_one():
    doc = build_collection_map("items", "feature", "keyword", 2, None)
    assert doc.slots == ["s0", "s1"]
    assert doc.version == 1
    assert doc.features == {}
    assert doc.slot_load == {"s0": 0, "s1": 0}


def test_copy_job_and_group_by_slot():
    doc = _map()
    doc.add_feature("books")
    with pytest.raises(InvalidMove, match="write_to"):
        copy_job(doc, "books")
    doc.features["books"].write_to = ["s2"]
    job = copy_job(doc, "books")
    assert job.collection == "items"
    assert job.targets == ["s2"]
    assert job.values == ["books"]
    grouped = group_by_slot(
        [Point(id=1, vector=[1.0], payload={}), Point(id=2, vector=[1.0], payload={})],
        ["s0", "s1"],
    )
    assert sum(len(points) for points in grouped.values()) == 2


def test_finish_job_lists_slots_outside_the_target_and_leaves_the_doc():
    doc = _map()
    doc.add_feature("books")
    info = doc.features["books"]
    info.state = FeatureState.cutover
    info.shard_keys = ["s0", "s1"]
    info.write_to = ["s1"]
    info.grace_until = NOW - timedelta(seconds=1)
    job = finish_job(doc, "books", NOW)
    assert job.old_slots == ["s0"]
    assert job.values == ["books"]
    assert doc.features["books"].state == FeatureState.cutover
    assert doc.features["books"].write_to == ["s1"]


def test_apply_stats_sets_counts_on_the_doc():
    doc = _map()
    doc.add_feature("books")
    apply_stats(doc, {"s0": {"books": 4}, "s1": {}})
    assert doc.features["books"].points_count == 4
    assert doc.slot_load["s0"] == 4


def test_alias_feature_rewrites_alias_payloads_after_grace():
    store, vectors = _ready(3)
    add_feature(store, "books")
    doc = store.load()
    doc.features["books"].aliases.append("1")
    doc.features["books"].shard_keys = ["s0", "s1"]
    doc.version += 1
    store.save(doc)
    _put(vectors, "s0", 1, "feature", "books")
    vectors.upsert(
        "items",
        "s0",
        [Point(id=2, vector=[1.0, 0.0], payload={"feature": "1", "color": "red"})],
    )
    _put(vectors, "s1", 3, "feature", "books")
    _put(vectors, "s2", 4, "feature", "books")
    _put(vectors, "s0", 5, "feature", "toys")
    seen: list[int] = []

    def sleep(seconds: int) -> None:
        seen.append(seconds)
        stored = vectors.read_ids("items", "s0", [1])[0]
        assert stored.payload["feature"] == "books"

    changed = alias_feature(store, vectors, "books", "libros", sleep, 9)
    assert seen == [9]
    assert changed == 3
    assert vectors.read_ids("items", "s0", [1])[0].payload == {"feature": "libros"}
    kept = vectors.read_ids("items", "s0", [2])[0].payload
    assert kept == {"feature": "libros", "color": "red"}
    assert vectors.read_ids("items", "s1", [3])[0].payload["feature"] == "libros"
    assert vectors.read_ids("items", "s2", [4])[0].payload["feature"] == "books"
    assert vectors.read_ids("items", "s0", [5])[0].payload["feature"] == "toys"
    saved = store.load()
    assert list(saved.features) == ["libros"]
    assert saved.features["libros"].aliases == ["1", "books"]


def test_alias_feature_types_an_integer_payload():
    store, vectors = _ready(1, feature_field="subject_id", feature_type="integer")
    add_feature(store, 7)
    _put(vectors, "s0", 1, "subject_id", 7)
    changed = alias_feature(store, vectors, 7, 8, lambda _seconds: None, 0)
    assert changed == 1
    assert vectors.read_ids("items", "s0", [1])[0].payload["subject_id"] == 8
    assert store.load().features["8"].aliases == ["7"]


def test_alias_feature_finishes_a_rename_saved_before_the_rewrite():
    store, vectors = _ready(1)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    doc = store.load()
    doc.alias_feature("books", "libros")
    doc.version += 1
    store.save(doc)
    slept: list[int] = []
    changed = alias_feature(store, vectors, "books", "libros", slept.append, 0)
    assert slept == [0]
    assert changed == 1
    assert vectors.read_ids("items", "s0", [1])[0].payload["feature"] == "libros"
    assert store.load().features["libros"].aliases == ["books"]


def test_alias_feature_with_zero_grace_still_sleeps():
    store, vectors = _ready(1)
    add_feature(store, "books")
    slept: list[int] = []
    alias_feature(store, vectors, "books", "libros", slept.append, 0)
    assert slept == [0]


def test_alias_feature_does_not_sleep_when_a_move_is_running():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    start_move(store, "books", ["s1"], NOW, 0)
    slept: list[int] = []
    with pytest.raises(InvalidMove, match="stable"):
        alias_feature(store, vectors, "books", "libros", slept.append, 5)
    assert slept == []
    assert vectors.read_ids("items", "s0", [1])[0].payload["feature"] == "books"
    assert "books" in store.load().features


def test_alias_feature_stops_when_the_feature_moves_during_grace():
    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")

    def sleep(_seconds: int) -> None:
        start_move(store, "libros", ["s1"], NOW, 0)

    with pytest.raises(InvalidMove, match="stable"):
        alias_feature(store, vectors, "books", "libros", sleep, 3)
    assert vectors.read_ids("items", "s0", [1])[0].payload["feature"] == "books"
    assert store.load().features["libros"].state == FeatureState.copying


def test_drop_alias_rewrites_a_late_point_and_removes_the_alias():
    from qdrant_custom_sharding.services import drop_alias

    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    alias_feature(store, vectors, "books", "libros", lambda _seconds: None, 0)
    _put(vectors, "s0", 2, "feature", "books")
    _put(vectors, "s0", 3, "feature", "toys")
    changed = drop_alias(store, vectors, "books")
    assert changed == 1
    assert vectors.read_ids("items", "s0", [1])[0].payload["feature"] == "libros"
    assert vectors.read_ids("items", "s0", [2])[0].payload["feature"] == "libros"
    assert vectors.read_ids("items", "s0", [3])[0].payload["feature"] == "toys"
    saved = store.load()
    assert saved.features["libros"].aliases == []
    assert saved.find("books") is None


def test_drop_alias_rejects_an_unknown_alias_and_a_running_move():
    from qdrant_custom_sharding.services import drop_alias

    store, vectors = _ready(2)
    add_feature(store, "books")
    _put(vectors, "s0", 1, "feature", "books")
    with pytest.raises(InvalidMove, match="unknown alias"):
        drop_alias(store, vectors, "books")
    alias_feature(store, vectors, "books", "libros", lambda _seconds: None, 0)
    _put(vectors, "s0", 2, "feature", "books")
    with pytest.raises(InvalidMove, match="unknown alias"):
        drop_alias(store, vectors, "libros")
    start_move(store, "libros", ["s1"], NOW, 0)
    with pytest.raises(InvalidMove, match="stable"):
        drop_alias(store, vectors, "books")
    assert vectors.read_ids("items", "s0", [2])[0].payload["feature"] == "books"
    assert store.load().features["libros"].aliases == ["books"]
