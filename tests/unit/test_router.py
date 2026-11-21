import pytest
from qdrant_client.models import (
    Batch,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
)

from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc

from qdrant_custom_sharding.router import (
    FilterCopy,
    chunks,
    filter_copy_targets,
    is_feature_move,
    iter_upload_points,
    moved_points,
    mutation_slots,
    needs_reload,
    plan_writes,
    read_selector,
    slots_to_delete,
)
from qdrant_custom_sharding.routing import pick_slot


def _doc() -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0", "s1", "s2"],
        features={
            "books": FeatureInfo(shard_keys=["s0", "s1"], aliases=["1"]),
            "toys": FeatureInfo(shard_keys=["s2"]),
        },
        slot_load={"s0": 0, "s1": 0, "s2": 0},
    )


def _filter(value: object) -> Filter:
    return Filter(must=FieldCondition(key="feature", match=MatchValue(value=value)))


def test_read_selector_fans_out_without_a_known_feature():
    doc = _doc()
    assert read_selector(doc, None) is None
    assert read_selector(doc, Filter()) is None
    other = Filter(must=FieldCondition(key="color", match=MatchValue(value="red")))
    assert read_selector(doc, other) is None
    unknown = Filter(
        must=FieldCondition(key="feature", match=MatchAny(any=["ghost"]))
    )
    assert read_selector(doc, unknown) is None


def test_read_selector_unions_known_features_and_keeps_aliases():
    doc = _doc()
    assert read_selector(doc, _filter("books")) == ["s0", "s1"]
    assert read_selector(doc, _filter("1")) == ["s0", "s1"]
    both = Filter(
        must=FieldCondition(key="feature", match=MatchAny(any=["books", "toys"]))
    )
    assert read_selector(doc, both) == ["s0", "s1", "s2"]
    mixed = Filter(
        must=FieldCondition(key="feature", match=MatchAny(any=["books", "ghost", "toys"]))
    )
    assert read_selector(doc, mixed) is None


def test_read_selector_uses_the_new_placement_while_dual():
    doc = _doc()
    doc.features["books"].state = FeatureState.dual
    doc.features["books"].write_to = ["s2"]
    assert read_selector(doc, _filter("books")) == ["s2"]


def test_plan_writes_groups_known_points_and_returns_unknown_values():
    doc = _doc()
    books = PointStruct(id=1, vector=[0.1], payload={"feature": "books", "color": "red"})
    alias = PointStruct(id=2, vector=[0.2], payload={"feature": "1"})
    unknown = PointStruct(id=3, vector=[0.3], payload={"feature": "games"})
    again = PointStruct(id=4, vector=[0.4], payload={"feature": "games"})
    grouped, new_values = plan_writes(doc, [books, alias, unknown, again])
    assert new_values == ["games"]
    assert "games" not in doc.features
    assert books.payload == {"feature": "books", "color": "red"}
    book_slot = pick_slot(["s0", "s1"], 1)
    alias_slot = pick_slot(["s0", "s1"], 2)
    assert books in grouped[book_slot]
    assert alias in grouped[alias_slot]
    assert all(point.id not in (3, 4) for points in grouped.values() for point in points)


def test_plan_writes_rejects_mismatched_payload_feature_types():
    keyword_doc = _doc()
    with pytest.raises(ValueError, match="string"):
        plan_writes(
            keyword_doc,
            [PointStruct(id=1, vector=[0.1], payload={"feature": 999})],
        )

    int_doc = MapDoc(
        collection="items",
        feature_field="subject_id",
        feature_type="integer",
        slots=["s0"],
        features={"42": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
    )
    with pytest.raises(ValueError, match="integer"):
        plan_writes(
            int_doc,
            [PointStruct(id=1, vector=[0.1], payload={"subject_id": "42"})],
        )


def test_plan_writes_uses_both_placements_while_moving():
    doc = _doc()
    doc.features["books"].state = FeatureState.copying
    doc.features["books"].write_to = ["s2"]
    point = PointStruct(id=5, vector=[0.5], payload={"feature": "books"})
    grouped, new_values = plan_writes(doc, [point])
    assert new_values == []
    assert set(grouped) == {"s2", pick_slot(["s0", "s1"], 5)}
    assert all(points == [point] for points in grouped.values())


def test_plan_writes_rejects_a_point_without_the_feature_field():
    doc = _doc()
    points = [
        PointStruct(id=1, vector=[0.1], payload={"feature": "books"}),
        PointStruct(id="missing", vector=[0.2], payload={"color": "red"}),
    ]
    with pytest.raises(ValueError, match="missing"):
        plan_writes(doc, points)


def test_plan_writes_accepts_a_batch():
    doc = _doc()
    batch = Batch(
        ids=[1, 2],
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        payloads=[{"feature": "toys"}, {"feature": "new"}],
    )
    grouped, new_values = plan_writes(doc, batch)
    assert new_values == ["new"]
    assert list(grouped) == ["s2"]
    assert grouped["s2"][0].id == 1
    assert grouped["s2"][0].vector == [0.1, 0.2]
    assert grouped["s2"][0].payload == {"feature": "toys"}


def test_plan_writes_accepts_a_named_vector_batch():
    doc = _doc()
    batch = Batch(
        ids=[8],
        vectors={"text": [[0.1, 0.2]]},
        payloads=[{"feature": "books"}],
    )
    grouped, new_values = plan_writes(doc, batch)
    assert new_values == []
    point = grouped[pick_slot(["s0", "s1"], 8)][0]
    assert point.vector == {"text": [0.1, 0.2]}


def test_plan_writes_keeps_integer_feature_values():
    doc = MapDoc(
        collection="items",
        feature_field="subject_id",
        feature_type="integer",
        slots=["s0"],
        features={"7": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
    )
    point = PointStruct(id=1, vector=[0.1], payload={"subject_id": 7})
    other = PointStruct(id=2, vector=[0.2], payload={"subject_id": 8})
    grouped, new_values = plan_writes(doc, [point, other])
    assert new_values == [8]
    assert grouped["s0"] == [point]
    assert "8" not in doc.features


def test_needs_reload_when_any_feature_value_is_unknown():
    doc = _doc()
    assert needs_reload(doc, None) is False
    assert needs_reload(doc, _filter("books")) is False
    mixed = Filter(
        must=FieldCondition(key="feature", match=MatchAny(any=["books", "ghost"]))
    )
    assert needs_reload(doc, mixed) is True


def test_mutation_slots_include_both_copies_and_fan_out_when_unknown():
    doc = _doc()
    doc.features["books"].state = FeatureState.cutover
    doc.features["books"].write_to = ["s2"]
    query = _filter("books")
    assert mutation_slots(doc, query) == ["s0", "s1", "s2"]
    assert mutation_slots(doc, FilterSelector(filter=query)) == ["s0", "s1", "s2"]
    assert mutation_slots(doc, [1, 2]) is None
    mixed = Filter(
        must=FieldCondition(key="feature", match=MatchAny(any=["books", "ghost"]))
    )
    assert mutation_slots(doc, mixed) is None


def test_filter_copy_targets_follow_the_move_state():
    doc = _doc()
    books = doc.features["books"]
    books.write_to = ["s2"]
    query = _filter("books")
    tag = Filter(must=FieldCondition(key="tag", match=MatchValue(value="pay")))
    assert filter_copy_targets(doc, query) == []
    assert filter_copy_targets(doc, tag) == []
    assert filter_copy_targets(doc, [1]) == []
    books.state = FeatureState.copying
    assert filter_copy_targets(doc, query) == [FilterCopy(("s0", "s1"), query)]
    assert filter_copy_targets(doc, FilterSelector(filter=query)) == [
        FilterCopy(("s0", "s1"), query)
    ]
    assert filter_copy_targets(doc, tag) == [FilterCopy(("s0", "s1"), tag)]
    assert filter_copy_targets(doc, _filter("toys")) == []
    books.state = FeatureState.dual
    assert filter_copy_targets(doc, query) == []
    assert filter_copy_targets(doc, tag) == []
    books.state = FeatureState.cutover
    assert filter_copy_targets(doc, query) == []
    assert filter_copy_targets(doc, tag) == []


def test_slots_to_delete_keeps_a_point_that_stays_on_the_same_feature():
    doc = _doc()
    assert slots_to_delete(doc, 1, "books", "books") == []
    assert is_feature_move(doc, 1, "books", "books") is False
    assert is_feature_move(doc, 1, "1", "books") is False
    assert slots_to_delete(doc, 1, "books", "toys") == [pick_slot(["s0", "s1"], 1)]
    assert is_feature_move(doc, 1, "books", "toys") is True


def test_slots_to_delete_includes_old_slot_during_cutover():
    doc = MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0", "s1", "s2"],
        features={
            "books": FeatureInfo(
                shard_keys=["s0"],
                state=FeatureState.cutover,
                write_to=["s1"],
            ),
            "toys": FeatureInfo(shard_keys=["s2"]),
            "games": FeatureInfo(shard_keys=["s1"]),
        },
        slot_load={"s0": 0, "s1": 0, "s2": 0},
    )
    assert slots_to_delete(doc, 1, "books", "toys") == ["s0", "s1"]
    assert slots_to_delete(doc, 1, "books", "games") == ["s0"]


def test_feature_move_on_the_same_slot_still_counts():
    doc = _doc()
    doc.features["toys"] = FeatureInfo(shard_keys=["s0"])
    doc.features["books"] = FeatureInfo(shard_keys=["s0"])
    assert slots_to_delete(doc, 4, "books", "toys") == []
    assert is_feature_move(doc, 4, "books", "toys") is True


def test_moved_points_lists_ids_that_left_their_slots():
    doc = _doc()
    changed = PointStruct(id=1, vector=[0.1], payload={"feature": "toys"})
    stayed = PointStruct(id=2, vector=[0.2], payload={"feature": "books", "color": "red"})
    missing = PointStruct(id=3, vector=[0.3], payload={"feature": "toys"})
    old_payloads = {1: {"feature": "books"}, 2: {"feature": "books"}}
    assert moved_points(doc, old_payloads, [changed, stayed, missing]) == [
        (1, [pick_slot(["s0", "s1"], 1)]),
    ]


def test_reupsert_of_an_alias_as_the_canonical_id_is_not_a_move():
    doc = _doc()
    doc.alias_feature("books", "libros")
    point = PointStruct(id=1, vector=[0.1], payload={"feature": "libros"})
    assert is_feature_move(doc, 1, "books", "libros") is False
    assert slots_to_delete(doc, 1, "books", "libros") == []
    assert moved_points(doc, {1: {"feature": "books"}}, [point]) == []

    numbers = MapDoc(
        collection="items",
        feature_field="subject_id",
        feature_type="integer",
        slots=["s0"],
        features={"7": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
    )
    numbers.alias_feature(7, 8)
    numbered = PointStruct(id=4, vector=[0.4], payload={"subject_id": 8})
    assert is_feature_move(numbers, 4, 7, 8) is False
    assert moved_points(numbers, {4: {"subject_id": 7}}, [numbered]) == []


def test_moved_points_includes_a_feature_change_that_stays_on_the_slot():
    doc = _doc()
    doc.features["toys"] = FeatureInfo(shard_keys=["s0"])
    doc.features["books"] = FeatureInfo(shard_keys=["s0"])
    point = PointStruct(id=4, vector=[0.4], payload={"feature": "toys"})
    assert moved_points(doc, {4: {"feature": "books"}}, [point]) == [(4, [])]


def test_chunks_yields_the_next_batch_before_the_rest_of_the_iterable():
    seen: list[int] = []

    def source():
        for item in range(5):
            seen.append(item)
            yield item

    stream = chunks(source(), 2)
    assert next(stream) == [0, 1]
    assert seen == [0, 1]
    assert list(stream) == [[2, 3], [4]]


def test_chunks_rejects_a_zero_batch_and_skips_a_negative_one():
    with pytest.raises(ValueError, match="must not be zero"):
        next(chunks([1], 0))
    assert list(chunks([1, 2], -1)) == []


class _Rows:
    def __init__(self, rows):
        self._rows = rows
        self.reads: list[int] = []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        self.reads.append(index)
        return self._rows[index]


def test_iter_upload_points_reads_one_row_at_a_time():
    vectors = _Rows([[0.1], [0.2], [0.3], [0.4]])
    payload = [{"n": 0}, {"n": 1}, {"n": 2}, {"n": 3}]
    stream = iter_upload_points(vectors, payload, [10, 11, 12, 13])
    assert vectors.reads == []
    batch = next(chunks(stream, 2))
    assert vectors.reads == [0, 1]
    assert [point.id for point in batch] == [10, 11]
    assert batch[0].payload == {"n": 0}
    batch[0].payload["n"] = 9
    assert payload[0] == {"n": 0}


def test_iter_upload_points_builds_numpy_rows_and_named_columns():
    numpy = pytest.importorskip("numpy")
    matrix = numpy.array([[0.1, 0.2], [0.3, 0.4]])
    points = list(iter_upload_points(matrix, [{"feature": "a"}, {"feature": "b"}], [1, 2]))
    assert points[0].vector == [0.1, 0.2]
    assert points[1].payload == {"feature": "b"}
    named = {"text": numpy.array([[0.5, 0.6], [0.7, 0.8]])}
    generated = list(iter_upload_points(named, None, None))
    assert generated[0].vector == {"text": [0.5, 0.6]}
    assert generated[0].payload == {}
    assert generated[0].id != generated[1].id


def test_iter_upload_points_checks_a_generator_before_yielding():
    def rows():
        yield [0.1]
        yield [0.2]

    with pytest.raises(ValueError, match="same length"):
        iter_upload_points(rows(), [{"n": 1}], [1, 2])
    points = list(iter_upload_points(rows(), [{"n": 1}, {"n": 2}], (i for i in (4, 5))))
    assert [point.id for point in points] == [4, 5]
    assert points[1].vector == [0.2]


def test_iter_upload_points_rejects_a_short_payload_before_reading_rows():
    vectors = _Rows([[0.1], [0.2]])
    with pytest.raises(ValueError, match="same length"):
        iter_upload_points(vectors, [{}], [1, 2])
    assert vectors.reads == []
    with pytest.raises(IndexError):
        iter_upload_points({"a": [[1.0], [2.0]], "b": [[3.0]]}, None, [1, 2])
