import pytest
from qdrant_client.models import (
    Batch,
    ClearPayloadOperation,
    DeleteOperation,
    DeletePayloadOperation,
    DeleteVectorsOperation,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    OverwritePayloadOperation,
    PointIdsList,
    PointStruct,
    PointVectors,
    PointsBatch,
    PointsList,
    SetPayload,
    SetPayloadOperation,
    UpdateVectors,
    UpdateVectorsOperation,
    UpsertOperation,
)

from qdrant_custom_sharding.calls import (
    CopyOnWrite,
    IdUpdatePlan,
    IdWriteGroup,
    OperationCall,
    batch_fields,
    delete_ids,
    copy_on_write,
    feature_retrieve,
    features_need_reload,
    full_retrieve,
    id_scroll_page,
    id_update_plan,
    id_write_groups,
    is_partial_update,
    unknown_feature_ids,
    ids_by_slot,
    ids_of,
    merged_payload,
    moved_upsert,
    operation_call,
    payload_moves_feature,
    points_for_records,
    points_with_payload,
    record_source,
    reject_clear,
    reject_feature_delete,
    request_on_slots,
    require_sdk_sharding,
    scroll_page,
    selector_ids,
    selector_shard,
    selector_value,
    selector_with_ids,
    slot_upsert,
    stored_payloads,
    upload_upsert,
    upsert_wait,
    write_uses_all_slots,
)
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc
from qdrant_custom_sharding.routing import pick_slot


def _doc() -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0", "s1"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0, "s1": 0},
    )


def _filter(value: object) -> Filter:
    return Filter(must=FieldCondition(key="feature", match=MatchValue(value=value)))


class _Record:
    def __init__(self, point_id, payload, vector=None, shard_key=None):
        self.id = point_id
        self.payload = payload
        self.vector = vector if vector is not None else [0.1]
        self.shard_key = shard_key


class _Request:
    def __init__(self, flt, shard_key=None):
        self.filter = flt
        self.shard_key = shard_key

    def model_copy(self, update):
        return _Request(self.filter, update["shard_key"])


def test_request_on_slots_copies_a_known_feature_and_keeps_fanout():
    doc = _doc()
    request = _Request(_filter("books"))
    copied = request_on_slots(doc, request)
    assert copied is not request
    assert copied.shard_key == ["s0"]
    assert request.shard_key is None
    same = _Request(None)
    assert request_on_slots(doc, same) is same
    bare = type("Bare", (), {"filter": _filter("books")})()
    assert request_on_slots(doc, bare) is bare


def test_upsert_wait_is_true_only_while_moving():
    assert upsert_wait(True, False) is True
    assert upsert_wait(False, False) is False
    assert upsert_wait(False, True) is True


def test_slot_upsert_does_not_change_the_original_call():
    flat = {"collection_name": "items", "points": [1], "wait": False, "shard_key_selector": None}
    call = slot_upsert(flat, "s0", [2], True)
    assert call["points"] == [2]
    assert call["shard_key_selector"] == "s0"
    assert call["wait"] is True
    assert flat["points"] == [1]
    assert flat["wait"] is False


def test_upload_and_move_kwargs_name_the_routed_fields():
    flat = {"collection_name": "items", "wait": True, "update_filter": None, "update_mode": "x"}
    assert upload_upsert(flat, ["p"]) == {
        "collection_name": "items",
        "points": ["p"],
        "wait": True,
        "update_filter": None,
        "update_mode": "x",
    }
    assert feature_retrieve("items", [1], "feature")["with_vectors"] is False
    assert full_retrieve("items", [1])["with_vectors"] is True
    page = scroll_page("items", _filter("books"), 4)
    assert page["limit"] == 64
    assert page["offset"] == 4
    moved = moved_upsert("items", [1], None, 5)
    assert moved["wait"] is True
    assert moved["shard_key_selector"] is None
    assert moved["timeout"] == 5
    assert delete_ids("items", [1], "s0", None)["shard_key_selector"] == "s0"
    assert batch_fields({"collection_name": "items", "wait": False})["timeout"] is None


def test_ids_by_slot_groups_deletes():
    assert ids_by_slot([(1, ["s0", "s1"]), (2, ["s0"])]) == {"s0": [1, 2], "s1": [1]}


def test_payload_rules():
    doc = _doc()
    assert payload_moves_feature(doc, {"feature": "toys"}, None, False) is True
    assert payload_moves_feature(doc, {"feature": "toys"}, "color", False) is False
    assert payload_moves_feature(doc, {"color": "red"}, None, False) is False
    with pytest.raises(ValueError, match="overwrite_payload"):
        payload_moves_feature(doc, {"color": "red"}, None, True)
    with pytest.raises(ValueError, match="delete_payload"):
        reject_feature_delete(doc, ["color", "feature"])
    reject_feature_delete(doc, ["color"])
    with pytest.raises(ValueError, match="clear_payload"):
        reject_clear(doc.feature_field)


def test_merged_payload_and_record_order():
    assert merged_payload({"feature": "books", "n": 1}, {"n": 2}, False) == {
        "feature": "books",
        "n": 2,
    }
    assert merged_payload({"feature": "books"}, {"feature": "toys"}, True) == {"feature": "toys"}
    assert merged_payload(None, {"feature": "toys"}, False) == {"feature": "toys"}
    records = [_Record(1, {"feature": "books"}, [0.2]), _Record(2, {"feature": "toys"})]
    points = points_with_payload(records, {"feature": "toys"}, True)
    assert points[0].id == 1
    assert points[0].vector == [0.2]
    assert points[0].payload == {"feature": "toys"}
    assert stored_payloads(records)[2] == {"feature": "toys"}
    newer = [PointStruct(id=2, vector=[0.3], payload={"feature": "toys"})]
    assert [point.id for point in points_for_records(records, newer)] == [2]


def test_record_source_and_selector_shapes():
    query = _filter("books")
    assert record_source(query) == ("filter", query)
    wrapped = FilterSelector(filter=query)
    assert record_source(wrapped) == ("filter", query)
    assert record_source([1, 2]) == ("ids", [1, 2])
    assert record_source(PointIdsList(points=[3])) == ("ids", [3])
    assert record_source(None) is None
    assert ids_of("nope") == []
    assert selector_value(wrapped) is query
    assert selector_value(PointIdsList(points=[4, 5])) == [4, 5]
    assert selector_value([9]) == [9]
    assert selector_shard(PointIdsList(points=[1], shard_key="s1")) == "s1"
    assert selector_shard([1]) is None


def test_require_sdk_sharding_sets_custom_and_rejects_overrides():
    kwargs = {"vectors_config": {"size": 4}}
    require_sdk_sharding(kwargs)
    assert kwargs["shard_number"] == 1
    assert getattr(kwargs["sharding_method"], "value", kwargs["sharding_method"]) == "custom"
    with pytest.raises(ValueError, match="sharding_method"):
        require_sdk_sharding({"shard_number": 2})


def test_operation_call_unpacks_each_kind():
    point = PointStruct(id=1, vector=[0.1], payload={"feature": "books"})
    batch = PointsBatch(batch=Batch(ids=[2], vectors=[[0.2]], payloads=[{"feature": "books"}]))
    upsert = operation_call(UpsertOperation(upsert=PointsList(points=[point], shard_key="s0")))
    assert upsert == OperationCall(
        "upsert",
        {
            "points": [point],
            "shard_key_selector": "s0",
            "update_filter": None,
            "update_mode": None,
        },
    )
    from_batch = operation_call(UpsertOperation(upsert=batch))
    assert from_batch.name == "upsert"
    assert from_batch.values["points"] is batch.batch
    deleted = operation_call(
        DeleteOperation(delete=FilterSelector(filter=_filter("books"), shard_key="s1"))
    )
    assert deleted.name == "delete"
    assert deleted.values["shard_key_selector"] == "s1"
    assert deleted.values["points_selector"] == _filter("books")
    payload = operation_call(
        SetPayloadOperation(
            set_payload=SetPayload(payload={"n": 1}, points=[1], key="n", shard_key="s0")
        )
    )
    assert payload.name == "set_payload"
    assert payload.values["key"] == "n"
    assert payload.values["points"] == [1]
    by_filter = operation_call(
        SetPayloadOperation(set_payload=SetPayload(payload={"n": 1}, filter=_filter("books")))
    )
    assert by_filter.values["points"] == _filter("books")
    overwrite = operation_call(
        OverwritePayloadOperation(
            overwrite_payload=SetPayload(payload={"feature": "toys"}, points=[1])
        )
    )
    assert overwrite.name == "overwrite_payload"
    assert "key" not in overwrite.values
    keys = operation_call(
        DeletePayloadOperation(
            delete_payload={"keys": ["color"], "points": [1], "shard_key": "s0"}
        )
    )
    assert keys.values["keys"] == ["color"]
    cleared = operation_call(
        ClearPayloadOperation(clear_payload=PointIdsList(points=[1, 2], shard_key="s0"))
    )
    assert cleared.name == "clear_payload"
    assert cleared.values["points_selector"] == [1, 2]
    vectors = operation_call(
        UpdateVectorsOperation(
            update_vectors=UpdateVectors(points=[PointVectors(id=1, vector=[0.1])], shard_key="s0")
        )
    )
    assert vectors.name == "update_vectors"
    assert vectors.values["shard_key_selector"] == "s0"
    dropped = operation_call(
        DeleteVectorsOperation(
            delete_vectors={"points": [1], "vector": ["dense"], "shard_key": "s1"}
        )
    )
    assert dropped.name == "delete_vectors"
    assert dropped.values["vectors"] == ["dense"]
    with pytest.raises(ValueError, match="unsupported"):
        operation_call(object())


def test_write_uses_all_slots_for_delete_and_filters_only():
    query = _filter("books")
    assert write_uses_all_slots("delete", [1]) is True
    assert write_uses_all_slots("delete", query) is True
    assert write_uses_all_slots("set_payload", query) is True
    assert write_uses_all_slots("set_payload", FilterSelector(filter=query)) is True
    assert write_uses_all_slots("delete_vectors", query) is True
    assert write_uses_all_slots("set_payload", [1, 2]) is False
    assert write_uses_all_slots("delete_payload", PointIdsList(points=[1])) is False
    assert write_uses_all_slots("update_vectors", [PointVectors(id=1, vector=[0.1])]) is False
    assert write_uses_all_slots("clear_payload", []) is True


def test_selector_ids_keeps_point_objects():
    vectors = [PointVectors(id=1, vector=[0.2]), PointVectors(id=2, vector=[0.3])]
    assert selector_ids(vectors) == [1, 2]
    assert [point.id for point in selector_with_ids(vectors, [2])] == [2]
    assert selector_ids(PointIdsList(points=[4, 5])) == [4, 5]
    assert selector_with_ids(PointIdsList(points=[4, 5]), [5]) == [5]
    assert selector_ids([8, 9]) == [8, 9]
    assert selector_with_ids([8, 9], [8]) == [8]


def test_id_write_groups_use_the_slot_that_holds_the_point():
    doc = _doc()
    records = [_Record(1, {"feature": "books"}), _Record(4, {"feature": "books"})]
    assert id_write_groups(doc, records) == [IdWriteGroup([1, 4], ["s0"])]
    doc.features["books"].aliases.append("libros")
    assert id_write_groups(doc, [_Record(1, {"feature": "libros"})]) == [
        IdWriteGroup([1], ["s0"])
    ]
    assert id_write_groups(doc, [_Record(1, {"other": "books"})]) == []
    assert id_write_groups(doc, [_Record(1, {"feature": "ghost"})]) == []

    split = MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0", "s1", "s2"],
        features={"books": FeatureInfo(shard_keys=["s0", "s1"])},
        slot_load={"s0": 0, "s1": 0, "s2": 0},
    )
    grouped = id_write_groups(
        split, [_Record(1, {"feature": "books"}), _Record(2, {"feature": "books"})]
    )
    by_slot = {group.slots[0]: group.ids for group in grouped}
    assert 1 in by_slot[pick_slot(["s0", "s1"], 1)]
    assert 2 in by_slot[pick_slot(["s0", "s1"], 2)]
    assert all(len(group.slots) == 1 for group in grouped)


def test_partial_update_slots_follow_the_move_state():
    doc = _doc()
    info = doc.features["books"]
    info.write_to = ["s1"]
    record = _Record(1, {"feature": "books"})
    info.state = FeatureState.stable
    assert id_write_groups(doc, [record]) == [IdWriteGroup([1], ["s0"])]
    assert copy_on_write(doc, [record]) == []
    info.state = FeatureState.copying
    assert id_write_groups(doc, [record]) == [
        IdWriteGroup([1], ["s0"]),
        IdWriteGroup([1], ["s1"]),
    ]
    assert copy_on_write(doc, [record]) == [CopyOnWrite([1], "s0", "s1")]
    info.state = FeatureState.dual
    assert id_write_groups(doc, [record]) == [
        IdWriteGroup([1], ["s0"]),
        IdWriteGroup([1], ["s1"]),
    ]
    assert copy_on_write(doc, [record]) == []
    info.state = FeatureState.cutover
    assert id_write_groups(doc, [record]) == [IdWriteGroup([1], ["s1"])]
    assert copy_on_write(doc, [record]) == []


def test_copy_on_write_groups_copying_points_and_skips_the_rest():
    doc = _doc()
    doc.features["books"].write_to = ["s1"]
    doc.features["books"].state = FeatureState.copying
    doc.features["toys"] = FeatureInfo(shard_keys=["s0"])
    records = [
        _Record(1, {"feature": "books"}),
        _Record(1, {"feature": "books"}),
        _Record(4, {"feature": "books"}),
        _Record(2, {"feature": "toys"}),
        _Record(3, {"other": "books"}),
        _Record(9, {"feature": "ghost"}),
    ]
    assert copy_on_write(doc, records) == [CopyOnWrite([1, 4], "s0", "s1")]
    assert is_partial_update("set_payload") is True
    assert is_partial_update("overwrite_payload") is True
    assert is_partial_update("delete_payload") is True
    assert is_partial_update("update_vectors") is True
    assert is_partial_update("delete_vectors") is True
    assert is_partial_update("delete") is False


def test_unknown_feature_ids_reload_once_then_use_one_key():
    doc = _doc()
    books = _Record(1, {"feature": "books"})
    toys = _Record(2, {"feature": "toys"}, shard_key="s1")
    assert unknown_feature_ids(doc, [books, toys]) == [2]
    assert features_need_reload(doc, [books]) is False
    assert features_need_reload(doc, [toys]) is True
    plan = id_update_plan(doc, [1, 2, 9], [books, toys], {1, 2})
    assert plan.groups == [IdWriteGroup([1], ["s0"]), IdWriteGroup([2], ["s1"])]
    assert plan.probe_ids == []
    assert plan.missing_ids == [9]
    assert plan.missing_slot == "s0"
    gone = id_update_plan(doc, [1, 2], [books, toys], {1})
    assert gone.missing_ids == [2]
    assert gone.probe_ids == []
    doc.features["toys"] = FeatureInfo(shard_keys=["s1"])
    assert features_need_reload(doc, [toys]) is False
    placed = id_update_plan(doc, [1, 2], [books, toys], {1, 2})
    assert placed.missing_ids == []
    assert placed.probe_ids == []
    assert placed.groups == [IdWriteGroup([1], ["s0"]), IdWriteGroup([2], ["s1"])]


def test_missing_feature_field_uses_the_record_shard_key():
    doc = _doc()
    blank = _Record(5, {"note": "kept"}, shard_key="s1")
    assert unknown_feature_ids(doc, [blank]) == [5]
    assert features_need_reload(doc, [blank]) is True
    assert id_write_groups(doc, [blank]) == [IdWriteGroup([5], ["s1"])]
    plan = id_update_plan(doc, [5], [blank], {5})
    assert plan == IdUpdatePlan([IdWriteGroup([5], ["s1"])], [], ["s0", "s1"], [], None)


def test_record_without_a_shard_key_is_probed_on_each_slot():
    doc = _doc()
    blank = _Record(5, {"note": "kept"})
    plan = id_update_plan(doc, [5, 9], [blank], {5})
    assert plan.groups == []
    assert plan.probe_ids == [5]
    assert plan.probe_slots == ["s0", "s1"]
    assert plan.missing_ids == [9]
    assert plan.missing_slot == "s0"


def test_copy_on_write_groups_copying_points_and_skips_the_rest_page():
    page = id_scroll_page("items", _filter("books"), "s0", "feature", None)
    assert page["with_payload"] == ["feature"]
    assert page["with_vectors"] is False
    assert page["shard_key_selector"] == "s0"
