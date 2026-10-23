from datetime import datetime, timedelta, timezone

import pytest

from qdrant_custom_sharding.errors import (
    InvalidMove,
    MapNotFound,
    MapUnavailable,
    MoveNotReady,
    ShardingError,
    VersionConflict,
    WriteDenied,
)
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc, Point

NOW = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)


def make_doc(
    slots: list[str] | None = None,
    features: dict[str, FeatureInfo] | None = None,
    slot_load: dict[str, int] | None = None,
    feature_type: str = "keyword",
) -> MapDoc:
    names = ["s0", "s1", "s2"] if slots is None else slots
    loads = {name: 0 for name in names} if slot_load is None else slot_load
    rows = {} if features is None else features
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=names,
        features=rows,
        slot_load=loads,
        feature_type=feature_type,
    )


def test_errors_extend_sharding_error():
    classes = (
        WriteDenied,
        VersionConflict,
        MapNotFound,
        MapUnavailable,
        MoveNotReady,
        InvalidMove,
    )
    for error in classes:
        assert issubclass(error, ShardingError)


def test_feature_state_values():
    assert FeatureState.stable == "stable"
    assert FeatureState.copying == "copying"
    assert FeatureState.dual == "dual"
    assert FeatureState.cutover == "cutover"


def test_point_keeps_a_list_or_a_named_vector():
    point = Point(id=1, vector=[0.1, 0.2])
    assert point.payload == {}
    named = Point(id="a", vector={"text": [0.1, 0.2]})
    assert named.vector == {"text": [0.1, 0.2]}


def test_feature_type_defaults_to_keyword():
    assert make_doc().feature_type == "keyword"


def test_find_matches_name_and_alias():
    doc = make_doc(
        features={"books": FeatureInfo(shard_keys=["s0"], aliases=["1"])},
    )
    assert doc.find("books") == ("books", doc.features["books"])
    assert doc.find("1")[0] == "books"
    assert doc.find("missing") is None


def test_find_and_add_feature_accept_int_and_str():
    doc = make_doc(feature_type="integer")
    info = doc.add_feature(7)
    assert list(doc.features) == ["7"]
    assert doc.find(7) == ("7", info)
    assert doc.find("7") == ("7", info)
    assert doc.add_feature("7") is info


def test_add_feature_picks_least_loaded_slot():
    doc = make_doc(slot_load={"s0": 5, "s1": 1, "s2": 1})
    doc.features["kept"] = FeatureInfo(shard_keys=["s2"])
    info = doc.add_feature("books")
    assert info.shard_keys == ["s1"]
    assert info.state == FeatureState.stable
    assert info.write_to is None
    assert doc.features["books"] is info
    assert doc.slot_load == {"s0": 5, "s1": 1, "s2": 1}
    assert doc.version == 1


def test_add_feature_uses_feature_count_on_a_tie():
    doc = make_doc()
    assert doc.add_feature("a").shard_keys == ["s0"]
    assert doc.add_feature("b").shard_keys == ["s1"]


def test_add_feature_returns_existing_row():
    doc = make_doc(features={"books": FeatureInfo(shard_keys=["s0"], aliases=["1"])})
    assert doc.add_feature("books") is doc.features["books"]
    assert doc.add_feature("1") is doc.features["books"]
    assert list(doc.features) == ["books"]


def test_add_feature_skips_busy_slots():
    doc = make_doc(slot_load={"s0": 0, "s1": 9, "s2": 9})
    doc.features["moving"] = FeatureInfo(
        shard_keys=["s0"],
        state=FeatureState.dual,
        write_to=["s1"],
    )
    assert doc.add_feature("books").shard_keys == ["s2"]


def test_add_feature_uses_all_slots_when_every_slot_is_busy():
    doc = make_doc(slots=["s0", "s1"], slot_load={"s0": 3, "s1": 1})
    doc.features["moving"] = FeatureInfo(
        shard_keys=["s0"],
        state=FeatureState.cutover,
        write_to=["s1"],
    )
    assert doc.add_feature("books").shard_keys == ["s1"]


def test_busy_slots_are_slots_of_features_that_are_not_stable():
    doc = make_doc()
    doc.add_feature("still")
    doc.features["moving"] = FeatureInfo(
        shard_keys=["s0"],
        state=FeatureState.copying,
        write_to=["s2"],
    )
    assert doc.busy_slots() == {"s0", "s2"}


def test_least_loaded_ranks_and_skips_excluded_slots():
    doc = make_doc(
        slot_load={"s0": 5, "s1": 1, "s2": 1},
        features={"a": FeatureInfo(shard_keys=["s2"])},
    )
    assert doc.least_loaded(2, set()) == ["s1", "s2"]
    assert doc.least_loaded(1, {"s1"}) == ["s2"]
    with pytest.raises(InvalidMove):
        doc.least_loaded(3, {"s0"})


def test_start_move_sets_copying():
    doc = make_doc()
    doc.add_feature("books")
    doc.start_move("books", ["s1", "s2"], NOW)
    info = doc.features["books"]
    assert info.state == FeatureState.copying
    assert info.write_to == ["s1", "s2"]
    assert info.shard_keys == ["s0"]
    assert info.grace_until == NOW
    assert info.copied is False
    assert doc.version == 1
    info.copied = True
    doc.features["books"].state = FeatureState.stable
    doc.features["books"].write_to = None
    doc.start_move("books", ["s2"], NOW)
    assert doc.features["books"].copied is False


def test_start_move_rejects_illegal_requests():
    doc = make_doc()
    doc.add_feature("books")
    with pytest.raises(InvalidMove):
        doc.start_move("missing", ["s1"], NOW)
    with pytest.raises(InvalidMove):
        doc.start_move("books", [], NOW)
    with pytest.raises(InvalidMove):
        doc.start_move("books", ["s9"], NOW)
    with pytest.raises(InvalidMove):
        doc.start_move("books", ["s0"], NOW)
    with pytest.raises(InvalidMove):
        doc.start_move("books", ["s1", "s1"], NOW)
    overlap = make_doc(slots=["s0", "s1", "s5", "s9"])
    overlap.features["books"] = FeatureInfo(shard_keys=["s1"])
    with pytest.raises(InvalidMove):
        overlap.start_move("books", ["s1", "s5", "s9"], NOW)
    busy = make_doc()
    busy.add_feature("books")
    busy.features["other"] = FeatureInfo(
        shard_keys=["s2"],
        state=FeatureState.copying,
        write_to=["s1"],
    )
    with pytest.raises(InvalidMove):
        busy.start_move("books", ["s1"], NOW)
    doc.start_move("books", ["s1"], NOW)
    with pytest.raises(InvalidMove):
        doc.start_move("books", ["s2"], NOW)


def test_move_sequence_and_grace():
    doc = make_doc()
    doc.features["books"] = FeatureInfo(shard_keys=["s0"])
    grace = NOW + timedelta(seconds=60)
    doc.start_move("books", ["s1", "s2"], grace)
    info = doc.features["books"]
    with pytest.raises(InvalidMove):
        doc.open_dual("missing", grace, grace)
    with pytest.raises(MoveNotReady):
        doc.open_dual("books", grace - timedelta(seconds=1), grace)
    next_grace = grace + timedelta(seconds=60)
    with pytest.raises(MoveNotReady):
        doc.open_dual("books", grace, next_grace)
    info.copied = True
    doc.open_dual("books", grace, next_grace)
    assert info.state == FeatureState.dual
    assert info.grace_until == next_grace
    assert info.shard_keys == ["s0"]
    assert info.write_to == ["s1", "s2"]
    with pytest.raises(InvalidMove):
        doc.open_dual("books", next_grace, next_grace)
    with pytest.raises(MoveNotReady):
        doc.cutover("books", next_grace - timedelta(seconds=1), next_grace)
    after_cut = next_grace + timedelta(seconds=60)
    old = doc.cutover("books", next_grace, after_cut)
    assert old == ["s0"]
    assert info.state == FeatureState.cutover
    assert info.shard_keys == ["s0"]
    assert info.write_to == ["s1", "s2"]
    assert info.grace_until == after_cut
    with pytest.raises(MoveNotReady):
        doc.finish_move("books", after_cut - timedelta(seconds=1))
    assert info.shard_keys == ["s0"]
    doc.finish_move("books", after_cut)
    assert info.state == FeatureState.stable
    assert info.shard_keys == ["s1", "s2"]
    assert info.write_to is None
    assert info.grace_until is None
    assert info.copied is False
    assert doc.version == 1


def test_open_dual_and_cutover_and_finish_require_the_previous_state():
    doc = make_doc()
    doc.add_feature("books")
    with pytest.raises(InvalidMove):
        doc.open_dual("books", NOW, NOW)
    with pytest.raises(InvalidMove):
        doc.cutover("books", NOW, NOW)
    with pytest.raises(InvalidMove):
        doc.finish_move("books", NOW)
    doc.start_move("books", ["s1"], NOW)
    with pytest.raises(InvalidMove):
        doc.cutover("books", NOW, NOW)
    with pytest.raises(InvalidMove):
        doc.finish_move("books", NOW)


def test_alias_feature_renames_and_keeps_the_old_id():
    doc = make_doc()
    doc.add_feature("books")
    doc.features["books"].aliases.append("1")
    doc.alias_feature("books", "libros")
    assert "books" not in doc.features
    assert doc.features["libros"].aliases == ["1", "books"]
    assert doc.find("books")[0] == "libros"
    assert doc.find("1")[0] == "libros"
    with pytest.raises(InvalidMove):
        doc.alias_feature("missing", "x")
    with pytest.raises(InvalidMove):
        doc.alias_feature("libros", "books")
    doc.add_feature("toys")
    with pytest.raises(InvalidMove):
        doc.alias_feature("libros", "toys")


def test_alias_feature_accepts_int_and_str():
    doc = make_doc(feature_type="integer")
    doc.add_feature(7)
    doc.alias_feature(7, 8)
    assert list(doc.features) == ["8"]
    assert doc.features["8"].aliases == ["7"]
    assert doc.find(7)[0] == "8"
    assert doc.find("8")[0] == "8"


def test_payload_values_lists_the_canonical_name_then_aliases():
    doc = make_doc(
        features={"books": FeatureInfo(shard_keys=["s0"], aliases=["1", "2"])},
    )
    assert doc.payload_values("books") == ["books", "1", "2"]


def test_payload_values_are_ints_for_an_integer_feature():
    doc = make_doc(
        feature_type="integer",
        features={"7": FeatureInfo(shard_keys=["s0"], aliases=["8"])},
    )
    assert doc.payload_values("7") == [7, 8]


def test_set_stats_replaces_counts():
    doc = make_doc()
    doc.add_feature("books")
    doc.add_feature("toys")
    doc.set_stats(
        {
            "s0": {"books": 4, "ghost": 3},
            "s1": {"toys": 6},
        }
    )
    assert doc.features["books"].points_count == 4
    assert doc.features["toys"].points_count == 6
    assert doc.slot_load == {"s0": 7, "s1": 6, "s2": 0}
    assert doc.version == 1


def test_set_stats_adds_alias_payloads_to_the_canonical_feature():
    doc = make_doc(
        features={
            "books": FeatureInfo(shard_keys=["s0"], aliases=["1"]),
            "toys": FeatureInfo(shard_keys=["s1"]),
        }
    )
    doc.set_stats({"s0": {"books": 2, "1": 3}, "s1": {"toys": 1, "1": 4}})
    assert doc.features["books"].points_count == 9
    assert doc.features["toys"].points_count == 1
    assert doc.slot_load == {"s0": 5, "s1": 5, "s2": 0}


def test_alias_feature_rejects_a_feature_that_is_moving():
    doc = make_doc()
    doc.add_feature("books")
    doc.start_move("books", ["s1"], NOW)
    with pytest.raises(InvalidMove, match="stable"):
        doc.alias_feature("books", "libros")
    assert "books" in doc.features
    assert doc.features["books"].aliases == []


def test_drop_alias_removes_one_alias_and_keeps_the_key():
    doc = make_doc()
    doc.add_feature("books")
    doc.features["books"].aliases.append("1")
    doc.alias_feature("books", "libros")
    doc.drop_alias("books")
    assert list(doc.features) == ["libros"]
    assert doc.features["libros"].aliases == ["1"]
    assert doc.find("books") is None
    assert doc.find("1")[0] == "libros"
    with pytest.raises(InvalidMove, match="unknown alias"):
        doc.drop_alias("libros")
    with pytest.raises(InvalidMove, match="unknown alias"):
        doc.drop_alias("missing")


def test_drop_alias_rejects_a_feature_that_is_moving():
    doc = make_doc()
    doc.add_feature("books")
    doc.alias_feature("books", "libros")
    doc.start_move("libros", ["s1"], NOW)
    with pytest.raises(InvalidMove, match="stable"):
        doc.drop_alias("books")
    assert doc.features["libros"].aliases == ["books"]


def test_drop_alias_accepts_an_int():
    doc = make_doc(feature_type="integer")
    doc.add_feature(7)
    doc.alias_feature(7, 8)
    doc.drop_alias(7)
    assert doc.features["8"].aliases == []
    assert doc.find(7) is None
