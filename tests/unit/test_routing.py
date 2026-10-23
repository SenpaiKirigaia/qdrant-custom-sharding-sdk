import zlib

import pytest

from qdrant_custom_sharding.models import FeatureInfo, FeatureState
from qdrant_custom_sharding.routing import pick_slot, read_slots, write_slots


def test_pick_slot_uses_crc32_and_is_stable_for_int_and_str():
    slots = ["s0", "s1", "s2"]
    expected = slots[zlib.crc32(b"42") % len(slots)]
    assert pick_slot(slots, 42) == expected
    assert pick_slot(slots, "42") == expected
    assert pick_slot(slots, 42) == pick_slot(slots, 42)


def test_stable_split_writes_one_slot_and_reads_all():
    info = FeatureInfo(shard_keys=["s0", "s1", "s2", "s3"])
    seen = {write_slots(info, point_id)[0] for point_id in range(40)}
    assert len(seen) > 1
    assert read_slots(info) == ["s0", "s1", "s2", "s3"]
    assert write_slots(info, 7) == [pick_slot(info.shard_keys, 7)]
    assert write_slots(info, "7") == write_slots(info, 7)


@pytest.mark.parametrize(
    ("state", "write_to", "writes", "reads"),
    [
        (FeatureState.stable, None, ["s0"], ["s0"]),
        (FeatureState.copying, ["s1"], ["s0", "s1"], ["s0"]),
        (FeatureState.dual, ["s1"], ["s0", "s1"], ["s1"]),
        (FeatureState.cutover, ["s1"], ["s1"], ["s1"]),
    ],
)
def test_write_and_read_slots_for_each_state(state, write_to, writes, reads):
    info = FeatureInfo(shard_keys=["s0"], state=state, write_to=write_to)
    assert write_slots(info, 1) == writes
    assert write_slots(info, "1") == writes
    assert read_slots(info) == reads


@pytest.mark.parametrize("state", [FeatureState.copying, FeatureState.dual])
def test_moving_writes_old_and_new_once_when_the_hash_collides(state):
    info = FeatureInfo(
        shard_keys=["s0", "s1"],
        state=state,
        write_to=["s0", "s1"],
    )
    assert write_slots(info, 3) == [pick_slot(["s0", "s1"], 3)]


def test_dual_reads_the_new_placement_and_writes_both():
    info = FeatureInfo(
        shard_keys=["s0", "s1"],
        state=FeatureState.dual,
        write_to=["s1", "s2"],
    )
    assert read_slots(info) == ["s1", "s2"]
    assert write_slots(info, 4) == _unique(
        [pick_slot(["s0", "s1"], 4), pick_slot(["s1", "s2"], 4)]
    )


def test_copying_reads_only_the_old_placement():
    info = FeatureInfo(
        shard_keys=["s0", "s1"],
        state=FeatureState.copying,
        write_to=["s1", "s2"],
    )
    assert read_slots(info) == ["s0", "s1"]


def _unique(slots: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for slot in slots:
        if slot in seen:
            continue
        seen.add(slot)
        result.append(slot)
    return result
