import zlib

from qdrant_custom_sharding.errors import InvalidMove
from qdrant_custom_sharding.models import FeatureInfo, FeatureState


def pick_slot(slots: list[str], point_id: int | str) -> str:
    """Pick a slot. crc32 stays the same across processes; hash() does not."""
    index = zlib.crc32(str(point_id).encode()) % len(slots)
    return slots[index]


def write_slots(info: FeatureInfo, point_id: int | str) -> list[str]:
    if info.state == FeatureState.stable:
        return [pick_slot(info.shard_keys, point_id)]
    targets = _write_to(info)
    if info.state in (FeatureState.copying, FeatureState.dual):
        return _unique(
            [
                pick_slot(info.shard_keys, point_id),
                pick_slot(targets, point_id),
            ]
        )
    if info.state == FeatureState.cutover:
        return [pick_slot(targets, point_id)]
    raise InvalidMove(f"unknown state {info.state}")


def read_slots(info: FeatureInfo) -> list[str]:
    if info.state in (FeatureState.stable, FeatureState.copying):
        return list(info.shard_keys)
    if info.state in (FeatureState.dual, FeatureState.cutover):
        return list(_write_to(info))
    raise InvalidMove(f"unknown state {info.state}")


def _write_to(info: FeatureInfo) -> list[str]:
    if not info.write_to:
        raise InvalidMove(f"{info.state} requires write_to")
    return info.write_to


def _unique(slots: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for slot in slots:
        if slot in seen:
            continue
        seen.add(slot)
        result.append(slot)
    return result
