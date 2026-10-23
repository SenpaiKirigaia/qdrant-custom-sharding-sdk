from dataclasses import dataclass
from enum import StrEnum
from math import ceil

from qdrant_custom_sharding.errors import InvalidMove
from qdrant_custom_sharding.load import LoadSnapshot
from qdrant_custom_sharding.models import FeatureInfo, FeatureState, MapDoc

BALANCE_RATIO = 1.5


class StopReason(StrEnum):

    within_ratio = "within_ratio"
    no_legal_move = "no_legal_move"


@dataclass(frozen=True)
class MoveOrder:

    feature: str
    to_slots: tuple[str, ...]
    points: int


@dataclass(frozen=True)
class RebalancePlan:

    collection: str
    moves: tuple[MoveOrder, ...]
    reason: StopReason


def plan_placement(counts: dict[str, int], slots: list[str]) -> dict[str, list[str]]:
    if not counts:
        return {}
    if not slots:
        raise ValueError("slots must not be empty")
    target = max(1, ceil(sum(counts.values()) / len(slots)))
    load = {slot: 0 for slot in slots}
    order = {slot: index for index, slot in enumerate(slots)}
    placement: dict[str, list[str]] = {}
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    for feature, count in ranked:
        width = 1
        if count > target:
            width = min(len(slots), ceil(count / target))
        chosen = _least_loaded(load, order, width)
        placement[feature] = chosen
        _add_load(load, chosen, count)
    return placement


def _least_loaded(
    load: dict[str, int],
    order: dict[str, int],
    width: int,
) -> list[str]:
    ranked = sorted(load, key=lambda slot: (load[slot], order[slot]))
    return ranked[:width]


def _add_load(load: dict[str, int], chosen: list[str], count: int) -> None:
    base, extra = divmod(count, len(chosen))
    for index, slot in enumerate(chosen):
        load[slot] += base + (1 if index < extra else 0)


def plan_rebalance(snapshot: LoadSnapshot, doc: MapDoc) -> RebalancePlan:
    if snapshot.collection != doc.collection:
        raise InvalidMove("snapshot collection does not match the map")
    if not doc.slots:
        raise InvalidMove("slots must not be empty")
    totals = {
        slot: sum(snapshot.counts.get(slot, {}).values()) for slot in doc.slots
    }
    mean = sum(totals.values()) / len(doc.slots)
    busy = doc.busy_slots()
    occupancy = doc.occupancy()
    order = {slot: index for index, slot in enumerate(doc.slots)}
    taken: set[str] = set()
    moves: list[MoveOrder] = []
    while True:
        fullest = max(totals.values())
        if fullest <= BALANCE_RATIO * mean:
            return RebalancePlan(doc.collection, tuple(moves), StopReason.within_ratio)
        picked = _pick(snapshot, doc, totals, occupancy, order, busy, taken, fullest)
        if picked is None:
            return RebalancePlan(doc.collection, tuple(moves), StopReason.no_legal_move)
        move, totals, occupancy = picked
        moves.append(move)
        taken.add(move.feature)


def _pick(
    snapshot: LoadSnapshot,
    doc: MapDoc,
    totals: dict[str, int],
    occupancy: dict[str, int],
    order: dict[str, int],
    busy: set[str],
    taken: set[str],
    fullest: int,
) -> tuple[MoveOrder, dict[str, int], dict[str, int]] | None:
    best: tuple[tuple[int, int, str], MoveOrder, dict[str, int], dict[str, int]] | None
    best = None
    for name, info in doc.features.items():
        if info.state != FeatureState.stable or name in taken:
            continue
        if not info.shard_keys:
            raise InvalidMove("shard_keys must not be empty")
        targets = _targets(doc.slots, totals, occupancy, order, busy, info.shard_keys)
        if targets is None:
            continue
        placed = _placed(snapshot, name, info)
        points = sum(placed.values())
        if points == 0 or not _on_fullest(placed, totals, fullest):
            continue
        updated, used = _transfer(
            totals, occupancy, placed, info.shard_keys, targets, points
        )
        rank = (max(updated.values()), points, name)
        move = MoveOrder(name, tuple(targets), points)
        if best is None or rank < best[0]:
            best = (rank, move, updated, used)
    if best is None or best[0][0] >= fullest:
        return None
    return best[1], best[2], best[3]


def _targets(
    slots: list[str],
    totals: dict[str, int],
    occupancy: dict[str, int],
    order: dict[str, int],
    busy: set[str],
    shard_keys: list[str],
) -> list[str] | None:
    home = set(shard_keys)
    free = [slot for slot in slots if slot not in busy and slot not in home]
    width = len(shard_keys)
    if len(free) < width:
        return None
    free.sort(key=lambda slot: (totals[slot], occupancy.get(slot, 0), order[slot]))
    return free[:width]


def _placed(snapshot: LoadSnapshot, name: str, info: FeatureInfo) -> dict[str, int]:
    names = {name, *(str(alias) for alias in info.aliases)}
    placed: dict[str, int] = {}
    for slot in info.shard_keys:
        facets = snapshot.counts.get(slot, {})
        placed[slot] = sum(count for key, count in facets.items() if str(key) in names)
    return placed


def _on_fullest(placed: dict[str, int], totals: dict[str, int], fullest: int) -> bool:
    return any(
        count > 0 and totals.get(slot) == fullest for slot, count in placed.items()
    )


def _transfer(
    totals: dict[str, int],
    occupancy: dict[str, int],
    placed: dict[str, int],
    shard_keys: list[str],
    targets: list[str],
    points: int,
) -> tuple[dict[str, int], dict[str, int]]:
    updated = dict(totals)
    for slot, count in placed.items():
        if slot in updated:
            updated[slot] -= count
    _add_load(updated, targets, points)
    used = dict(occupancy)
    for slot in set(shard_keys):
        if slot in used:
            used[slot] -= 1
    for slot in targets:
        used[slot] += 1
    return updated, used
