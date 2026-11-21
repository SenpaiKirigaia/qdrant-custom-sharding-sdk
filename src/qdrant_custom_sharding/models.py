from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from qdrant_custom_sharding.errors import InvalidMove, MoveNotReady


class FeatureState(StrEnum):

    stable = "stable"
    copying = "copying"
    dual = "dual"
    cutover = "cutover"


@dataclass
class FeatureInfo:

    shard_keys: list[str]
    state: FeatureState = FeatureState.stable
    write_to: list[str] | None = None
    points_count: int = 0
    aliases: list[str] = field(default_factory=list)
    grace_until: datetime | None = None
    copied: bool = False


@dataclass
class MapDoc:

    collection: str
    feature_field: str
    slots: list[str]
    features: dict[str, FeatureInfo]
    slot_load: dict[str, int]
    feature_type: str = "keyword"
    version: int = 1

    def find(self, value: int | str) -> tuple[str, FeatureInfo] | None:
        feature = str(value)
        info = self.features.get(feature)
        if info is not None:
            return feature, info
        for name, row in self.features.items():
            if feature in row.aliases:
                return name, row
        return None

    def add_feature(self, value: int | str) -> FeatureInfo:
        found = self.find(value)
        if found is not None:
            return found[1]
        exclude = self.busy_slots()
        if set(self.slots) <= exclude:
            exclude = set()
        slot = self.least_loaded(1, exclude)[0]
        info = FeatureInfo(shard_keys=[slot])
        self.features[str(value)] = info
        return info

    def add_slot(self, slot: str) -> None:
        if slot in self.slots:
            return
        self.slots.append(slot)
        self.slot_load[slot] = 0

    def start_move(
        self, feature: int | str, targets: list[str], grace_until: datetime
    ) -> None:
        _name, info = self._require(feature)
        if info.state != FeatureState.stable:
            raise InvalidMove("start_move requires state stable")
        if not targets:
            raise InvalidMove("targets must not be empty")
        if any(slot not in self.slots for slot in targets):
            raise InvalidMove("targets must be known slots")
        if len(set(targets)) != len(targets):
            raise InvalidMove("targets must not repeat a slot")
        busy = self.busy_slots()
        if any(slot in busy for slot in targets):
            raise InvalidMove("targets must not be busy")
        if any(slot in info.shard_keys for slot in targets):
            raise InvalidMove("targets must not overlap shard_keys")
        info.write_to = list(targets)
        info.state = FeatureState.copying
        info.grace_until = grace_until
        info.copied = False

    def copy_allowed(self, feature: int | str, now: datetime) -> None:
        self._copy_ready(feature, now)

    def mark_copied(self, feature: int | str, now: datetime) -> None:
        info = self._copy_ready(feature, now)
        info.copied = True

    def open_dual(
        self, feature: int | str, now: datetime, grace_until: datetime
    ) -> None:
        _name, info = self._require(feature)
        if info.state != FeatureState.copying:
            raise InvalidMove("open_dual requires state copying")
        _check_grace(info, now)
        if not info.copied:
            raise MoveNotReady("open_dual requires copied")
        info.state = FeatureState.dual
        info.grace_until = grace_until

    def cutover(
        self, feature: int | str, now: datetime, grace_until: datetime
    ) -> list[str]:
        _name, info = self._require(feature)
        if info.state != FeatureState.dual:
            raise InvalidMove("cutover requires state dual")
        _check_grace(info, now)
        if not info.write_to:
            raise InvalidMove("cutover requires write_to")
        targets = set(info.write_to)
        old = [slot for slot in info.shard_keys if slot not in targets]
        info.state = FeatureState.cutover
        info.grace_until = grace_until
        return old

    def finish_move(self, feature: int | str, now: datetime) -> None:
        _name, info = self._require(feature)
        if info.state != FeatureState.cutover:
            raise InvalidMove("finish_move requires state cutover")
        _check_grace(info, now)
        if not info.write_to:
            raise InvalidMove("finish_move requires write_to")
        info.shard_keys = list(info.write_to)
        info.write_to = None
        info.state = FeatureState.stable
        info.grace_until = None
        info.copied = False

    def alias_feature(self, old: int | str, new: int | str) -> None:
        old_name = str(old)
        new_name = str(new)
        if old_name not in self.features:
            raise InvalidMove(f"unknown feature {old_name}")
        info = self.features[old_name]
        if info.state != FeatureState.stable:
            raise InvalidMove("alias_feature requires state stable")
        if self.find(new_name) is not None:
            raise InvalidMove(f"{new_name} is already in use")
        self.features.pop(old_name)
        info.aliases.append(old_name)
        self.features[new_name] = info

    def drop_alias(self, old: int | str) -> None:
        old_name = str(old)
        for info in self.features.values():
            if old_name not in info.aliases:
                continue
            if info.state != FeatureState.stable:
                raise InvalidMove("drop_alias requires state stable")
            info.aliases.remove(old_name)
            return
        raise InvalidMove(f"unknown alias {old_name}")

    def payload_values(self, name: str) -> list[int | str]:
        info = self.features[name]
        raw: list[str] = [name, *info.aliases]
        if self.feature_type == "integer":
            return [int(value) for value in raw]
        return raw

    def set_stats(self, counts: dict[str, dict[str, int]]) -> None:
        totals = {name: 0 for name in self.features}
        for slot in self.slots:
            by_feature = counts.get(slot, {})
            self.slot_load[slot] = sum(by_feature.values())
            for feature, count in by_feature.items():
                found = self.find(str(feature))
                if found is None:
                    continue
                totals[found[0]] += count
        for name, total in totals.items():
            self.features[name].points_count = total

    def busy_slots(self) -> set[str]:
        busy: set[str] = set()
        for info in self.features.values():
            if info.state == FeatureState.stable:
                continue
            busy.update(info.shard_keys)
            if info.write_to:
                busy.update(info.write_to)
        return busy

    def occupancy(self) -> dict[str, int]:
        counts = {slot: 0 for slot in self.slots}
        for info in self.features.values():
            used = set(info.shard_keys)
            if info.write_to:
                used.update(info.write_to)
            for slot in used:
                if slot in counts:
                    counts[slot] += 1
        return counts

    def least_loaded(self, n: int, exclude: set[str]) -> list[str]:
        if n < 0:
            raise ValueError("n must not be negative")
        occupancy = self.occupancy()
        order = {slot: index for index, slot in enumerate(self.slots)}
        candidates = [slot for slot in self.slots if slot not in exclude]
        candidates.sort(
            key=lambda slot: (
                self.slot_load.get(slot, 0),
                occupancy.get(slot, 0),
                order[slot],
            )
        )
        if len(candidates) < n:
            raise InvalidMove("not enough free slots")
        return candidates[:n]

    def _copy_ready(self, feature: int | str, now: datetime) -> FeatureInfo:
        _name, info = self._require(feature)
        if info.state != FeatureState.copying:
            raise MoveNotReady("copy requires state copying")
        _check_grace(info, now)
        return info

    def _require(self, feature: int | str) -> tuple[str, FeatureInfo]:
        found = self.find(feature)
        if found is None:
            raise InvalidMove(f"unknown feature {feature}")
        return found


@dataclass
class Point:

    id: int | str
    vector: Any
    payload: dict[str, object] = field(default_factory=dict)


def _check_grace(info: FeatureInfo, now: datetime) -> None:
    if info.grace_until is None or now < info.grace_until:
        raise MoveNotReady("grace period has not ended")


def remaining_grace_seconds(info: FeatureInfo, now: datetime) -> int:
    if info.grace_until is None or now >= info.grace_until:
        return 0
    return math.ceil((info.grace_until - now).total_seconds())
