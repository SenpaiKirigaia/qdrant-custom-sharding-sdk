from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class LoadSnapshot:

    collection: str
    slots: tuple[str, ...]
    counts: Mapping[str, Mapping[str, int]]

    def as_facet_counts(self) -> dict[str, dict[str, int]]:
        return {slot: dict(self.counts[slot]) for slot in self.slots}


def snapshot_from_facets(
    collection: str,
    slots: Sequence[str],
    facets: Mapping[str, Mapping[str, int]],
) -> LoadSnapshot:
    counts = {
        slot: MappingProxyType(dict(facets.get(slot, {})))
        for slot in slots
    }
    return LoadSnapshot(
        collection=collection,
        slots=tuple(slots),
        counts=MappingProxyType(counts),
    )
