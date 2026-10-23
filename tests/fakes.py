from collections.abc import Iterator

from qdrant_custom_sharding.errors import MapNotFound, MapUnavailable, VersionConflict
from qdrant_custom_sharding.map_codec import map_from_json, map_to_json
from qdrant_custom_sharding.models import MapDoc, Point


class FakeMapStore:

    def __init__(self) -> None:
        self._text: str | None = None
        self.fail_with_unavailable = False

    def load(self) -> MapDoc:
        if self.fail_with_unavailable:
            raise MapUnavailable("map store unavailable")
        if self._text is None:
            raise MapNotFound("map not found")
        return map_from_json(self._text)

    def save(self, doc: MapDoc) -> None:
        if self._text is None:
            if doc.version != 1:
                raise VersionConflict("first save must be version 1")
        else:
            stored = map_from_json(self._text).version
            if stored != doc.version - 1:
                raise VersionConflict(f"stored version is {stored}, got {doc.version}")
        self._text = map_to_json(doc)

    def delete(self) -> None:
        self._text = None


class FakeMapFolder:

    def __init__(self) -> None:
        self._stores: dict[str, FakeMapStore] = {}

    def store_for(self, collection: str) -> FakeMapStore:
        store = self._stores.get(collection)
        if store is None:
            store = FakeMapStore()
            self._stores[collection] = store
        return store


class FakeVectorStore:

    def __init__(self) -> None:
        self._slots: dict[str, dict[str, dict[int | str, Point]]] = {}

    def create_collection(self, collection: str, **kwargs: object) -> None:
        self._slots[collection] = {}

    def create_slot(self, collection: str, slot: str) -> None:
        self._slots[collection][slot] = {}

    def create_feature_index(
        self, collection: str, field: str, feature_type: str
    ) -> None:
        if collection not in self._slots:
            raise KeyError(collection)

    def upsert(self, collection: str, slot: str, points: list[Point]) -> None:
        stored = self._require_slot(collection, slot)
        for point in points:
            stored[point.id] = _copy_point(point)

    def insert_missing(self, collection: str, slot: str, points: list[Point]) -> None:
        stored = self._require_slot(collection, slot)
        for point in points:
            if point.id in stored:
                continue
            stored[point.id] = _copy_point(point)

    def ids_on_slot(
        self, collection: str, slot: str, ids: list[int | str]
    ) -> set[int | str]:
        stored = self._require_slot(collection, slot)
        return {point_id for point_id in ids if point_id in stored}

    def read_ids(self, collection: str, slot: str, ids: list[int | str]) -> list[Point]:
        stored = self._require_slot(collection, slot)
        return [_copy_point(stored[point_id]) for point_id in ids if point_id in stored]

    def delete_ids(self, collection: str, slot: str, ids: list[int | str]) -> None:
        stored = self._require_slot(collection, slot)
        for point_id in ids:
            stored.pop(point_id, None)

    def scroll(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
        batch: int,
    ) -> Iterator[list[Point]]:
        wanted = set(values)
        matched = [
            point
            for point in self._require_slot(collection, slot).values()
            if point.payload.get(field) in wanted
        ]
        matched.sort(key=lambda point: str(point.id))
        for start in range(0, len(matched), batch):
            chunk = matched[start : start + batch]
            yield [_copy_point(point) for point in chunk]

    def scroll_ids(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
        batch: int,
    ) -> Iterator[list[int | str]]:
        for chunk in self.scroll(collection, slot, field, values, batch):
            yield [point.id for point in chunk]

    def delete_feature(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
    ) -> None:
        wanted = set(values)
        points = self._require_slot(collection, slot)
        self._slots[collection][slot] = {
            point_id: point
            for point_id, point in points.items()
            if point.payload.get(field) not in wanted
        }

    def rewrite_feature(
        self,
        collection: str,
        slots: list[str],
        field: str,
        old_values: list[int | str],
        new_value: int | str,
    ) -> int:
        wanted = set(old_values)
        count = 0
        for slot in slots:
            for point in self._require_slot(collection, slot).values():
                if point.payload.get(field) not in wanted:
                    continue
                point.payload[field] = new_value
                count += 1
        return count

    def feature_counts(self, collection: str, slot: str, field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for point in self._require_slot(collection, slot).values():
            value = point.payload.get(field)
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                continue
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _require_slot(self, collection: str, slot: str) -> dict[int | str, Point]:
        slots = self._slots[collection]
        if slot not in slots:
            raise KeyError(slot)
        return slots[slot]


def _copy_point(point: Point) -> Point:
    return Point(
        id=point.id,
        vector=_copy_vector(point.vector),
        payload=dict(point.payload),
    )


def _copy_vector(vector: object) -> object:
    if isinstance(vector, list):
        return list(vector)
    if isinstance(vector, dict):
        return dict(vector)
    return vector
