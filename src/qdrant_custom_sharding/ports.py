from collections.abc import Iterator
from typing import Protocol

from qdrant_custom_sharding.models import MapDoc, Point


class MapStore(Protocol):

    def load(self) -> MapDoc:
        pass

    def save(self, doc: MapDoc) -> None:
        pass

    def delete(self) -> None:
        pass


class MapFolder(Protocol):

    def store_for(self, collection: str) -> MapStore:
        pass


class ListableMapFolder(MapFolder, Protocol):

    def list_collections(self) -> list[str]:
        pass


class VectorStore(Protocol):

    def create_collection(self, collection: str, **kwargs: object) -> None: ...

    def create_slot(self, collection: str, slot: str) -> None: ...

    def create_feature_index(
        self, collection: str, field: str, feature_type: str
    ) -> None: ...

    def upsert(self, collection: str, slot: str, points: list[Point]) -> None: ...

    def insert_missing(self, collection: str, slot: str, points: list[Point]) -> None: ...

    def ids_on_slot(
        self, collection: str, slot: str, ids: list[int | str]
    ) -> set[int | str]: ...

    def read_ids(self, collection: str, slot: str, ids: list[int | str]) -> list[Point]: ...

    def delete_ids(self, collection: str, slot: str, ids: list[int | str]) -> None: ...

    def scroll_ids(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
        batch: int,
    ) -> Iterator[list[int | str]]: ...

    def scroll(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
        batch: int,
    ) -> Iterator[list[Point]]: ...

    def delete_feature(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
    ) -> None: ...

    def feature_counts(
        self, collection: str, slot: str, field: str
    ) -> dict[str, int]: ...

    def rewrite_feature(
        self,
        collection: str,
        slots: list[str],
        field: str,
        old_values: list[int | str],
        new_value: int | str,
    ) -> int:
        ...
