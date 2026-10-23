from collections.abc import AsyncIterator, Iterator

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    IntegerIndexParams,
    KeywordIndexParams,
    MatchAny,
    PointStruct,
    ShardingMethod,
)

from qdrant_custom_sharding.errors import ShardingError
from qdrant_custom_sharding.models import Point

_INSERT_ONLY_ERROR = (
    "moves need qdrant-client and server >= 1.17 for insert-only upserts"
)

_FACET_LIMIT = 100_000


class QdrantVectorStore:

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def create_collection(self, collection: str, **kwargs: object) -> None:
        kwargs.setdefault("sharding_method", ShardingMethod.CUSTOM)
        kwargs.setdefault("shard_number", 1)
        self._client.create_collection(collection_name=collection, **kwargs)

    def create_slot(self, collection: str, slot: str) -> None:
        self._client.create_shard_key(collection_name=collection, shard_key=slot)

    def create_feature_index(
        self, collection: str, field: str, feature_type: str
    ) -> None:
        self._client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=_index_params(feature_type),
            wait=True,
        )

    def upsert(self, collection: str, slot: str, points: list[Point]) -> None:
        self._client.upsert(
            collection_name=collection,
            points=_point_structs(points),
            shard_key_selector=slot,
            wait=True,
        )

    def insert_missing(self, collection: str, slot: str, points: list[Point]) -> None:
        if not points:
            return
        self._client.upsert(
            collection_name=collection,
            points=_point_structs(points),
            shard_key_selector=slot,
            wait=True,
            update_mode=_insert_only_mode(),
        )

    def ids_on_slot(
        self, collection: str, slot: str, ids: list[int | str]
    ) -> set[int | str]:
        if not ids:
            return set()
        records = self._client.retrieve(
            collection_name=collection,
            ids=list(ids),
            shard_key_selector=slot,
            with_payload=False,
            with_vectors=False,
        )
        return {record.id for record in records}

    def read_ids(self, collection: str, slot: str, ids: list[int | str]) -> list[Point]:
        if not ids:
            return []
        records = self._client.retrieve(
            collection_name=collection,
            ids=list(ids),
            shard_key_selector=slot,
            with_payload=True,
            with_vectors=True,
        )
        return [_point(record) for record in records]

    def delete_ids(self, collection: str, slot: str, ids: list[int | str]) -> None:
        if not ids:
            return
        self._client.delete(
            collection_name=collection,
            points_selector=list(ids),
            shard_key_selector=slot,
            wait=True,
        )

    def scroll(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
        batch: int,
    ) -> Iterator[list[Point]]:
        offset = None
        query = _feature_filter(field, values)
        while True:
            records, offset = self._client.scroll(
                collection_name=collection,
                scroll_filter=query,
                shard_key_selector=slot,
                with_payload=True,
                with_vectors=True,
                limit=batch,
                offset=offset,
            )
            if records:
                yield [_point(record) for record in records]
            if offset is None:
                return

    def scroll_ids(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
        batch: int,
    ) -> Iterator[list[int | str]]:
        offset = None
        query = _feature_filter(field, values)
        while True:
            records, offset = self._client.scroll(
                collection_name=collection,
                scroll_filter=query,
                shard_key_selector=slot,
                with_payload=[field],
                with_vectors=False,
                limit=batch,
                offset=offset,
            )
            if records:
                yield [record.id for record in records]
            if offset is None:
                return

    def delete_feature(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
    ) -> None:
        self._client.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=_feature_filter(field, values)),
            shard_key_selector=slot,
            wait=True,
        )

    def feature_counts(self, collection: str, slot: str, field: str) -> dict[str, int]:
        response = self._client.facet(
            collection_name=collection,
            key=field,
            shard_key_selector=slot,
            exact=True,
            limit=_FACET_LIMIT,
        )
        return {str(hit.value): hit.count for hit in response.hits}

    def rewrite_feature(
        self,
        collection: str,
        slots: list[str],
        field: str,
        old_values: list[int | str],
        new_value: int | str,
    ) -> int:
        if not slots or not old_values:
            return 0
        query = _feature_filter(field, old_values)
        counted = self._client.count(
            collection_name=collection,
            count_filter=query,
            exact=True,
            shard_key_selector=slots,
        )
        if counted.count == 0:
            return 0
        self._client.set_payload(
            collection_name=collection,
            payload={field: new_value},
            points=query,
            shard_key_selector=slots,
            wait=True,
        )
        return counted.count


class AsyncQdrantVectorStore:

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def create_collection(self, collection: str, **kwargs: object) -> None:
        kwargs.setdefault("sharding_method", ShardingMethod.CUSTOM)
        kwargs.setdefault("shard_number", 1)
        await self._client.create_collection(collection_name=collection, **kwargs)

    async def create_slot(self, collection: str, slot: str) -> None:
        await self._client.create_shard_key(collection_name=collection, shard_key=slot)

    async def create_feature_index(
        self, collection: str, field: str, feature_type: str
    ) -> None:
        await self._client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=_index_params(feature_type),
            wait=True,
        )

    async def upsert(self, collection: str, slot: str, points: list[Point]) -> None:
        await self._client.upsert(
            collection_name=collection,
            points=_point_structs(points),
            shard_key_selector=slot,
            wait=True,
        )

    async def insert_missing(self, collection: str, slot: str, points: list[Point]) -> None:
        if not points:
            return
        await self._client.upsert(
            collection_name=collection,
            points=_point_structs(points),
            shard_key_selector=slot,
            wait=True,
            update_mode=_insert_only_mode(),
        )

    async def ids_on_slot(
        self, collection: str, slot: str, ids: list[int | str]
    ) -> set[int | str]:
        if not ids:
            return set()
        records = await self._client.retrieve(
            collection_name=collection,
            ids=list(ids),
            shard_key_selector=slot,
            with_payload=False,
            with_vectors=False,
        )
        return {record.id for record in records}

    async def read_ids(self, collection: str, slot: str, ids: list[int | str]) -> list[Point]:
        if not ids:
            return []
        records = await self._client.retrieve(
            collection_name=collection,
            ids=list(ids),
            shard_key_selector=slot,
            with_payload=True,
            with_vectors=True,
        )
        return [_point(record) for record in records]

    async def delete_ids(self, collection: str, slot: str, ids: list[int | str]) -> None:
        if not ids:
            return
        await self._client.delete(
            collection_name=collection,
            points_selector=list(ids),
            shard_key_selector=slot,
            wait=True,
        )

    async def scroll(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
        batch: int,
    ) -> AsyncIterator[list[Point]]:
        offset = None
        query = _feature_filter(field, values)
        while True:
            records, offset = await self._client.scroll(
                collection_name=collection,
                scroll_filter=query,
                shard_key_selector=slot,
                with_payload=True,
                with_vectors=True,
                limit=batch,
                offset=offset,
            )
            if records:
                yield [_point(record) for record in records]
            if offset is None:
                return

    async def scroll_ids(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
        batch: int,
    ) -> AsyncIterator[list[int | str]]:
        offset = None
        query = _feature_filter(field, values)
        while True:
            records, offset = await self._client.scroll(
                collection_name=collection,
                scroll_filter=query,
                shard_key_selector=slot,
                with_payload=[field],
                with_vectors=False,
                limit=batch,
                offset=offset,
            )
            if records:
                yield [record.id for record in records]
            if offset is None:
                return

    async def delete_feature(
        self,
        collection: str,
        slot: str,
        field: str,
        values: list[int | str],
    ) -> None:
        await self._client.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=_feature_filter(field, values)),
            shard_key_selector=slot,
            wait=True,
        )

    async def feature_counts(self, collection: str, slot: str, field: str) -> dict[str, int]:
        response = await self._client.facet(
            collection_name=collection,
            key=field,
            shard_key_selector=slot,
            exact=True,
            limit=_FACET_LIMIT,
        )
        return {str(hit.value): hit.count for hit in response.hits}

    async def rewrite_feature(
        self,
        collection: str,
        slots: list[str],
        field: str,
        old_values: list[int | str],
        new_value: int | str,
    ) -> int:
        if not slots or not old_values:
            return 0
        query = _feature_filter(field, old_values)
        counted = await self._client.count(
            collection_name=collection,
            count_filter=query,
            exact=True,
            shard_key_selector=slots,
        )
        if counted.count == 0:
            return 0
        await self._client.set_payload(
            collection_name=collection,
            payload={field: new_value},
            points=query,
            shard_key_selector=slots,
            wait=True,
        )
        return counted.count


def insert_only_supported() -> bool:
    return _load_insert_only() is not None


def _insert_only_mode():
    mode = _load_insert_only()
    if mode is None:
        raise ShardingError(_INSERT_ONLY_ERROR)
    return mode


def _load_insert_only():
    try:
        from qdrant_client.models import UpdateMode
    except ImportError:
        return None
    return getattr(UpdateMode, "INSERT_ONLY", None)


def _point_structs(points: list[Point]) -> list[PointStruct]:
    return [
        PointStruct(id=point.id, vector=point.vector, payload=point.payload)
        for point in points
    ]


def _index_params(feature_type: str) -> KeywordIndexParams | IntegerIndexParams:
    if feature_type == "keyword":
        return KeywordIndexParams(type="keyword", is_tenant=True)
    if feature_type == "integer":
        return IntegerIndexParams(
            type="integer", lookup=True, range=False, is_principal=True
        )
    raise ValueError(f"unknown feature_type {feature_type}")


def _feature_filter(field: str, values: list[int | str]) -> Filter:
    return Filter(must=[FieldCondition(key=field, match=MatchAny(any=values))])


def _point(record) -> Point:
    return Point(
        id=record.id,
        vector=record.vector,
        payload=_payload(record.payload),
    )


def _payload(value: object) -> dict[str, object]:
    if not value:
        return {}
    return dict(value)
