from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient

from qdrant_custom_sharding.load import LoadSnapshot
from qdrant_custom_sharding.planning import RebalancePlan
from qdrant_custom_sharding.models import FeatureInfo, MapDoc
from qdrant_custom_sharding.services import CutoverResult
from qdrant_custom_sharding.ports import MapFolder

class ShardingAdmin:
    def create_collection(
        self,
        collection_name: str,
        feature_field: str,
        feature_type: str = ...,
        n_slots: int = ...,
        feature_counts: dict[str, int] | None = ...,
        **create_collection_kwargs: Any,
    ) -> MapDoc: ...
    def add_feature(self, collection: str, value: int | str) -> FeatureInfo: ...
    def add_slot(self, collection: str, slot: str) -> MapDoc: ...
    def migrate(
        self, collection: str, value: int | str, to_slot: str, grace_seconds: int = ...
    ) -> MapDoc: ...
    def resume_move(
        self, collection: str, value: int | str, grace_seconds: int = ...
    ) -> MapDoc: ...
    def split(
        self,
        collection: str,
        value: int | str,
        partitions: int,
        grace_seconds: int = ...,
    ) -> MapDoc: ...
    def start_move(
        self,
        collection: str,
        value: int | str,
        targets: list[str],
        grace_seconds: int = ...,
    ) -> None: ...
    def copy_feature(self, collection: str, value: int | str, batch: int = ...) -> int: ...
    def open_dual(
        self, collection: str, value: int | str, grace_seconds: int = ...
    ) -> None: ...
    def cutover(
        self, collection: str, value: int | str, grace_seconds: int = ...
    ) -> CutoverResult: ...
    def finish_move(self, collection: str, value: int | str) -> None: ...
    def alias_feature(
        self,
        collection: str,
        old: int | str,
        new: int | str,
        grace_seconds: int = ...,
    ) -> int: ...
    def drop_alias(self, collection: str, old: int | str) -> int: ...
    def read_load(self, collection: str) -> LoadSnapshot: ...
    def plan_rebalance(self, collection: str) -> RebalancePlan: ...
    def apply_rebalance(
        self, collection: str, plan: RebalancePlan, grace_seconds: int = ...
    ) -> MapDoc: ...
    def refresh_stats(self, collection: str) -> MapDoc: ...
    def get_map(self, collection: str) -> MapDoc | None: ...

class ShardedQdrantClient(QdrantClient):
    sharding: ShardingAdmin
    def __init__(
        self,
        location: str | None = ...,
        url: str | None = ...,
        port: int | None = ...,
        grpc_port: int = ...,
        prefer_grpc: bool = ...,
        https: bool | None = ...,
        api_key: str | None = ...,
        prefix: str | None = ...,
        timeout: int | None = ...,
        host: str | None = ...,
        path: str | None = ...,
        force_disable_check_same_thread: bool = ...,
        grpc_options: dict[str, Any] | None = ...,
        auth_token_provider: Callable[[], str] | Callable[[], Awaitable[str]] | None = ...,
        cloud_inference: bool = ...,
        local_inference_batch_size: int | None = ...,
        check_compatibility: bool = ...,
        pool_size: int | None = ...,
        headers: dict[str, str] | None = ...,
        *,
        map_uri: str | None = ...,
        map_ttl_seconds: float = ...,
        sharding_warnings: bool = ...,
        detect_feature_changes: bool = ...,
        map_folder: MapFolder | None = ...,
        now_fn: Callable[[], datetime] | None = ...,
        sleep_fn: Callable[[int], None] | None = ...,
        **kwargs: Any,
    ) -> None: ...
