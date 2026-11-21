import threading
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum

from qdrant_custom_sharding.errors import MapNotFound, MapUnavailable
from qdrant_custom_sharding.models import MapDoc
from qdrant_custom_sharding.ports import MapStore


class Reason(StrEnum):

    stale_map = "stale_map"
    no_map = "no_map"


class MapCache:

    def __init__(
        self,
        store: MapStore,
        ttl_seconds: float,
        now_fn: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._ttl_seconds = ttl_seconds
        self._now = now_fn
        self._doc: MapDoc | None = None
        self._loaded_at: datetime | None = None
        self._no_map = False
        self._lock = threading.Lock()

    def get(self) -> tuple[MapDoc | None, Reason | None]:
        doc, reason, _reloaded = self.get_status()
        return doc, reason

    def get_status(self) -> tuple[MapDoc | None, Reason | None, bool]:
        with self._lock:
            if self._fresh():
                if self._no_map:
                    return None, None, False
                return self._doc, None, False
            try:
                doc = self._store.load()
            except MapUnavailable:
                if self._doc is not None:
                    return self._doc, Reason.stale_map, False
                return None, Reason.no_map, False
            except MapNotFound:
                self._remember_missing()
                return None, None, False
            self._remember(doc)
            return doc, None, True

    def put(self, doc: MapDoc) -> None:
        with self._lock:
            self._remember(doc)

    def forget(self) -> None:
        with self._lock:
            self._doc = None
            self._loaded_at = None
            self._no_map = False

    def expire(self) -> None:
        with self._lock:
            self._loaded_at = None

    def _fresh(self) -> bool:
        if self._loaded_at is None:
            return False
        if self._doc is None and not self._no_map:
            return False
        age = (self._now() - self._loaded_at).total_seconds()
        return age < self._ttl_seconds

    def _remember(self, doc: MapDoc) -> None:
        self._doc = doc
        self._no_map = False
        self._loaded_at = self._now()

    def _remember_missing(self) -> None:
        self._doc = None
        self._no_map = True
        self._loaded_at = self._now()
