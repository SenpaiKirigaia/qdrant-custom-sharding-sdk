import threading

import pytest
from botocore.exceptions import ClientError

from qdrant_custom_sharding.errors import MapNotFound, MapUnavailable, VersionConflict
from qdrant_custom_sharding.map_stores import (
    S3MapFolder,
    S3MapStore,
    _s3_client,
    open_map_folder,
)
from qdrant_custom_sharding.models import FeatureInfo, MapDoc

pytestmark = pytest.mark.integration

_BUCKET = "sharding-maps"


class _JoinedReads:

    def __init__(self, client) -> None:
        self._client = client
        self._armed = False
        self._gate = threading.Barrier(2)

    def arm(self) -> None:
        self._armed = True

    def get_object(self, **kwargs):
        try:
            return self._client.get_object(**kwargs)
        finally:
            if self._armed:
                self._gate.wait(timeout=10)

    def put_object(self, **kwargs):
        return self._client.put_object(**kwargs)


def _doc(version: int = 1) -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
        version=version,
    )


def _race(left: S3MapStore, right: S3MapStore, version: int) -> list[BaseException | None]:
    outcomes: list[BaseException | None] = []
    lock = threading.Lock()

    def attempt(store: S3MapStore) -> None:
        result: BaseException | None = None
        try:
            store.save(_doc(version=version))
        except Exception as exc:
            result = exc
        with lock:
            outcomes.append(result)

    threads = [
        threading.Thread(target=attempt, args=(left,)),
        threading.Thread(target=attempt, args=(right,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def _one_conflict(outcomes: list[BaseException | None]) -> VersionConflict:
    assert len(outcomes) == 2
    winners = [item for item in outcomes if item is None]
    conflicts = [item for item in outcomes if isinstance(item, VersionConflict)]
    other = [
        item
        for item in outcomes
        if item is not None and not isinstance(item, VersionConflict)
    ]
    assert other == []
    assert len(winners) == 1
    assert len(conflicts) == 1
    return conflicts[0]


def test_missing_key_is_not_found(object_key):
    store = S3MapStore(_BUCKET, object_key)
    with pytest.raises(MapNotFound):
        store.load()


def test_save_and_load_round_trip(object_key):
    store = S3MapStore(_BUCKET, object_key)
    store.save(_doc())
    assert store.load() == _doc()
    updated = _doc(version=2)
    store.save(updated)
    assert store.load() == updated


def test_saving_version_2_again_conflicts(object_key):
    store = S3MapStore(_BUCKET, object_key)
    store.save(_doc())
    updated = _doc(version=2)
    store.save(updated)
    with pytest.raises(VersionConflict):
        store.save(updated)
    assert store.load().version == 2


def test_two_stores_one_wins_if_match(object_key):
    ready = S3MapStore(_BUCKET, object_key)
    ready.save(_doc())
    ready.save(_doc(version=2))
    gate = _JoinedReads(_s3_client())
    left = S3MapStore(_BUCKET, object_key, client=gate)
    right = S3MapStore(_BUCKET, object_key, client=gate)
    assert left.load().version == 2
    assert right.load().version == 2
    gate.arm()
    conflict = _one_conflict(_race(left, right, 3))
    assert isinstance(conflict.__cause__, ClientError)
    status = conflict.__cause__.response["ResponseMetadata"]["HTTPStatusCode"]
    code = conflict.__cause__.response["Error"]["Code"]
    assert status in {412, 409}, (status, code)
    assert S3MapStore(_BUCKET, object_key).load() == _doc(version=3)


def test_first_save_conflicts_when_object_exists(object_key):
    gate = _JoinedReads(_s3_client())
    left = S3MapStore(_BUCKET, object_key, client=gate)
    right = S3MapStore(_BUCKET, object_key, client=gate)
    gate.arm()
    conflict = _one_conflict(_race(left, right, 1))
    assert isinstance(conflict.__cause__, ClientError)
    status = conflict.__cause__.response["ResponseMetadata"]["HTTPStatusCode"]
    code = conflict.__cause__.response["Error"]["Code"]
    assert status in {412, 409}, (status, code)
    assert S3MapStore(_BUCKET, object_key).load() == _doc()


def test_unreachable_endpoint_is_unavailable(monkeypatch, object_key):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9999")
    store = S3MapStore(_BUCKET, object_key)
    with pytest.raises(MapUnavailable):
        store.load()
    with pytest.raises(MapUnavailable):
        store.save(_doc())


def test_open_s3_folder_and_delete(object_key):
    prefix = object_key.removesuffix(".json") + "/"
    folder = open_map_folder(f"s3://{_BUCKET}/{prefix}")
    assert isinstance(folder, S3MapFolder)
    assert folder.prefix == prefix
    store = folder.store_for("items")
    assert store.key == prefix + "items.json"
    try:
        store.delete()
        store.save(_doc())
        assert folder.store_for("items").load() == _doc()
        store.delete()
        with pytest.raises(MapNotFound):
            store.load()
        store.delete()
    finally:
        _s3_client().delete_object(Bucket=_BUCKET, Key=store.key)


def test_missing_bucket_is_unavailable(object_key):
    store = S3MapStore("sharding-maps-missing", object_key)
    with pytest.raises(MapUnavailable) as loaded:
        store.load()
    assert isinstance(loaded.value.__cause__, ClientError)
    with pytest.raises(MapUnavailable) as saved:
        store.save(_doc())
    assert isinstance(saved.value.__cause__, ClientError)
