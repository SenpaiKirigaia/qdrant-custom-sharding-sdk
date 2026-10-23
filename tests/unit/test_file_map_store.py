import os
import time
from pathlib import Path

import pytest

from qdrant_custom_sharding.errors import MapNotFound, MapUnavailable, VersionConflict
from qdrant_custom_sharding.map_stores import (
    FileMapFolder,
    FileMapStore,
    S3MapFolder,
    _drop_drive_letter,
    _s3_client,
    open_map_folder,
)
from qdrant_custom_sharding.models import FeatureInfo, MapDoc


def _doc(version: int = 1) -> MapDoc:
    return MapDoc(
        collection="items",
        feature_field="feature",
        slots=["s0"],
        features={"books": FeatureInfo(shard_keys=["s0"])},
        slot_load={"s0": 0},
        version=version,
    )


def test_file_map_store_version_rule_and_conflict(tmp_path: Path):
    path = tmp_path / "map.json"
    store = FileMapStore(path)
    with pytest.raises(MapNotFound):
        store.load()
    with pytest.raises(VersionConflict):
        store.save(_doc(version=2))
    assert not path.exists()

    store.save(_doc())
    loaded = store.load()
    assert loaded == _doc()
    loaded.version = 2
    store.save(loaded)
    with pytest.raises(VersionConflict):
        store.save(loaded)
    assert store.load().version == 2
    assert not Path(str(path) + ".lock").exists()


def test_file_map_store_read_error_is_unavailable(tmp_path: Path):
    store = FileMapStore(tmp_path)
    with pytest.raises(MapUnavailable):
        store.load()


def test_save_times_out_when_the_lock_is_held(tmp_path: Path):
    path = tmp_path / "map.json"
    store = FileMapStore(path)
    lock = Path(str(path) + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    try:
        with pytest.raises(MapUnavailable):
            store.save(_doc())
        assert not path.exists()
    finally:
        os.close(fd)
        os.remove(lock)


def test_file_map_store_delete_is_fine_when_the_file_is_missing(tmp_path: Path):
    path = tmp_path / "items.json"
    store = FileMapStore(path)
    store.delete()
    store.save(_doc())
    assert path.exists()
    store.delete()
    assert not path.exists()
    with pytest.raises(MapNotFound):
        store.load()


def test_file_map_folder_uses_one_json_file_per_collection(tmp_path: Path):
    folder = FileMapFolder(tmp_path)
    store = folder.store_for("items")
    assert isinstance(store, FileMapStore)
    assert store.path == tmp_path / "items.json"
    store.save(_doc())
    assert folder.store_for("items").load() == _doc()
    other = folder.store_for("other")
    with pytest.raises(MapNotFound):
        other.load()


def test_open_map_folder_picks_a_directory_or_an_s3_prefix(tmp_path: Path):
    direct = open_map_folder(str(tmp_path))
    assert isinstance(direct, FileMapFolder)
    assert direct.directory == tmp_path
    assert direct.store_for("items").path == tmp_path / "items.json"

    uri = tmp_path.as_uri()
    if not uri.endswith("/"):
        uri = uri + "/"
    prefixed = open_map_folder(uri)
    assert isinstance(prefixed, FileMapFolder)
    assert prefixed.store_for("items").path == tmp_path / "items.json"

    remote = open_map_folder("s3://bucket/maps")
    assert isinstance(remote, S3MapFolder)
    assert remote.bucket == "bucket"
    assert remote.prefix == "maps/"
    assert remote.store_for("items").key == "maps/items.json"

    root = open_map_folder("s3://bucket/")
    assert isinstance(root, S3MapFolder)
    assert root.prefix == ""
    assert root.store_for("items").key == "items.json"

    with pytest.raises(ValueError):
        open_map_folder("s3://")


def test_release_does_not_unlink_a_lock_another_process_created(tmp_path: Path):
    path = tmp_path / "map.json"
    first = FileMapStore(path)
    second = FileMapStore(path)
    held = first._acquire()
    os.unlink(first._lock_path())
    other = second._acquire()
    first._release(held)
    assert first._lock_path().exists()
    current = os.stat(first._lock_path())
    still = os.fstat(other)
    assert (still.st_ino, still.st_dev) == (current.st_ino, current.st_dev)
    second._release(other)
    assert not first._lock_path().exists()


def test_stale_lock_does_not_block_save(tmp_path: Path):
    path = tmp_path / "map.json"
    store = FileMapStore(path)
    lock = Path(str(path) + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    os.close(fd)
    old = time.time() - 60
    os.utime(lock, (old, old))
    store.save(_doc())
    assert store.load() == _doc()
    assert not lock.exists()


def test_file_uri_drive_letter():
    folder = open_map_folder("file:///C:/dir/")
    if os.name == "nt":
        assert folder.directory.as_posix().rstrip("/") == "C:/dir"
    else:
        assert folder.directory.as_posix().rstrip("/") == "/dir"


def test_posix_file_uri_drops_the_drive_letter():
    assert _drop_drive_letter("/C:/x/map.json") == "/x/map.json"
    assert _drop_drive_letter("/x/map.json") == "/x/map.json"


def test_endpoint_without_region_uses_us_east_1(monkeypatch, tmp_path: Path):
    import boto3

    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-creds"))
    seen: dict[str, object] = {}
    real_client = boto3.client

    def capture(service_name, **kwargs):
        seen.update(kwargs)
        return real_client(service_name, **kwargs)

    monkeypatch.setattr(boto3, "client", capture)
    client = _s3_client()
    assert seen.get("region_name") == "us-east-1"
    assert client.meta.region_name == "us-east-1"


def test_endpoint_keeps_a_configured_region(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-creds"))
    client = _s3_client()
    assert client.meta.region_name == "eu-west-1"
