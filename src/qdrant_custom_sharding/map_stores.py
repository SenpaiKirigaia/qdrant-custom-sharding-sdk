import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

try:
    from botocore.exceptions import ClientError as _BotoClientError
    from botocore.exceptions import ConnectionError as _BotoConnectionError
    from botocore.exceptions import HTTPClientError as _BotoHTTPClientError
except ImportError:
    _CLIENT_ERRORS: tuple[type[BaseException], ...] = ()
    _NETWORK_ERRORS: tuple[type[BaseException], ...] = ()
else:
    _CLIENT_ERRORS = (_BotoClientError,)
    _NETWORK_ERRORS = (_BotoConnectionError, _BotoHTTPClientError)

from qdrant_custom_sharding.errors import (
    MapNotFound,
    MapUnavailable,
    ShardingError,
    VersionConflict,
)
from qdrant_custom_sharding.map_codec import map_from_json, map_to_json
from qdrant_custom_sharding.models import MapDoc
from qdrant_custom_sharding.ports import MapFolder

_LOCK_TRIES = 5
_LOCK_WAIT_SECONDS = 0.01
_STALE_LOCK_SECONDS = 30
_S3_CONNECT_TIMEOUT = 2
_S3_READ_TIMEOUT = 5
_S3_MAX_ATTEMPTS = 2


class FileMapStore:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> MapDoc:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise MapNotFound("map not found") from exc
        except OSError as exc:
            raise MapUnavailable("map unreadable") from exc
        return _parse_map(text)

    def save(self, doc: MapDoc) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = self._acquire()
        try:
            self._write(doc)
        finally:
            self._release(fd)

    def _acquire(self) -> int:
        last: FileExistsError | None = None
        removed_stale = False
        for _ in range(_LOCK_TRIES):
            try:
                return _open_lock(self._lock_path())
            except FileExistsError as exc:
                last = exc
                if not removed_stale and _stale_lock(self._lock_path()):
                    _clear_stale_lock(self._lock_path())
                    removed_stale = True
                    continue
                time.sleep(_LOCK_WAIT_SECONDS)
        raise MapUnavailable("map lock timeout") from last

    def _release(self, fd: int) -> None:
        lock_path = self._lock_path()
        try:
            held = os.fstat(fd)
            try:
                current = os.stat(lock_path)
            except FileNotFoundError:
                return
            if held.st_ino == current.st_ino and held.st_dev == current.st_dev:
                os.unlink(lock_path)
        finally:
            os.close(fd)

    def _write(self, doc: MapDoc) -> None:
        if not self.path.exists():
            if doc.version != 1:
                raise VersionConflict("first save must be version 1")
        else:
            stored = self.load().version
            if stored != doc.version - 1:
                raise VersionConflict(
                    f"stored version is {stored}, got {doc.version}"
                )
        temporary = Path(str(self.path) + ".tmp")
        temporary.write_text(map_to_json(doc), encoding="utf-8")
        os.replace(temporary, self.path)

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return

    def _lock_path(self) -> Path:
        return Path(str(self.path) + ".lock")


class FileMapFolder:

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def store_for(self, collection: str) -> FileMapStore:
        return FileMapStore(self.directory / f"{collection}.json")

    def list_collections(self) -> list[str]:
        if not self.directory.exists():
            return []
        try:
            entries = list(self.directory.iterdir())
        except OSError as exc:
            raise MapUnavailable("map folder unreadable") from exc
        names: list[str] = []
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".json"):
                names.append(entry.name[:-5])
        return sorted(names)


class S3MapStore:

    def __init__(self, bucket: str, key: str, client=None) -> None:
        self.bucket = bucket
        self.key = key
        self._client = client

    def load(self) -> MapDoc:
        stored, _etag = self._read()
        if stored is None:
            raise MapNotFound("map not found")
        return stored

    def save(self, doc: MapDoc) -> None:
        stored, etag = self._read()
        if stored is None:
            if doc.version != 1:
                raise VersionConflict("first save must be version 1")
        elif stored.version != doc.version - 1:
            raise VersionConflict(
                f"stored version is {stored.version}, got {doc.version}"
            )
        self._put(doc, etag)

    def _read(self) -> tuple[MapDoc | None, str | None]:
        try:
            response = self._s3().get_object(Bucket=self.bucket, Key=self.key)
        except _CLIENT_ERRORS as exc:
            if _missing(exc):
                return None, None
            raise MapUnavailable("map store unavailable") from exc
        except _NETWORK_ERRORS as exc:
            raise MapUnavailable("map store unavailable") from exc
        body = response["Body"].read()
        text = body.decode("utf-8")
        return _parse_map(text), response["ETag"]

    def _put(self, doc: MapDoc, etag: str | None) -> None:
        payload = {
            "Bucket": self.bucket,
            "Key": self.key,
            "Body": map_to_json(doc).encode("utf-8"),
        }
        if etag is None:
            payload["IfNoneMatch"] = "*"
        else:
            payload["IfMatch"] = etag
        try:
            self._s3().put_object(**payload)
        except _CLIENT_ERRORS as exc:
            if _conflict(exc):
                raise VersionConflict("map version conflict") from exc
            raise MapUnavailable("map store unavailable") from exc
        except _NETWORK_ERRORS as exc:
            raise MapUnavailable("map store unavailable") from exc

    def delete(self) -> None:
        try:
            self._s3().delete_object(Bucket=self.bucket, Key=self.key)
        except _CLIENT_ERRORS as exc:
            raise MapUnavailable("map store unavailable") from exc
        except _NETWORK_ERRORS as exc:
            raise MapUnavailable("map store unavailable") from exc

    def _s3(self):
        if self._client is None:
            self._client = _s3_client()
        return self._client


class S3MapFolder:

    def __init__(self, bucket: str, prefix: str, client=None) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self._client = client

    def store_for(self, collection: str) -> S3MapStore:
        return S3MapStore(self.bucket, f"{self.prefix}{collection}.json", client=self._client)

    def list_collections(self) -> list[str]:
        client = self._s3()
        names: list[str] = []
        token: str | None = None
        while True:
            request: dict[str, object] = {
                "Bucket": self.bucket,
                "Prefix": self.prefix,
            }
            if token is not None:
                request["ContinuationToken"] = token
            try:
                response = client.list_objects_v2(**request)
            except _CLIENT_ERRORS as exc:
                raise MapUnavailable("map store unavailable") from exc
            except _NETWORK_ERRORS as exc:
                raise MapUnavailable("map store unavailable") from exc
            for item in response.get("Contents", ()):
                key = str(item.get("Key", ""))
                if not key.startswith(self.prefix):
                    continue
                tail = key[len(self.prefix) :]
                if "/" in tail or not tail.endswith(".json"):
                    continue
                names.append(tail[:-5])
            if not response.get("IsTruncated"):
                return sorted(set(names))
            token = response.get("NextContinuationToken")

    def _s3(self):
        if self._client is None:
            self._client = _s3_client()
        return self._client


_SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*$")
_SCHEME_OPENERS: dict[str, Callable[[str], MapFolder]] = {}


def register_map_folder_scheme(
    scheme: str, opener: Callable[[str], MapFolder]
) -> None:
    """Register a custom URI scheme opener for open_map_folder."""
    normalized = scheme.strip().lower().rstrip(":/")
    if not _SCHEME_PATTERN.match(normalized):
        raise ValueError(f"invalid URI scheme: {scheme!r}")
    _SCHEME_OPENERS[normalized] = opener


def unregister_map_folder_scheme(scheme: str) -> None:
    """Remove a previously registered URI scheme opener."""
    normalized = scheme.strip().lower().rstrip(":/")
    _SCHEME_OPENERS.pop(normalized, None)


def list_map_collections(folder: MapFolder) -> list[str]:
    """Return sorted collection names present in folder."""
    lister = getattr(folder, "list_collections", None)
    if lister is None:
        raise ShardingError("map folder does not support listing collections")
    return sorted(set(lister()))


def open_map_folder(uri: str) -> MapFolder:
    if "://" in uri:
        scheme = uri.split("://", 1)[0].lower()
        opener = _SCHEME_OPENERS.get(scheme)
        if opener is not None:
            return opener(uri)
    if uri.startswith("s3://"):
        return _s3_folder(uri)
    return FileMapFolder(_local_path(uri))


def folder_from_uri(map_uri: str | None) -> MapFolder | None:
    uri = map_uri if map_uri is not None else os.environ.get("MAP_URI")
    if not uri:
        return None
    return open_map_folder(uri)


def _s3_folder(uri: str) -> S3MapFolder:
    rest = uri[len("s3://") :]
    bucket, _separator, prefix = rest.partition("/")
    if bucket == "":
        raise ValueError(f"bad s3 uri {uri}")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return S3MapFolder(bucket, prefix)


def _local_path(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return uri
    raw = unquote(parsed.path)
    if os.name != "nt":
        raw = _drop_drive_letter(raw)
    return url2pathname(raw)


def _drop_drive_letter(path: str) -> str:
    # file:///C:/x/map.json is /x/map.json on POSIX.
    if (
        len(path) >= 4
        and path[0] == "/"
        and path[1].isalpha()
        and path[2] == ":"
        and path[3] == "/"
    ):
        return path[3:]
    return path


def _s3_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ShardingError(
            "S3 map store needs boto3. Install qdrant-custom-sharding-sdk[s3]."
        ) from exc

    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        connect_timeout=_S3_CONNECT_TIMEOUT,
        read_timeout=_S3_READ_TIMEOUT,
        retries={"total_max_attempts": _S3_MAX_ATTEMPTS, "mode": "standard"},
    )
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        return boto3.client("s3", config=config)
    # S3-compatible servers ignore region. Botocore still needs one.
    if os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION"):
        return boto3.client("s3", endpoint_url=endpoint, config=config)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        config=config,
    )


def _missing(exc: Any) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    # NoSuchBucket is also HTTP 404. That is an outage, not a missing map.
    if code == "NoSuchBucket":
        return False
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NoSuchKey", "404", "NotFound"} or status == 404


def _conflict(exc: Any) -> bool:
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status in {412, 409}


def _open_lock(path: Path) -> int:
    if os.name != "nt":
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    return _open_lock_windows(path)


def _open_lock_windows(path: Path) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    handle = kernel.CreateFileW(
        str(path),
        0x80000000 | 0x40000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        1,
        0x80,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        err = ctypes.get_last_error()
        if err == 80:
            raise FileExistsError(err, "lock exists", str(path))
        raise OSError(err, "cannot create lock", str(path))
    return msvcrt.open_osfhandle(handle, os.O_RDWR)


def _stale_lock(path: Path) -> bool:
    try:
        modified = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return time.time() - modified > _STALE_LOCK_SECONDS


def _clear_stale_lock(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def _parse_map(text: str) -> MapDoc:
    try:
        return map_from_json(text)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise MapUnavailable("map data is invalid") from exc
