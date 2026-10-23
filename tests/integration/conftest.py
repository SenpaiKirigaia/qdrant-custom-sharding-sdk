import socket
import uuid

import pytest

from qdrant_custom_sharding.map_stores import _s3_client

pytestmark = pytest.mark.integration

_BUCKET = "sharding-maps"


def _minio_is_up() -> bool:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("localhost", 9000))
    except OSError:
        return False
    finally:
        sock.close()
    return True


@pytest.fixture(scope="session", autouse=True)
def minio_reachable():
    if not _minio_is_up():
        pytest.skip("run `make up` first")


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def object_key(aws_env):
    key = f"tests/{uuid.uuid4()}.json"
    client = _s3_client()
    yield key
    client.delete_object(Bucket=_BUCKET, Key=key)
