import os
import socket
import uuid

import pytest

from qdrant_custom_sharding import ShardedQdrantClient
from tests.local_qdrant import local_kwargs

_BUCKET = "sharding-maps"


def _port_open(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


@pytest.fixture(scope="session", autouse=True)
def local_stack():
    if not _port_open(6333):
        pytest.skip("Qdrant is not running on localhost:6333")
    if not _port_open(9000):
        pytest.skip("MinIO is not running on localhost:9000")


@pytest.fixture(scope="session", autouse=True)
def aws_env():
    keys = {
        "AWS_ENDPOINT_URL": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin123",
        "AWS_DEFAULT_REGION": "us-east-1",
    }
    saved = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def open_client(map_uri: str, **kwargs: object) -> ShardedQdrantClient:
    return ShardedQdrantClient(map_uri=map_uri, **local_kwargs(timeout=120), **kwargs)


@pytest.fixture
def cluster(aws_env):
    prefix = f"e2e-{uuid.uuid4()}/"
    map_uri = f"s3://{_BUCKET}/{prefix}"
    client = open_client(map_uri)
    created: list[str] = []
    try:
        yield client, map_uri, created
    finally:
        try:
            for name in created:
                if client.collection_exists(name):
                    client.delete_collection(name)
        finally:
            _delete_prefix(prefix)
            client.close()


def _delete_prefix(prefix: str) -> None:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    token = None
    while True:
        kwargs = {"Bucket": _BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=_BUCKET, Delete={"Objects": keys})
        if not page.get("IsTruncated"):
            return
        token = page["NextContinuationToken"]
