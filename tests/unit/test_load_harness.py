import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from qdrant_custom_sharding.errors import InvalidMove
from tests.load.harness import LoadHarnessRequest, _failure_kind, _open_client


def test_timeout_is_not_counted_as_an_http_error():
    assert _failure_kind(TimeoutError()) == "timeout"
    assert _failure_kind(httpx.TimeoutException("slow")) == "timeout"


def test_http_and_connection_failures_are_http_errors():
    response = UnexpectedResponse(500, "err", b"no", {})
    assert _failure_kind(response) == "http"
    assert _failure_kind(ConnectionError("reset")) == "http"
    assert _failure_kind(httpx.ConnectError("reset")) == "http"


def test_a_wrapped_timeout_stays_a_timeout():
    wrapped = ResponseHandlingException(TimeoutError())
    assert _failure_kind(wrapped) == "timeout"


def test_a_sharding_error_is_not_a_counted_failure():
    assert _failure_kind(InvalidMove("busy")) is None


def test_localhost_clients_keep_connections():
    client = _open_client(
        LoadHarnessRequest(
            url="http://localhost:6333",
            map_uri="maps",
            collection="unused",
            feature_field="feature",
            feature="load",
            point_count=1,
            workers=1,
            grace_seconds=0,
        )
    )
    try:
        pool = client._client.http.client._client._transport._pool
        assert pool._max_keepalive_connections == 8
    finally:
        client.close()
