import inspect

from qdrant_client import QdrantClient


def compatibility_kwargs() -> dict:
    """qdrant-client 1.12 has no check_compatibility and passes it on to httpx."""
    if "check_compatibility" in inspect.signature(QdrantClient.__init__).parameters:
        return {"check_compatibility": False}
    return {}


def local_kwargs(timeout: int = 60) -> dict:
    """Talk to localhost directly. A local HTTP proxy resets these connections."""
    kwargs = {
        "url": "http://localhost:6333",
        "timeout": timeout,
        "prefer_grpc": False,
        "trust_env": False,
    }
    kwargs.update(compatibility_kwargs())
    return kwargs
