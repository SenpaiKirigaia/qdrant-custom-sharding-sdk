import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--client-floor",
        default=None,
        help="Require qdrant_client.__version__ to start with this text",
    )


def pytest_configure(config):
    floor = config.getoption("--client-floor")
    if not floor:
        return
    import qdrant_client
    from importlib.metadata import version as package_version

    # 1.12.2 does not set qdrant_client.__version__. The distribution version does.
    version = getattr(qdrant_client, "__version__", None)
    if not isinstance(version, str) or not version:
        version = package_version("qdrant-client")
    print(f"qdrant_client {version}", flush=True)
    if not version.startswith(floor):
        raise pytest.UsageError(
            f"qdrant_client {version} does not start with {floor}"
        )
