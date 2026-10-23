import warnings

_STACKLEVEL = 4


class ShardingWarning(UserWarning):
    pass


class FanOutWarning(ShardingWarning):
    """The call did not name one feature, so the client did extra work."""


class FeatureMovedWarning(ShardingWarning):
    """A point's feature value changed, so it was moved."""


class StaleMapWarning(ShardingWarning):
    """The map could not be reloaded, so a cached copy was used."""


def warn(
    category: type[Warning],
    message: str,
    enabled: bool,
    stacklevel: int = _STACKLEVEL,
) -> None:
    if not enabled:
        return
    warnings.warn(message, category, stacklevel=stacklevel)
