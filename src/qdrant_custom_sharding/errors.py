class ShardingError(Exception):
    pass


class WriteDenied(ShardingError):
    """A write cannot proceed for this feature."""


class VersionConflict(ShardingError):
    """The stored map version is not the one this save expected."""


class MapNotFound(ShardingError):
    """The map has not been saved yet."""


class MapUnavailable(ShardingError):
    """The map store cannot be read."""


class MoveNotReady(ShardingError):
    pass


class InvalidMove(ShardingError):
    pass
