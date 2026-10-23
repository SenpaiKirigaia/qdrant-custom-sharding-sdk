import linecache
import warnings

from qdrant_custom_sharding.sharding_warnings import (
    FanOutWarning,
    FeatureMovedWarning,
    ShardingWarning,
    StaleMapWarning,
    warn,
)


def test_warning_classes_extend_sharding_warning():
    assert issubclass(ShardingWarning, UserWarning)
    assert issubclass(FanOutWarning, ShardingWarning)
    assert issubclass(FeatureMovedWarning, ShardingWarning)
    assert issubclass(StaleMapWarning, ShardingWarning)


def test_warn_is_silent_when_disabled():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn(FanOutWarning, "fan-out", False)
    assert caught == []


def _helper(enabled: bool) -> None:
    warn(FanOutWarning, "fan-out", enabled)


def _override(enabled: bool) -> None:
    _helper(enabled)


def test_warn_points_at_the_caller_of_the_override():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _override(True)
    assert len(caught) == 1
    record = caught[0]
    assert record.category is FanOutWarning
    assert str(record.message) == "fan-out"
    text = linecache.getline(record.filename, record.lineno)
    assert "_override(True)" in text
