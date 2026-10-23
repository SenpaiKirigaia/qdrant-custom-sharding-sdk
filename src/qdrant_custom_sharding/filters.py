from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


def feature_values(flt: object, field: str) -> list[int | str] | None:
    if not isinstance(flt, Filter):
        return None
    return _match_must(flt.must, field)


def _match_must(must: object, field: str) -> list[int | str] | None:
    if must is None:
        return None
    conditions = must if isinstance(must, list) else [must]
    for condition in conditions:
        found = _match_condition(condition, field)
        if found is not None:
            return found
    return None


def _match_condition(condition: object, field: str) -> list[int | str] | None:
    if isinstance(condition, Filter):
        return _match_must(condition.must, field)
    if not isinstance(condition, FieldCondition):
        return None
    if condition.key != field:
        return None
    return _match_values(condition.match)


def _match_values(match: object) -> list[int | str] | None:
    if isinstance(match, MatchValue):
        return [match.value]
    if isinstance(match, MatchAny):
        return list(match.any)
    return None
