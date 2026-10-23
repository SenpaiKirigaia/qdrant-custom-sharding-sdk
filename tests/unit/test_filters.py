from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from qdrant_custom_sharding.filters import feature_values


def test_none_and_non_filters_have_no_feature_values():
    assert feature_values(None, "feature") is None
    assert feature_values("feature", "feature") is None
    assert feature_values(Filter(), "feature") is None


def test_match_value_returns_one_value():
    flt = Filter(must=FieldCondition(key="feature", match=MatchValue(value="books")))
    assert feature_values(flt, "feature") == ["books"]
    listed = Filter(
        must=[FieldCondition(key="feature", match=MatchValue(value=7))]
    )
    assert feature_values(listed, "feature") == [7]


def test_match_any_returns_the_list():
    flt = Filter(
        must=[
            FieldCondition(key="color", match=MatchValue(value="red")),
            FieldCondition(key="feature", match=MatchAny(any=["books", "toys"])),
        ]
    )
    assert feature_values(flt, "feature") == ["books", "toys"]


def test_nested_filter_is_searched_and_the_first_match_wins():
    flt = Filter(
        must=[
            Filter(
                must=FieldCondition(key="feature", match=MatchValue(value="inner"))
            ),
            FieldCondition(key="feature", match=MatchValue(value="outer")),
        ]
    )
    assert feature_values(flt, "feature") == ["inner"]


def test_should_and_must_not_are_ignored():
    flt = Filter(
        should=FieldCondition(key="feature", match=MatchValue(value="books")),
        must_not=FieldCondition(key="feature", match=MatchAny(any=["toys"])),
        must=FieldCondition(key="color", match=MatchValue(value="red")),
    )
    assert feature_values(flt, "feature") is None
