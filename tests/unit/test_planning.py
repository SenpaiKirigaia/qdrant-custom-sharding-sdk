from qdrant_custom_sharding.planning import plan_placement


def test_plan_placement_empty_input():
    assert plan_placement({}, ["s0", "s1"]) == {}
    assert plan_placement({}, []) == {}


def test_plan_placement_splits_a_feature_bigger_than_target():
    placement = plan_placement({"a": 3, "b": 1}, ["s0", "s1"])
    assert placement == {"a": ["s0", "s1"], "b": ["s1"]}


def test_plan_placement_breaks_ties_by_name_then_slot_order():
    placement = plan_placement({"b": 5, "a": 5}, ["s0", "s1"])
    assert placement == {"a": ["s0"], "b": ["s1"]}


def test_plan_placement_balances_skewed_counts():
    counts = {
        "big": 9000,
        "f02": 2000,
        "f03": 800,
        "f04": 400,
        "f05": 200,
        "f06": 100,
        "f07": 50,
        "f08": 20,
        "f09": 10,
        "f10": 5,
    }
    slots = [f"s{index}" for index in range(8)]
    first = plan_placement(counts, slots)
    assert first == plan_placement(counts, slots)
    reversed_counts = dict(reversed(list(counts.items())))
    assert first == plan_placement(reversed_counts, slots)
    assert len(first["big"]) > 1
    assert len(first["big"]) < len(slots)
    load = _spread_load(first, counts, slots)
    average = sum(counts.values()) / len(slots)
    assert max(load.values()) <= 1.5 * average


def _spread_load(
    placement: dict[str, list[str]],
    counts: dict[str, int],
    slots: list[str],
) -> dict[str, int]:
    load = {slot: 0 for slot in slots}
    for feature, chosen in placement.items():
        base, extra = divmod(counts[feature], len(chosen))
        for index, slot in enumerate(chosen):
            load[slot] += base + (1 if index < extra else 0)
    return load
