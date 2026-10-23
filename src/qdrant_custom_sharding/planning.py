from math import ceil


def plan_placement(counts: dict[str, int], slots: list[str]) -> dict[str, list[str]]:
    if not counts:
        return {}
    if not slots:
        raise ValueError("slots must not be empty")
    target = max(1, ceil(sum(counts.values()) / len(slots)))
    load = {slot: 0 for slot in slots}
    order = {slot: index for index, slot in enumerate(slots)}
    placement: dict[str, list[str]] = {}
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    for feature, count in ranked:
        width = 1
        if count > target:
            width = min(len(slots), ceil(count / target))
        chosen = _least_loaded(load, order, width)
        placement[feature] = chosen
        _add_load(load, chosen, count)
    return placement


def _least_loaded(
    load: dict[str, int],
    order: dict[str, int],
    width: int,
) -> list[str]:
    ranked = sorted(load, key=lambda slot: (load[slot], order[slot]))
    return ranked[:width]


def _add_load(load: dict[str, int], chosen: list[str], count: int) -> None:
    base, extra = divmod(count, len(chosen))
    for index, slot in enumerate(chosen):
        load[slot] += base + (1 if index < extra else 0)
