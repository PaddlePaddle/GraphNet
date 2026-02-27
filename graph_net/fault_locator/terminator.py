class BiserachTerminator:
    def __init__(self, config):
        self.config = config

    def __call__(self, history: list[(int, float)], high: int):
        from pprint import pprint

        pprint(history)
        return bi_search_terminator(history, high)


def bi_search_terminator(history: list[(int, float)], high: int):
    """Stops when the search interval converges (range is 0 or 1)."""
    last_idx, is_broken = history[-1]
    if last_idx == 1 and is_broken:
        return True
    if last_idx == high and not is_broken:
        return True
    if len(history) == 1 and history[0][0] == high and not history[0][1]:
        return True
    if len(history) < 2:
        return False
    return abs(history[-1][0] - history[-2][0]) <= 1
