class BiserachTerminator:
    def __init__(self, config):
        self.config = config

    def __call__(self, history: list[(int, float)], high: int):
        from pprint import pprint

        pprint(history)
        print(f"{high=}")
        return bi_search_terminator(history, high)


def bi_search_terminator(history: list[(int, float)], high: int):
    """Stops when the search interval converges (range is 0 or 1)."""
    if len(history) == 1 and history[0][0] == high and not history[0][1]:
        return True
    if len(history) < 2:
        return False
    return abs(history[-1][0] - history[-2][0]) <= 1
