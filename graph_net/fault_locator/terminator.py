def bi_search_terminator(self, history):
    """Stops when the search interval converges (range is 0 or 1)."""
    if len(history) < 2:
        return False
    return abs(history[-1][0] - history[-2][0]) <= 1
