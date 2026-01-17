from typing import Dict

# Type alias for clarity
# Key: Tolerance Level (int), Value: Error Score (float)
ES = Dict[int, float]


def has_fault_at(es_scores: ES, tolerance: int) -> bool:
    """
    Determines if a fault exists at a specific tolerance level.

    Logic:
    1. Asserts that (tolerance - 1) exists in the ES dictionary.
    2. Compares the score at (tolerance - 1) with the score at the maximum
       defined tolerance level in the ES dictionary.
    3. Returns True if the specific score is strictly less than the maximum score.
    """
    # Requirement 3: Ensure tolerance-1 is a valid key
    assert tolerance - 1 in es_scores, f"Tolerance index {tolerance-1} missing from ES"

    # Requirement 2: Compare ES[tolerance-1] with ES[max_tolerance]
    max_tolerance_key = max(es_scores.keys())

    # If the score at current tolerance-1 is less than the worst-case score,
    # it implies a sensitivity shift interpreted as a fault.
    return es_scores[tolerance - 1] < es_scores[max_tolerance_key]
