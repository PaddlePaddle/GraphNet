# filter_valid_groups.py
from typing import Set, List, Callable

SampleUid = str
SampleType = str
Group = Set[SampleUid]  # Original group: set of sample UIDs


def filter_valid_groups(
    groups: List[Group], sample_type_getter: Callable[[SampleUid], SampleType]
) -> List[Group]:
    """
    Filter valid groups: all samples in a group must have the same type.

    Args:
        groups: List of original groups, each a set of sample UIDs.
        sample_type_getter: Function that returns the type string for a given sample UID.

    Returns:
        List of valid groups, each still as a set of sample UIDs (no wrapping).
    """
    valid_groups = []
    for group in groups:
        # Collect the types of all samples in the group
        types_in_group = {sample_type_getter(uid) for uid in group}
        # Keep the group only if the set of types has exactly one element
        if len(types_in_group) == 1:
            valid_groups.append(group)
    return valid_groups


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------

    # Sample type mapping (simulates a database)
    sample_types = {
        "s1": "typeA",
        "s2": "typeA",
        "s3": "typeB",
        "s4": "typeB",
        "s5": "typeC",
        "s6": "typeC",
        "s7": "typeA",
        "s8": "typeB",
        "empty_sample": "typeX",
    }

    def getter(uid: str) -> str:
        return sample_types[uid]

    # Test 1: All groups valid
    groups1 = [{"s1", "s2"}, {"s3", "s4"}, {"s5"}]  # s5 alone is valid
    expected1 = [{"s1", "s2"}, {"s3", "s4"}, {"s5"}]
    result1 = filter_valid_groups(groups1, getter)
    assert result1 == expected1, f"Test 1 failed: {result1}"

    # Test 2: Mixed valid/invalid
    groups2 = [
        {"s1", "s2"},  # valid (typeA)
        {"s1", "s3"},  # invalid (typeA + typeB)
        {"s5", "s6"},  # valid (typeC)
        {"s7", "s8"},  # invalid (typeA + typeB)
        set(),  # empty group -> invalid (len(types)=0)
    ]
    expected2 = [{"s1", "s2"}, {"s5", "s6"}]
    result2 = filter_valid_groups(groups2, getter)
    assert result2 == expected2, f"Test 2 failed: {result2}"

    # Test 3: No valid groups
    groups3 = [{"s1", "s3"}, {"s2", "s4", "s5"}, {"s1", "s2", "s3"}]
    expected3 = []
    result3 = filter_valid_groups(groups3, getter)
    assert result3 == expected3, f"Test 3 failed: {result3}"

    # Test 4: Single element groups (always valid)
    groups4 = [{"s1"}, {"s3"}, {"s5"}, {"s7"}]
    expected4 = [{"s1"}, {"s3"}, {"s5"}, {"s7"}]
    result4 = filter_valid_groups(groups4, getter)
    assert result4 == expected4, f"Test 4 failed: {result4}"

    # Test 5: Case sensitivity (if types are case‑sensitive)
    sample_types_case = {"sA": "type", "sB": "TYPE"}

    def getter_case(uid):
        return sample_types_case[uid]

    groups5 = [{"sA", "sB"}]  # different strings -> invalid
    expected5 = []
    result5 = filter_valid_groups(groups5, getter_case)
    assert result5 == expected5, f"Test 5 failed: {result5}"

    # Test 6: Large group with identical type
    large_group = {f"x{i}" for i in range(100)}
    # Assign same type to all members
    for uid in large_group:
        sample_types[uid] = "uniform"
    groups6 = [large_group]
    expected6 = [large_group]
    result6 = filter_valid_groups(groups6, getter)
    assert result6 == expected6, f"Test 6 failed: {result6}"

    print("All tests passed.")
