from itertools import groupby


def compose_subgraph_ranges(
    child_subgraph_ranges: dict[str, list[(int, int)]],
    parent_subgraph_ranges: dict[str, dict[str, list[(int, int)]]],
) -> dict[str, list[(int, int)]]:
    def has_parent(rel_model_path):
        return len(parent_subgraph_ranges.get(rel_model_path, {})) > 0

    def get_belonging_ranges(rel_model_path, ranges):
        if not has_parent(rel_model_path):
            yield rel_model_path, ranges
            return
        for parent_rel_model_path, parent_ranges in parent_subgraph_ranges[
            rel_model_path
        ].items():
            yield parent_rel_model_path, flatmap_subgraph_ranges(ranges, parent_ranges)

    ret_rel_model_path_and_ranges = [
        pair
        for parent_rel_model_path, ranges_in_parent in child_subgraph_ranges.items()
        for pair in get_belonging_ranges(parent_rel_model_path, ranges_in_parent)
    ]
    return dict(merge_subgraph_ranges_list(ret_rel_model_path_and_ranges))


def merge_subgraph_ranges(
    lhs_subgraph_ranges: dict[str, list[(int, int)]],
    rhs_subgraph_ranges: dict[str, list[(int, int)]],
) -> dict[str, list[(int, int)]]:
    return dict(
        merge_subgraph_ranges_list(
            [
                *lhs_subgraph_ranges.items(),
                *rhs_subgraph_ranges.items(),
            ]
        )
    )


def merge_subgraph_ranges_list(
    rel_model_path_and_ranges: list[(str, list[(int, int)])]
) -> list[(str, list[(int, int)])]:
    rel_model_path_and_ranges = sorted(
        rel_model_path_and_ranges, key=lambda pair: pair[0]
    )
    rel_model_path_and_groups = groupby(
        rel_model_path_and_ranges, key=lambda pair: pair[0]
    )

    def merge_all_ranges_in_group(group):
        return sorted(
            set(
                tuple(belong_range)
                for _, belong_ranges in group
                for belong_range in belong_ranges
            )
        )

    merged_rel_model_path_and_ranges = [
        (rel_model_path, merge_all_ranges_in_group(group))
        for rel_model_path, group in rel_model_path_and_groups
    ]
    return merged_rel_model_path_and_ranges


def flatmap_subgraph_ranges(
    subgraph_ranges: list[(int, int)], parent_subgraph_ranges: list[(int, int)]
) -> list[(int, int)]:
    assert len(subgraph_ranges) > 0
    assert len(parent_subgraph_ranges) > 0
    subgraph_ranges_lengths = set(end - start for start, end in subgraph_ranges)
    parent_subgraph_ranges_lengths = set(
        end - start for start, end in parent_subgraph_ranges
    )
    assert len(subgraph_ranges_lengths) == 1
    assert len(parent_subgraph_ranges_lengths) == 1
    subgraph_size = next(iter(subgraph_ranges_lengths))
    parent_subgraph_size = next(iter(parent_subgraph_ranges_lengths))
    assert subgraph_size <= parent_subgraph_size
    return sorted(
        set(
            (parent_range_start + range_start, parent_range_start + range_end)
            for range_start, range_end in subgraph_ranges
            for parent_range_start, _ in parent_subgraph_ranges
        )
    )
