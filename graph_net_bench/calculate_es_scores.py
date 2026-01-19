import json
import itertools
from typing import Callable
import argparse
from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory
from graph_net_bench import analysis_util
from graph_net_bench import verify_aggregated_params
from graph_net_bench.positive_tolerance_interpretation_manager import (
    get_supported_positive_tolerance_interpretation_types,
    get_positive_tolerance_interpretation,
)


class ESScoresWrapper:
    """Wrapper for es_scores dict to allow attribute assignment."""

    def __init__(self, es_scores_dict):
        self._dict = es_scores_dict
        self._aggregated_results = {}

    def items(self):
        return self._dict.items()

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value


def es_result_checker(
    es_from_microscopic: float, es_from_macro: float, atol: float, rtol: float
) -> bool:
    """
    Check if ES(t) values from microscopic and macro calculations match.

    Args:
        es_from_microscopic: ES(t) value from microscopic-level calculation
        es_from_macro: ES(t) value from aggregated-level calculation
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        True if values match within tolerance, False otherwise
    """
    return np.allclose(es_from_microscopic, es_from_macro, rtol=rtol, atol=atol)


def compare_aggregated_es_and_microscopic_es(
    tolerance: int,
    microscopic_es: float,
    aggregated_es: float | None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> tuple[bool, float, float]:
    """
    Compare ES(t) values from aggregated and microscopic calculations at a tolerance level.

    Args:
        tolerance: Tolerance level t
        microscopic_es: ES(t) value from microscopic-level calculation
        aggregated_es: ES(t) value from aggregated-level calculation, or None if missing
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        Tuple of (is_matched, diff, relative_diff)
    """
    if aggregated_es is None:
        return False, 0.0, 0.0

    diff = abs(microscopic_es - aggregated_es)
    relative_diff = diff / max(abs(microscopic_es), abs(aggregated_es), 1e-10)
    is_matched = es_result_checker(microscopic_es, aggregated_es, atol, rtol)

    return is_matched, diff, relative_diff


def print_verification_result(
    tolerance: int,
    microscopic_es: float,
    aggregated_es: float | None,
    diff: float,
    relative_diff: float,
    is_matched: bool,
) -> None:
    """Print verification result for a single tolerance level."""
    if aggregated_es is None:
        print(f"ERROR: No aggregated result for t={tolerance}, cannot verify")
    elif is_matched:
        print(
            f"t={tolerance:3d}: MATCHED  - Microscopic: {microscopic_es:.6f}, Aggregated: {aggregated_es:.6f}, Diff: {diff:.2e}"
        )
    else:
        print(
            f"t={tolerance:3d}: MISMATCH - Microscopic: {microscopic_es:.6f}, Aggregated: {aggregated_es:.6f}, Diff: {diff:.2e} ({relative_diff * 100:.4f}%)"
        )


def get_verified_aggregated_es_values(es_scores: dict, folder_name: str) -> dict:
    """
    Get verified ES(t) values by checking consistency between aggregated and microscopic-level calculations.

    Args:
        es_scores: Dictionary of ES(t) scores from microscopic-level calculation
        folder_name: Name of the folder being verified

    Returns:
        Dictionary of verified ES(t) values (only matched tolerance levels).

    Raises:
        AssertionError: If aggregated and microscopic results do not match (fail-fast).
    """
    aggregated_results = getattr(es_scores, "_aggregated_results", {})
    verified_es_values = {}
    mismatches = []

    print(f"\n{'=' * 80}")
    print(f"Verifying Aggregated/Microscopic Consistency for '{folder_name}'")
    print(f"{'=' * 80}")

    for tolerance, microscopic_es in es_scores.items():
        aggregated_es = aggregated_results.get(tolerance)
        is_matched, diff, relative_diff = compare_aggregated_es_and_microscopic_es(
            tolerance, microscopic_es, aggregated_es
        )

        print_verification_result(
            tolerance,
            microscopic_es,
            aggregated_es,
            diff,
            relative_diff,
            is_matched,
        )

        if aggregated_es is None:
            mismatches.append(
                f"t={tolerance}: Missing aggregated result (microscopic={microscopic_es:.6f})"
            )
        elif not is_matched:
            mismatches.append(
                f"t={tolerance}: Mismatch - Microscopic={microscopic_es:.6f}, "
                f"Aggregated={aggregated_es:.6f}, Diff={diff:.2e} ({relative_diff * 100:.4f}%)"
            )
        else:
            verified_es_values[tolerance] = microscopic_es

    if mismatches:
        error_msg = (
            f"\n{'=' * 80}\n"
            f"ERROR: Aggregated and microscopic results do not match for '{folder_name}'!\n"
            f"{'=' * 80}\n"
            f"Mismatches:\n"
            + "\n".join(f"  - {mismatch}" for mismatch in mismatches)
            + f"\n\nCalculation validation failed. Please verify the calculation logic "
            f"using verify_aggregated_params.py\n"
            f"{'=' * 80}\n"
        )
        print(error_msg)
        raise AssertionError(error_msg)

    print(
        f"\nSUCCESS: All aggregated and microscopic results match for '{folder_name}'."
    )
    print(f"{'=' * 80}\n")
    return verified_es_values


def calculate_es_scores_for_each_model_path(
    log_content: str,
    get_model_path_for_each_log_line: Callable[[str], str | None],
    interpretation_type: str = "default",
    negative_speedup_penalty: float = 0.0,
    fpdb: float = 0.1,
    enable_aggregation_mode: bool = True,
) -> dict[str, dict[int, float]]:
    """
    Groups log content by model path using accumulate and groupby,
    then calculates ES scores for each group.
    """
    lines = log_content.splitlines()

    # 1. Get f(line) = 1 if get_model_path_for_each_log_line(line) is not None else 0
    line_indicators = [
        1 if get_model_path_for_each_log_line(line) is not None else 0 for line in lines
    ]

    # 2. Use itertools.accumulate to get the cumulative sum (cumsum) of indicators
    cum_sums = list(itertools.accumulate(line_indicators))

    # 3. Use itertools.groupby to group lines based on the cumsum
    # 4. Adjust results to get log_contents grouped by model_path
    model_paths = []
    log_contents_list = []

    for _, group in itertools.groupby(zip(cum_sums, lines), key=lambda x: x[0]):
        group_lines = [item[1] for item in group]
        if not group_lines:
            continue

        # Extract the model_path from the first line of the group
        path = get_model_path_for_each_log_line(group_lines[0])
        if path is not None:
            model_paths.append(path)
            # Combine the lines belonging to this model path back into a single string
            log_contents_list.append("\n".join(group_lines))

    assert len(model_paths) == len(log_contents_list)

    # 5. Call the lower-level function calculate_es_scores_for_log_contents
    # Note: This function returns list[dict[int, float]] based on your instruction
    es_scores_list = calculate_es_scores_for_log_contents(
        log_contents=log_contents_list,
        interpretation_type=interpretation_type,
        negative_speedup_penalty=negative_speedup_penalty,
        fpdb=fpdb,
        enable_aggregation_mode=enable_aggregation_mode,
    )

    # 6. Organize return results with model_path as key
    # Returns dict[str, dict[int, float]]
    return {model_paths[i]: es_scores_list[i] for i in range(len(model_paths))}


def calculate_es_scores_for_log_contents(
    log_contents: list[str],
    interpretation_type: str = "default",
    negative_speedup_penalty: float = 0.0,
    fpdb: float = 0.1,
    enable_aggregation_mode: bool = True,
) -> list[dict[int, float]]:
    """
    Wraps raw log contents into temporary files to compute Error Sign (ES) scores.
    """

    def _write_logs_to_dir(target_dir: str):
        """Write each log content into tmp_dir/{i}.log."""
        for i, content in enumerate(log_contents):
            (Path(target_dir) / f"{i}.log").write_text(content, encoding="utf-8")

    with TemporaryDirectory() as tmp_dir:
        # Step 1: Create a temporary directory and write logs
        _write_logs_to_dir(tmp_dir)

        # Step 2: Get index_str2es_scores from the file-based utility
        index_str2es_scores = calculate_es_scores_for_log_file_or_dir(
            benchmark_path=tmp_dir,
            interpretation_type=interpretation_type,
            negative_speedup_penalty=negative_speedup_penalty,
            fpdb=fpdb,
            enable_aggregation_mode=enable_aggregation_mode,
        )

        # Step 3: Convert keys of index_str2es_scores to int to get index2es_scores
        index2es_scores = {int(k): v for k, v in index_str2es_scores.items()}

        # Step 4: Convert index2es_scores to list to get es_scores_list
        # Ensuring the order matches the original log_contents indices
        es_scores_list = [index2es_scores[i] for i in range(len(log_contents))]

        # Step 5: Return es_scores_list
        return es_scores_list


def calculate_es_scores_for_log_file_or_dir(
    benchmark_path: str,
    interpretation_type: str = "default",
    negative_speedup_penalty: float = 0.0,
    fpdb: float = 0.1,
    enable_aggregation_mode: bool = True,
):
    # 1. Scan folders to get data
    all_results = analysis_util.scan_all_folders(benchmark_path)
    if not all_results:
        print("No valid data found. Exiting.")
        return {}

    # 2. Calculate scores for each curve and verify aggregated/microscopic consistency
    all_es_scores = {}
    all_aggregated_results = {}
    positive_tolerance_interpretation = get_positive_tolerance_interpretation(
        interpretation_type
    )

    for folder_name, samples in all_results.items():
        print(f"{folder_name=}")
        print(f"\nCalculating ESt scores for '{folder_name}'...")

        es_scores = analysis_util.calculate_scores(
            samples,
            p=negative_speedup_penalty,
            b=fpdb,
            type="ESt",
            positive_tolerance_interpretation=positive_tolerance_interpretation,
        )

        # Keep original behavior: assign es_scores directly
        all_es_scores[folder_name] = es_scores

        # Verify aggregated/microscopic consistency if aggregation mode is enabled
        if enable_aggregation_mode:
            # Calculate aggregated results and attach to es_scores
            aggregated_results = (
                verify_aggregated_params.verify_es_constructor_params_across_tolerances(
                    samples,
                    folder_name,
                    negative_speedup_penalty=negative_speedup_penalty,
                    fpdb=fpdb,
                    positive_tolerance_interpretation=positive_tolerance_interpretation,
                )
            )
            # Store aggregated results
            all_aggregated_results[folder_name] = aggregated_results

            # Extract expected_es values and attach as _aggregated_results
            # Wrap es_scores to allow attribute assignment
            es_scores_wrapper = ESScoresWrapper(es_scores)
            es_scores_wrapper._aggregated_results = {
                tolerance: result["expected_es"]
                for tolerance, result in aggregated_results.items()
            }

            # Fail-fast: raise AssertionError if validation fails
            verified_es_values = get_verified_aggregated_es_values(
                es_scores_wrapper, folder_name
            )
            all_es_scores[folder_name] = verified_es_values
    return all_es_scores


def main(args):
    all_es_scores = calculate_es_scores_for_log_file_or_dir(
        benchmark_path=args.benchmark_path,
        interpretation_type=args.interpretation_type,
        negative_speedup_penalty=args.negative_speedup_penalty,
        fpdb=args.fpdb,
        enable_aggregation_mode=args.enable_aggregation_mode,
    )
    assert len(all_es_scores) == 1
    with open(args.output_json_file_path, "w") as f:
        json.dump(next(iter(all_es_scores.items()))[1], f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and aggregate ES(t) scores from benchmark results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Path to the benchmark log file or directory containing benchmark JSON files or sub-folders.",
    )
    parser.add_argument(
        "--output-json-file-path",
        type=str,
        required=True,
        help="json file path for saving the aggregated score",
    )
    parser.add_argument(
        "--negative-speedup-penalty",
        type=float,
        default=0.0,
        help="Penalty power (p) for negative speedup. Formula: speedup**(p+1). Default: 0.0.",
    )
    parser.add_argument(
        "--fpdb",
        type=float,
        default=0.1,
        help="Base penalty for severe errors (e.g., crashes, correctness failures).",
    )
    parser.add_argument(
        "--enable-aggregation-mode",
        action="store_true",
        help="Enable aggregation mode to verify aggregated/microscopic consistency. Default: enabled.",
    )
    parser.add_argument(
        "--disable-aggregation-mode",
        dest="enable_aggregation_mode",
        action="store_false",
        help="Disable aggregation mode verification.",
    )
    parser.add_argument(
        "--positive-tolerance-interpretation",
        dest="interpretation_type",
        choices=get_supported_positive_tolerance_interpretation_types(),
        default="default",
        help="Select how positive tolerance values are interpreted into error types.",
    )
    parser.set_defaults(enable_aggregation_mode=True)
    args = parser.parse_args()
    main(args)
