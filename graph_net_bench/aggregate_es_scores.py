import json
import argparse
import numpy as np
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


def main(args):
    # 1. Scan folders to get data
    all_results = analysis_util.scan_all_folders(args.benchmark_path)
    if not all_results:
        print("No valid data found. Exiting.")
        return

    # 2. Calculate scores for each curve and verify aggregated/microscopic consistency
    all_es_scores = {}
    all_aggregated_results = {}
    positive_tolerance_interpretation = get_positive_tolerance_interpretation(
        args.interpretation_type
    )

    for folder_name, samples in all_results.items():
        print(f"\nCalculating ESt scores for '{folder_name}'...")

        es_scores = analysis_util.calculate_scores(
            samples,
            p=args.negative_speedup_penalty,
            b=args.fpdb,
            type="ESt",
            positive_tolerance_interpretation=positive_tolerance_interpretation,
        )

        # Keep original behavior: assign es_scores directly
        all_es_scores[folder_name] = es_scores

        # Verify aggregated/microscopic consistency if aggregation mode is enabled
        if args.enable_aggregation_mode:
            # Calculate aggregated results and attach to es_scores
            aggregated_results = (
                verify_aggregated_params.verify_es_constructor_params_across_tolerances(
                    samples,
                    folder_name,
                    negative_speedup_penalty=args.negative_speedup_penalty,
                    fpdb=args.fpdb,
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

    weights = get_weights()
    assert set(weights.keys()) == set(all_es_scores["validation"].keys())
    weighted_sum = sum(
        weight * np.log(score) / np.log(10)
        for tolerance in weights.keys()
        for weight in [weights[tolerance]]
        for score in [all_es_scores["validation"][tolerance]]
    )
    result = {
        "id": args.sample_id,
        "score": float(weighted_sum),
    }
    with open(args.output_json_file_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"{weighted_sum=}")
    print(f"Result is saved to {args.output_json_file_path}")


def get_weights():
    # `weights` is derived from the NLP ES metrics of NVIDIA A100 relative to H20
    weights = {
        -10: np.float64(0.001),
        -9: np.float64(0.001),
        -8: np.float64(0.001),
        -7: np.float64(0.13),
        -6: np.float64(0.40),
        -5: np.float64(0.48),
        -4: np.float64(0.48),
        -3: np.float64(0.48),
        -2: np.float64(0.48),
        -1: np.float64(0.48),
        0: np.float64(0.48),
        1: np.float64(0.48),
        2: np.float64(0.48),
        3: np.float64(0.48),
        4: np.float64(0.48),
    }
    sum_weights = sum(v for k, v in weights.items())
    return dict((k, v / sum_weights) for k, v in weights.items())


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
        "--sample-id",
        type=int,
        required=True,
        help="Sample Id",
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
