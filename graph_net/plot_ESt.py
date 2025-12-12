import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from graph_net import analysis_util
from graph_net import verify_aggregated_params
from graph_net.positive_tolerance_interpretation_manager import (
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


def plot_ES_results(s_scores: dict, args: argparse.Namespace):
    """
    Plot ES(t) curve
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    all_x_coords = []

    for idx, (folder_name, scores_dict) in enumerate(s_scores.items()):
        plot_points = []
        for (
            t_key,
            score_data,
        ) in scores_dict.items():
            if isinstance(score_data, dict):
                score = score_data["score"]
            else:
                score = score_data

            all_x_coords.append(t_key)
            plot_points.append({"x": t_key, "y": score})

        # Sort by x value
        plot_points.sort(key=lambda p: p["x"])

        x_vals = np.array([p["x"] for p in plot_points])
        y_vals = np.array([p["y"] for p in plot_points])

        color = colors[idx % len(colors)]

        # Find index where t=0
        zero_index = np.where(x_vals == 0)[0][0] if 0 in x_vals else None

        # If t=0 exists, plot in segments
        if zero_index is not None:
            # Plot continuous line for t <= 0
            ax.plot(
                x_vals[: zero_index + 1],
                y_vals[: zero_index + 1],
                "o-",
                color=color,
                label=folder_name,
                linewidth=2,
                markersize=6,
            )
            # Plot stepwise portion for t > 0
            ax.plot(
                x_vals[zero_index:],
                y_vals[zero_index:],
                "o-",
                color=color,
                linewidth=2,
                markersize=6,
                drawstyle="steps-post",
            )
        else:
            # If no t=0, plot the entire curve as a regular line
            ax.plot(
                x_vals,
                y_vals,
                "o-",
                color=color,
                label=folder_name,
                linewidth=2,
                markersize=6,
            )

    p = args.negative_speedup_penalty
    config = f"p = {p}, b = {args.fpdb}"
    fig.text(0.5, 0.9, config, ha="center", fontsize=16, style="italic")

    ax.set_xlabel("t", fontsize=18)
    ax.set_ylabel("ES(t)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)

    if all_x_coords:
        x_min = int(np.floor(min(all_x_coords)))
        x_max = int(np.ceil(max(all_x_coords)))
        ax.set_xticks(np.arange(x_min, x_max + 1))

    ax.xaxis.grid(True, which="major", lw=0.7, ls=":", color="grey", alpha=0.5)
    ax.yaxis.grid(True, which="major", lw=0.7, ls=":", color="grey", alpha=0.5)

    return fig, ax, all_x_coords


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
            # Store aggregated results for plotting
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

    # 3. Plot the results
    if any(all_es_scores.values()):
        os.makedirs(args.output_dir, exist_ok=True)
        fig, ax, all_x_coords = plot_ES_results(all_es_scores, args)

        # Manually add aggregated curves if available
        if args.enable_aggregation_mode and all_aggregated_results:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]

            for idx, (folder_name, aggregated_results) in enumerate(
                all_aggregated_results.items()
            ):
                if folder_name not in all_es_scores:
                    continue

                color = colors[idx % len(colors)]
                agg_plot_points = []
                for tolerance, result in aggregated_results.items():
                    if isinstance(result, dict) and "expected_es" in result:
                        agg_plot_points.append(
                            {"x": tolerance, "y": result["expected_es"]}
                        )

                if agg_plot_points:
                    agg_plot_points.sort(key=lambda p: p["x"])
                    agg_x_vals = np.array([p["x"] for p in agg_plot_points])
                    agg_y_vals = np.array([p["y"] for p in agg_plot_points])

                    agg_zero_index = (
                        np.where(agg_x_vals == 0)[0][0] if 0 in agg_x_vals else None
                    )

                    if agg_zero_index is not None:
                        ax.plot(
                            agg_x_vals[: agg_zero_index + 1],
                            agg_y_vals[: agg_zero_index + 1],
                            "s--",
                            color=color,
                            label=f"{folder_name} (aggregated)",
                            linewidth=2,
                            markersize=6,
                            alpha=0.7,
                        )
                        ax.plot(
                            agg_x_vals[agg_zero_index:],
                            agg_y_vals[agg_zero_index:],
                            "s--",
                            color=color,
                            linewidth=2,
                            markersize=6,
                            drawstyle="steps-post",
                            alpha=0.7,
                        )
                    else:
                        ax.plot(
                            agg_x_vals,
                            agg_y_vals,
                            "s--",
                            color=color,
                            label=f"{folder_name} (aggregated)",
                            linewidth=2,
                            markersize=6,
                            alpha=0.7,
                        )

            # Update x-axis range if needed
            if all_x_coords:
                for folder_name, aggregated_results in all_aggregated_results.items():
                    for tolerance in aggregated_results.keys():
                        all_x_coords.append(tolerance)
                x_min = int(np.floor(min(all_x_coords)))
                x_max = int(np.ceil(max(all_x_coords)))
                ax.set_xticks(np.arange(x_min, x_max + 1))

        # Always show legend (whether aggregated curves are added or not)
        ax.legend(fontsize=16, loc="best")

        output_file = os.path.join(args.output_dir, "ESt_result.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nComparison plot saved to {output_file}")
    else:
        print("No ES(t) scores were calculated. Skipping plot generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and plot ES(t) scores from benchmark results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Path to the benchmark log file or directory containing benchmark JSON files or sub-folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Output directory for saving the plot. Default: analysis_results",
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
