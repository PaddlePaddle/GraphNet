import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from graph_net import analysis_util


def compare_single_tolerance_level(
    tolerance_level: int,
    micro_es: float,
    aggregated_es: float | None,
    tolerance_threshold: float,
) -> tuple[bool, float, float]:
    """
    Compare micro and aggregated ES(t) values for a single tolerance level.

    Args:
        tolerance_level: Tolerance level t
        micro_es: ES(t) value from micro-level calculation
        aggregated_es: ES(t) value from aggregated-level calculation, or None if missing
        tolerance_threshold: Floating point comparison tolerance

    Returns:
        Tuple of (is_matched, diff, relative_diff)
    """
    if aggregated_es is None:
        return False, 0.0, 0.0

    diff = abs(micro_es - aggregated_es)
    relative_diff = diff / max(abs(micro_es), abs(aggregated_es), 1e-10)
    is_matched = diff < tolerance_threshold or relative_diff < tolerance_threshold

    return is_matched, diff, relative_diff


def print_verification_result(
    tolerance_level: int,
    micro_es: float,
    aggregated_es: float | None,
    diff: float,
    relative_diff: float,
    is_matched: bool,
) -> None:
    """Print verification result for a single tolerance level."""
    if aggregated_es is None:
        print(f"ERROR: No aggregated result for t={tolerance_level}, cannot verify")
    elif is_matched:
        print(
            f"t={tolerance_level:3d}: MATCHED  - Micro: {micro_es:.6f}, Aggregated: {aggregated_es:.6f}, Diff: {diff:.2e}"
        )
    else:
        print(
            f"t={tolerance_level:3d}: MISMATCH - Micro: {micro_es:.6f}, Aggregated: {aggregated_es:.6f}, Diff: {diff:.2e} ({relative_diff*100:.4f}%)"
        )


def verify_aggregated_micro_consistency(
    es_scores: dict, folder_name: str, tolerance_threshold: float
) -> tuple[dict, bool]:
    """
    Verify consistency between aggregated and micro-level ES(t) calculations.

    Args:
        es_scores: Dictionary of ES(t) scores from micro-level calculation
        folder_name: Name of the folder being verified
        tolerance_threshold: Floating point comparison tolerance

    Returns:
        Tuple of (verified_scores, all_matched):
        - verified_scores: Dictionary of verified scores (only matched tolerance levels)
        - all_matched: True if all tolerance levels matched, False otherwise
    """
    aggregated_results = getattr(es_scores, "_aggregated_results", {})
    verified_scores = {}
    all_matched = True

    print(f"\n{'='*80}")
    print(f"Verifying Aggregated/Micro Consistency for '{folder_name}'")
    print(f"{'='*80}")

    for tolerance_level, micro_es in es_scores.items():
        aggregated_es = aggregated_results.get(tolerance_level)
        is_matched, diff, relative_diff = compare_single_tolerance_level(
            tolerance_level, micro_es, aggregated_es, tolerance_threshold
        )

        print_verification_result(
            tolerance_level, micro_es, aggregated_es, diff, relative_diff, is_matched
        )

        if aggregated_es is None or not is_matched:
            all_matched = False
        if is_matched:
            verified_scores[tolerance_level] = micro_es

    if not all_matched:
        print(
            f"\nERROR: Aggregated and micro results do not match for '{folder_name}'!"
        )
        print("Calculation validation failed. Results will NOT be used for plotting.")
        print("Please verify the calculation logic using verify_aggregated_params.py")
        print(f"{'='*80}\n")
    else:
        print(f"\nSUCCESS: All aggregated and micro results match for '{folder_name}'.")
        print(f"{'='*80}\n")

    return verified_scores, all_matched


def plot_ES_results(s_scores: dict, cli_args: argparse.Namespace):
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
        ) in scores_dict.items():  # Change variable name to score_data
            # Access the 'score' key from the nested dictionary
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

    p = cli_args.negative_speedup_penalty
    config = f"p = {p}, b = {cli_args.fpdb}"
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

    ax.legend(fontsize=16, loc="best")
    output_file = os.path.join(cli_args.output_dir, "ES_result.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to {output_file}")


def main():
    """Main execution function for plotting ES(t)."""
    parser = argparse.ArgumentParser(
        description="Calculate and plot ES(t) scores from benchmark results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Add arguments (same as plot_St)
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
    args = parser.parse_args()

    # 1. Scan folders to get data
    all_results = analysis_util.scan_all_folders(args.benchmark_path)
    if not all_results:
        print("No valid data found. Exiting.")
        return

    # 2. Calculate scores for each curve and verify aggregated/micro consistency
    all_es_scores = {}
    tolerance_threshold = 1e-6  # Tolerance for floating point comparison

    for folder_name, samples in all_results.items():
        _, es_scores = analysis_util.calculate_s_scores(
            samples,
            folder_name,
            negative_speedup_penalty=args.negative_speedup_penalty,
            fpdb=args.fpdb,
        )

        # Keep original behavior: assign es_scores directly
        all_es_scores[folder_name] = es_scores

        # Verify aggregated/micro consistency
        verified_scores, all_matched = verify_aggregated_micro_consistency(
            es_scores, folder_name, tolerance_threshold
        )

        if not all_matched:
            continue  # Skip this curve if validation fails

        all_es_scores[folder_name] = verified_scores

    # 3. Plot the results
    if any(all_es_scores.values()):
        os.makedirs(args.output_dir, exist_ok=True)
        plot_ES_results(all_es_scores, args)
    else:
        print("No ES(t) scores were calculated. Skipping plot generation.")


if __name__ == "__main__":
    main()
