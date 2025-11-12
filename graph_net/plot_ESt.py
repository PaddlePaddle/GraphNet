import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from graph_net import analysis_util


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

    # 2. Calculate scores for each curve
    all_es_scores = {}
    for folder_name, samples in all_results.items():
        _, es_scores = analysis_util.calculate_s_scores(
            samples,
            folder_name,
            negative_speedup_penalty=args.negative_speedup_penalty,
            fpdb=args.fpdb,
        )
        all_es_scores[folder_name] = es_scores

    # 3. Plot the results
    if any(all_es_scores.values()):
        os.makedirs(args.output_dir, exist_ok=True)
        plot_ES_results(all_es_scores, args)
    else:
        print("No ES(t) scores were calculated. Skipping plot generation.")


if __name__ == "__main__":
    main()
