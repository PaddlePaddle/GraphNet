import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from graph_net import analysis_util


def plot_St(s_scores: dict, cli_args: argparse.Namespace):
    """
    Plot S(t) curve
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    all_x_coords = []

    for idx, (folder_name, scores_dict) in enumerate(s_scores.items()):
        plot_points = []
        for t_key, score in scores_dict.items():
            if t_key <= 0:
                all_x_coords.append(t_key)
                plot_points.append({"x": t_key, "y": score})

        plot_points.sort(key=lambda p: p["x"])

        x_vals = np.array([p["x"] for p in plot_points])
        y_vals = np.array([p["y"] for p in plot_points])

        color = colors[idx % len(colors)]
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
    ax.set_ylabel("S(t)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)

    if all_x_coords:
        x_min = int(np.floor(min(all_x_coords)))
        x_max = int(np.ceil(max(all_x_coords)))
        ax.set_xticks(np.arange(x_min, x_max + 1))

    ax.xaxis.grid(True, which="major", lw=0.8, ls=":", color="grey", alpha=0.5)
    ax.yaxis.grid(True, which="major", lw=0.8, ls=":", color="grey", alpha=0.5)

    ax.legend(fontsize=16, loc="best")
    output_file = os.path.join(cli_args.output_dir, "S_result.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved to {output_file}")


def main(args):
    # 1. Scan folders to get data
    all_results = analysis_util.scan_all_folders(args.benchmark_path)
    if not all_results:
        print("No valid data found. Exiting.")
        return

    # 2. Calculate scores for each curve
    all_s_scores = {}
    for folder_name, samples in all_results.items():
        s_scores, _ = analysis_util.calculate_s_scores(
            samples,
            folder_name,
            negative_speedup_penalty=args.negative_speedup_penalty,
            fpdb=args.fpdb,
        )
        all_s_scores[folder_name] = s_scores

    # 3. Plot the results
    if any(all_s_scores.values()):
        os.makedirs(args.output_dir, exist_ok=True)
        plot_St(all_s_scores, args)
    else:
        print("No S(t) scores were calculated. Skipping plot generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and plot S(t) scores from benchmark results.",
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
    args = parser.parse_args()
    main(args)
