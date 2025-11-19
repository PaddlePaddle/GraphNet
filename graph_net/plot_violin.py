import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from graph_net import analysis_util


def plot_violin(df: pd.DataFrame, output_dir: str):
    """
    Generates a single plot with multiple violins for each category.
    """
    if df.empty:
        print("Warning: No valid data available for plotting.")
        return

    category_order = sorted(df["Category"].unique())
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    ax = sns.violinplot(
        x="Category",
        y="log2(speedup)",
        data=df,
        order=category_order,
        color="white",
        linewidth=0.8,
        inner=None,
    )

    sns.boxplot(
        x="Category",
        y="log2(speedup)",
        data=df,
        order=category_order,
        showcaps=False,
        boxprops={"facecolor": "cornflowerblue", "alpha": 0.7},
        medianprops={"color": "red", "linewidth": 2},
        flierprops={"marker": ".", "markerfacecolor": "black"},
        width=0.1,
        ax=ax,
    )

    ax.axhline(0, ls="--", color="gray", linewidth=1.2)
    ax.set_ylabel("log2(speedup)", fontsize=18)
    ax.set_xlabel(None)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=14)
    sns.despine(trim=True, left=True)

    legend_elements = [
        Line2D(
            [0], [0], color="cornflowerblue", lw=4, label="Interquartile Range (IQR)"
        ),
        Line2D([0], [0], color="red", lw=2, label="Median Speedup"),
        Line2D([0], [0], ls="--", color="gray", lw=1.2, label="No Speedup (Baseline)"),
    ]
    ax.legend(handles=legend_elements, fontsize=16, loc="best")
    # ax.set_title("Speedup Distribution Comparison", fontsize=20, pad=20)

    output_file = os.path.join(output_dir, "Violin_Eval_Result.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nViolin plot saved to: {output_file}")
    plt.close()


def main():
    """
    Main analysis function to read data, aggregate it, and generate a single combined plot.
    """
    parser = argparse.ArgumentParser(
        description="Analyze benchmark speedups and generate a combined violin plot."
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Path to a log file (.log or .txt) or a directory containing log files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Output directory path for saving plots. Default: analysis_results",
    )
    args = parser.parse_args()

    # 1. Parse log files and extract speedup data
    # Use scan_all_folders to handle both single log file and directory with log files
    all_samples_by_curve = analysis_util.scan_all_folders(args.benchmark_path)

    if not all_samples_by_curve:
        print("Error: No valid benchmark data was found. Exiting.")
        return

    # 2. Extract speedup values from samples and process for plotting
    plot_data = {"Category": [], "log2(speedup)": []}
    for category_name, samples in all_samples_by_curve.items():
        for sample in samples:
            speedup_raw = sample.get("performance", {}).get("speedup", {})

            # Extract numeric speedup value: use 'e2e' from dict format {"e2e": x, "gpu": y} or direct numeric value
            speedup_numeric = None
            if isinstance(speedup_raw, dict):
                speedup_numeric = speedup_raw.get("e2e")
            elif isinstance(speedup_raw, (float, int)):
                speedup_numeric = speedup_raw
            else:
                # speedup_raw is neither dict nor numeric (e.g., None, empty dict, or other types)
                # Skip this sample as it doesn't contain valid speedup data
                continue

            # Only process positive numeric speedup values (log2 requires positive values)
            if isinstance(speedup_numeric, (float, int)) and speedup_numeric > 0:
                plot_data["log2(speedup)"].append(np.log2(float(speedup_numeric)))
                plot_data["Category"].append(category_name)
            else:
                # speedup_numeric is None, non-numeric, or non-positive
                # Skip this sample as it doesn't have a valid positive speedup value for log2 calculation
                continue

    if not plot_data["log2(speedup)"]:
        print("Error: No valid speedup data was found. Exiting.")
        return

    df_all = pd.DataFrame(plot_data)

    # 3. Create output directory and generate plot
    os.makedirs(args.output_dir, exist_ok=True)
    plot_violin(df_all, args.output_dir)


if __name__ == "__main__":
    main()
