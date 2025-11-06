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
        help="Path to the root directory containing benchmark result subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Output directory path for saving plots. Default: analysis_results",
    )
    args = parser.parse_args()

    # 1. Use the utility function to extract data
    data_by_subdir = analysis_util.extract_speedup_data_from_subdirs(
        args.benchmark_path
    )

    if not data_by_subdir:
        print("Error: No valid benchmark data was found. Exiting.")
        return

    # 2. Process data for plotting
    plot_data = {"Category": [], "log2(speedup)": []}
    for subdir_name, speedups in data_by_subdir.items():
        if not speedups:
            print(f"Warning: No speedup values found for '{subdir_name}'.")
            continue

        speedups_array = np.array(speedups)
        positive_speedups = speedups_array[speedups_array > 0]
        if len(positive_speedups) == 0:
            print(
                f"Warning: No positive speedup values for '{subdir_name}' to plot (log2 requires positive values)."
            )
            continue

        log2_speedups = np.log2(positive_speedups)
        plot_data["log2(speedup)"].extend(log2_speedups)
        plot_data["Category"].extend([subdir_name] * len(log2_speedups))

    df_all = pd.DataFrame(plot_data)

    if df_all.empty:
        print("Error: No valid data available for plotting after processing.")
        return

    # 3. Create output directory and generate plot
    os.makedirs(args.output_dir, exist_ok=True)
    plot_violin(df_all, args.output_dir)


if __name__ == "__main__":
    main()
