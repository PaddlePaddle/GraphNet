import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import json
from collections import defaultdict


def read_benchmark_data(benchmark_path):
    """
    Reads speedup data from JSON files within each immediate subdirectory of the benchmark_path.
    Each subdirectory is treated as a separate category for plotting.
    """
    data_by_subdir = defaultdict(list)

    if not os.path.exists(benchmark_path):
        print(f"Error: Path does not exist -> {benchmark_path}")
        return {}

    try:
        subdirs = [
            d
            for d in os.listdir(benchmark_path)
            if os.path.isdir(os.path.join(benchmark_path, d))
        ]
    except FileNotFoundError:
        print(f"Error: Benchmark path not found -> {benchmark_path}")
        return {}

    if not subdirs:
        print(f"Warning: No subdirectories found in -> {benchmark_path}")
        return {}

    print(f"Found subdirectories to process: {', '.join(subdirs)}")

    for subdir_name in subdirs:
        current_dir_path = os.path.join(benchmark_path, subdir_name)
        for root, _, files in os.walk(current_dir_path):
            for file in files:
                if file.endswith(".json"):
                    json_file = os.path.join(root, file)
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                            speedup_data = data.get("performance", {}).get("speedup")

                            if isinstance(speedup_data, dict):
                                if "e2e" in speedup_data:
                                    data_by_subdir[subdir_name].append(
                                        speedup_data["e2e"]
                                    )
                                elif "gpu" in speedup_data:
                                    data_by_subdir[subdir_name].append(
                                        speedup_data["gpu"]
                                    )
                            elif isinstance(speedup_data, (float, int)):
                                data_by_subdir[subdir_name].append(speedup_data)

                    except (json.JSONDecodeError, KeyError) as e:
                        print(
                            f"Warning: Failed to read or parse file -> {json_file}, Error: {e}"
                        )
                        continue

    return data_by_subdir


def plot_violin(df, output_dir):
    """
    Generates a single plot with multiple violins for each subdirectory.
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

    ax.set_ylabel("log2(speedup)", fontsize=18)
    ax.set(xlabel="")
    ax.set_xticks(range(len(category_order)))
    ax.set_xticklabels(category_order, fontsize=18)
    sns.despine(trim=True, left=True)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Median"),
        Line2D([0], [0], marker=".", color="black", lw=0, label="Outliers"),
    ]
    ax.legend(handles=legend_elements, fontsize=16, loc="best")

    output_file = os.path.join(output_dir, "Violin_Eval_Result.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nCombined comparison plot saved to: {output_file}")
    plt.close()


def analysis(args):
    """
    Main analysis function to read data, aggregate it, and generate a single combined plot.
    """
    data_by_subdir = read_benchmark_data(args.benchmark_path)

    if not data_by_subdir:
        print("Error: No valid benchmark data was found. Exiting.")
        return

    plot_data = {"Category": [], "log2(speedup)": []}

    for subdir_name, speedups in data_by_subdir.items():
        if not speedups:
            print(f"Warning: No speedup values found for '{subdir_name}'.")
            continue

        speedups_array = np.array(speedups)
        positive_speedups = speedups_array[speedups_array > 0]
        if len(positive_speedups) == 0:
            print(f"Warning: No positive speedup values for '{subdir_name}' to plot.")
            continue

        log2_speedups = np.log2(positive_speedups)
        plot_data["log2(speedup)"].extend(log2_speedups)
        plot_data["Category"].extend([subdir_name] * len(log2_speedups))

    df_all = pd.DataFrame(plot_data)

    if df_all.empty:
        print("Error: No valid data available for plotting after processing.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    plot_violin(df_all, args.output_dir)


if __name__ == "__main__":
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
    analysis(args)
