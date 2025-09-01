import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import json
from collections import defaultdict


def parse_filename(filename):
    """
    Parses the model name and compiler name from a JSON filename.
    According to output filename format of graph_net.torch.test_compiler: <model_name>_<compiler_name>.json
    """
    parts = os.path.splitext(filename)[0].split("_")
    if len(parts) < 2:
        return None, None
    compiler = parts[-1]
    model = "_".join(parts[:-1])
    return model, compiler


def read_all_speedups(benchmark_path):
    """
    Recursively finds all .json files in a given path, extracts the speedup values,
    and organizes them by compiler and category (library).
    """
    data_by_compiler_category = defaultdict(lambda: defaultdict(list))
    all_compilers = set()

    if not os.path.exists(benchmark_path):
        print(f"Error: Path does not exist -> {benchmark_path}")
        return {}, []

    for root, _, files in os.walk(benchmark_path):
        for file in files:
            if file.endswith(".json"):
                _, compiler = parse_filename(file)
                if not compiler:
                    continue

                all_compilers.add(compiler)

                category = os.path.relpath(root, benchmark_path)
                if category == ".":
                    category = os.path.basename(benchmark_path)

                json_file = os.path.join(root, file)
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        speedup_data = data.get("performance", {}).get("speedup")

                        if isinstance(speedup_data, dict):
                            # Handle new format with 'e2e' and 'gpu' keys
                            if "e2e" in speedup_data:
                                data_by_compiler_category[compiler][category].append(
                                    speedup_data["e2e"]
                                )
                            elif "gpu" in speedup_data:
                                data_by_compiler_category[compiler][category].append(
                                    speedup_data["gpu"]
                                )
                        elif isinstance(speedup_data, float):
                            # Handle old format where speedup is just a number
                            data_by_compiler_category[compiler][category].append(
                                speedup_data
                            )

                except (json.JSONDecodeError, KeyError) as e:
                    print(
                        f"Warning: Failed to read or parse file -> {json_file}, Error: {e}"
                    )
                    continue

    return data_by_compiler_category, sorted(list(all_compilers))


def plot_summary_comparison(df, all_compilers, output_dir):
    """
    Generates a summary plot comparing the overall performance of all compilers.
    """
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    ax = sns.violinplot(
        x="Compiler",
        y="log2(speedup)",
        data=df,
        order=all_compilers,
        color="white",
        linewidth=0.8,
        inner=None,
    )

    sns.boxplot(
        x="Compiler",
        y="log2(speedup)",
        data=df,
        order=all_compilers,
        showcaps=False,
        boxprops={"facecolor": "royalblue", "edgecolor": "black"},
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": ".", "markerfacecolor": "black"},
        width=0.1,
        ax=ax,
    )

    sample_counts = df["Compiler"].value_counts().to_dict()
    x_labels = [
        f"{compiler}\n({sample_counts.get(compiler, 0)} samples)"
        for compiler in all_compilers
    ]

    ax.set_ylabel("log2(speedup)", fontsize=14)
    ax.set_xlabel("")
    ax.set_xticks(ticks=range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=11)
    ax.set_title("Overall Compiler Performance Comparison", fontsize=16)

    sns.despine(trim=True, left=True)

    output_file = os.path.join(output_dir, "summary_speedup_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSummary comparison plot saved to: {output_file}")
    plt.close()


def plot_per_compiler_detail(df_all, compiler_name, output_dir):
    """
    Generates a detailed plot for a single compiler, showing its performance across different categories.
    """
    df_compiler = df_all[df_all["Compiler"] == compiler_name]
    if df_compiler.empty:
        print(
            f"Warning: No valid data found for compiler '{compiler_name}'. Skipping detailed plot."
        )
        return

    categories = sorted(df_compiler["Category"].unique())

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.violinplot(
        x="Category",
        y="log2(speedup)",
        data=df_compiler,
        order=categories,
        color="white",
        linewidth=0.8,
        inner=None,
    )

    sns.boxplot(
        x="Category",
        y="log2(speedup)",
        data=df_compiler,
        order=categories,
        showcaps=False,
        boxprops={"facecolor": "royalblue", "edgecolor": "black"},
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": ".", "markerfacecolor": "black"},
        width=0.1,
        ax=ax,
    )

    sample_counts = df_compiler["Category"].value_counts().to_dict()
    # Use os.path.basename to get only the package name from the path
    x_labels = [
        f"{os.path.basename(cat)}\n(n={sample_counts.get(cat, 0)})"
        for cat in categories
    ]

    ax.set_ylabel("log2(speedup)", fontsize=14)
    ax.set_xlabel("")
    ax.set_xticks(ticks=range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=11)
    # Add the benchmark path to the title
    ax.set_title(f"Speedup for {compiler_name} by Categories", fontsize=16)

    sns.despine(trim=True, left=True)

    output_file = os.path.join(output_dir, f"{compiler_name}_speedup_by_category.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Detailed plot for '{compiler_name}' saved to: {output_file}")
    plt.close()


def analysis(args):
    data_by_compiler_category, all_compilers = read_all_speedups(args.benchmark_path)

    if not data_by_compiler_category:
        print("Error: No valid benchmark data found.")
        return

    print(f"\nDiscovered compilers: {all_compilers}")

    # Prepare data for DataFrame
    plot_data = {"Compiler": [], "Category": [], "log2(speedup)": []}

    for compiler, categories_data in data_by_compiler_category.items():
        for category, speedups in categories_data.items():
            if not speedups:
                continue

            speedups_array = np.array(speedups)
            # Filter out non-positive values before taking the logarithm
            log2_speedups = np.log2(speedups_array[speedups_array > 0])

            plot_data["log2(speedup)"].extend(log2_speedups)
            plot_data["Compiler"].extend([compiler] * len(log2_speedups))
            plot_data["Category"].extend([category] * len(log2_speedups))

    df_all = pd.DataFrame(plot_data)

    if df_all.empty:
        print("Error: No valid data available for plotting after processing.")
        return

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Generate the summary comparison plot
    plot_summary_comparison(df_all, all_compilers, args.output_dir)

    # 2. Generate a detailed plot for each compiler
    for compiler in all_compilers:
        plot_per_compiler_detail(df_all, compiler, args.output_dir)


def main(args):
    analysis(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze speedup from different compile frameworks/hardware types and generate plots."
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Path to the root directory containing benchmark result subdirectories and JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save the output figures.",
    )
    args = parser.parse_args()
    main(args)
