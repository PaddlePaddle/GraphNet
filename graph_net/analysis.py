import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import json
import re


def read_speedups_from_log(log_file):
    speedups = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                match = re.search(
                    r"duration.*eager:\s*(\d+\.?\d*).*compiled:\s*(\d+\.?\d*)", line
                )
                if match:
                    eager_time = float(match.group(1))
                    compiled_time = float(match.group(2))
                    if compiled_time > 0:
                        speedups.append(eager_time / compiled_time)
    except FileNotFoundError:
        print(f"Error: Log file not found -> {log_file}")
        return []
    return speedups


def read_speedups_from_json(benchmark_path):
    speedups = []
    if not os.path.exists(benchmark_path):
        print(f"Error: Path does not exist -> {benchmark_path}")
        return []

    try:
        for root, _, files in os.walk(benchmark_path):
            for file in files:
                if file.endswith(".json"):
                    json_file = os.path.join(root, file)
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                            if (
                                "performance" in data
                                and "speedup" in data["performance"]
                            ):
                                speedups.append(data["performance"]["speedup"])
                            else:
                                print(
                                    f"Warning: Invalid JSON format (missing 'performance.speedup') -> {json_file}"
                                )
                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON file -> {json_file}")
                        continue
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return []

    return speedups


def analysis(args):
    compilers = ["CINN", "torch.inductor", "tvm", "XLA", "TensorRT", "BladeDISC"]
    num_samples_per_compiler = 200
    data = {"Compiler": [], "log2(speedup)": []}

    # A: CINN (Simulate)
    # data["log2(speedup)"].extend(
    #     np.random.normal(loc=0.35, scale=0.2, size=num_samples_per_compiler)
    # )
    # data["Compiler"].extend(["CINN"] * num_samples_per_compiler)

    # B: torch.inductor
    # inductor_log = os.path.join(args.test_compiler_log_file)
    # inductor_speedup = read_speedups_from_log(inductor_log)
    inductor_speedup = read_speedups_from_json(args.benchmark_path)
    print(f"Find {len(inductor_speedup)} samples.")
    log2_speedups = np.log2(inductor_speedup)

    mask = log2_speedups <= 2
    filtered_log2_speedups = log2_speedups[mask]
    filtered_count = len(filtered_log2_speedups)
    print(
        f"After filtering, {filtered_count} samples remain (removed {len(log2_speedups) - filtered_count} outliers)."
    )

    data["log2(speedup)"].extend(filtered_log2_speedups)
    data["Compiler"].extend(["torch.inductor"] * len(filtered_log2_speedups))
    # data["log2(speedup)"].extend(log2_speedups)
    # data["Compiler"].extend(["torch.inductor"] * len(log2_speedups))

    # C: tvm (Simulate)
    # data["log2(speedup)"].extend(
    #     np.random.normal(loc=0.3, scale=0.15, size=num_samples_per_compiler)
    # )
    # data["Compiler"].extend(["tvm"] * num_samples_per_compiler)

    # D: XLA (Simulate)
    # data["log2(speedup)"].extend(
    #     np.concatenate(
    #         [
    #             np.random.normal(
    #                 loc=-0.5, scale=0.1, size=int(num_samples_per_compiler * 0.6)
    #             ),
    #             np.random.normal(
    #                 loc=0.2, scale=0.2, size=int(num_samples_per_compiler * 0.4)
    #             ),
    #         ]
    #     )
    # )
    # data["Compiler"].extend(["XLA"] * num_samples_per_compiler)

    # E: TensorRT (Simulate)
    # data["log2(speedup)"].extend(
    #     np.random.normal(loc=0.5, scale=0.1, size=num_samples_per_compiler)
    # )
    # data["Compiler"].extend(["TensorRT"] * num_samples_per_compiler)

    # F: BladeDISC (Simulate)
    # data["log2(speedup)"].extend(
    #     np.random.normal(loc=0.05, scale=0.3, size=num_samples_per_compiler)
    # )
    # data["Compiler"].extend(["BladeDISC"] * num_samples_per_compiler)

    df = pd.DataFrame(data)
    df["Compiler"] = pd.Categorical(df["Compiler"], categories=compilers, ordered=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    ax = sns.violinplot(
        x="Compiler",
        y="log2(speedup)",
        data=df,
        order=compilers,
        color="white",
        linewidth=0.8,
        inner=None,
    )

    sns.boxplot(
        x="Compiler",
        y="log2(speedup)",
        data=df,
        order=compilers,
        showcaps=False,
        boxprops={"facecolor": "royalblue", "edgecolor": "black"},
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": ".", "markerfacecolor": "black"},
        width=0.1,
        ax=ax,
    )

    ax.set_ylabel("log2(speedup)", fontsize=14)
    ax.set_xlabel("")
    x_labels = [f"{chr(65+i)}\n{compiler}" for i, compiler in enumerate(compilers)]
    ax.set_xticks(ticks=range(len(x_labels)), labels=x_labels, fontsize=12)
    ax.tick_params(axis="y", colors="black")
    sns.despine(trim=True, left=True)

    plt.savefig(args.output_file, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {args.output_file}")


def main(args):
    analysis(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse speedup from different compile frameworks/hardware types"
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Path include multiple benchmark results from test_compiler",
    )
    parser.add_argument(
        "--test-compiler-log-file",
        type=str,
        required=False,
        help="Log from test_compiler (Outdated)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="compiler_speedup.png",
        help="Output figure file name",
    )
    args = parser.parse_args()
    main(args=args)
