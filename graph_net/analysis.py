import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import json
import re


def read_speedups(log_file):
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
        print(f"错误：日志文件未找到 -> {log_file}")
        return []
    return speedups


def analysis(args):
    compilers = ["CINN", "torch.inductor", "tvm", "XLA", "TensorRT", "BladeDISC"]
    num_samples_per_compiler = 200
    data = {"Compiler": [], "log2(speedup)": []}

    # A: CINN (使用模拟数据)
    data["log2(speedup)"].extend(
        np.random.normal(loc=0.35, scale=0.2, size=num_samples_per_compiler)
    )
    data["Compiler"].extend(["CINN"] * num_samples_per_compiler)

    # B: torch.inductor (使用传入参数指定的文件)
    inductor_data_file = os.path.join(args.benchmark_log_file)
    log2_speedups = np.log2(read_speedups(inductor_data_file))
    data["log2(speedup)"].extend(log2_speedups)
    data["Compiler"].extend(["torch.inductor"] * len(log2_speedups))

    # C: tvm (使用模拟数据)
    data["log2(speedup)"].extend(
        np.random.normal(loc=0.3, scale=0.15, size=num_samples_per_compiler)
    )
    data["Compiler"].extend(["tvm"] * num_samples_per_compiler)

    # D: XLA (使用模拟数据)
    data["log2(speedup)"].extend(
        np.concatenate(
            [
                np.random.normal(
                    loc=-0.5, scale=0.1, size=int(num_samples_per_compiler * 0.6)
                ),
                np.random.normal(
                    loc=0.2, scale=0.2, size=int(num_samples_per_compiler * 0.4)
                ),
            ]
        )
    )
    data["Compiler"].extend(["XLA"] * num_samples_per_compiler)

    # E: TensorRT (使用模拟数据)
    data["log2(speedup)"].extend(
        np.random.normal(loc=0.5, scale=0.1, size=num_samples_per_compiler)
    )
    data["Compiler"].extend(["TensorRT"] * num_samples_per_compiler)

    # F: BladeDISC (使用模拟数据)
    data["log2(speedup)"].extend(
        np.random.normal(loc=0.05, scale=0.3, size=num_samples_per_compiler)
    )
    data["Compiler"].extend(["BladeDISC"] * num_samples_per_compiler)

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
    # assert os.path.isdir(args.benchmark_path), f"目录不存在: {args.benchmark_path}"
    analysis(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析不同编译器在硬件平台上的性能加速比")
    # parser.add_argument(
    #     "--benchmark-path",
    #     type=str,
    #     required=True,
    #     help="包含基准测试结果的目录路径"
    # )
    parser.add_argument("--benchmark-log-file", type=str, required=True, help="测试log")
    parser.add_argument(
        "--output-file",
        type=str,
        default="compiler_speedup.png",
        help="输出图表文件名(默认: compiler_speedup.png)",
    )
    args = parser.parse_args()
    main(args=args)
