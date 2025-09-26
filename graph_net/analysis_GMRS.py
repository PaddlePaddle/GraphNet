import os
import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
from collections import OrderedDict
from scipy.optimize import curve_fit
from graph_net.datatype_tolerance_config import get_precision


# ---------- 1. 数据加载与处理 ----------
def load_json_file(filepath: str) -> dict:
    """安全地加载 JSON 文件并返回数据，若加载失败则返回空字典"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    Warning: Could not process file {filepath}. Error: {e}")
        return {}


def load_one_folder(folder_path: str) -> list:
    """
    遍历 *单个* 文件夹下的所有 .json 文件，加载所有原始数据。
    返回一个包含原始数据字典的列表。
    """
    if not os.path.isdir(folder_path):
        return []

    folder_name = os.path.basename(folder_path)
    samples = []
    print(f"  - Loading JSON files from folder: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            data = load_json_file(filepath)
            if data:
                samples.append(data)
    return samples


def scan_all_folders(benchmark_path: str) -> dict:
    """
    统一入口：
      - 如果 benchmark_path 下直接有 .json → 当成单条曲线（曲线名用目录名）。
      - 否则退回到“子目录=曲线”的老逻辑。
    返回 dict[folder_name] -> list_of_samples
    """
    if not os.path.isdir(benchmark_path):
        print(f"Error: Provided path '{benchmark_path}' is not a valid directory.")
        return {}

    print(f"Scanning '{benchmark_path}' ...")

    # 先尝试扁平结构，直接读取 JSON
    flat_samples = load_one_folder(benchmark_path)
    if flat_samples:  # ≥1 个 json 被成功加载
        folder_name = os.path.basename(benchmark_path) or "benchmark"
        print(
            f"  - Detected flat structure → 1 curve '{folder_name}' "
            f"with {len(flat_samples)} samples."
        )
        return {folder_name: flat_samples}

    # 退回到子目录为曲线的逻辑
    all_results = {}
    print("  - No JSON files found at top level → scanning sub-folders.")
    for entry in os.listdir(benchmark_path):
        folder_full_path = os.path.join(benchmark_path, entry)
        if os.path.isdir(folder_full_path):
            samples = load_one_folder(folder_full_path)
            if samples:
                all_results[entry] = samples
                print(f"  - Folder '{entry}' loaded {len(samples)} samples.")
    print(f"Total folders loaded: {len(all_results)}")
    return all_results


# ---------- 2. 核心计算逻辑 ----------
def get_correctness(dtype: str, t: str, sample: dict) -> bool:
    """
    根据标准尺度的 t key 和 dtype，从配置中查找实际的 atol/rtol 值，从而检查样本的正确性。
    """
    precision_pair = get_precision(t, dtype)
    atol, rtol = precision_pair[1], precision_pair[0]

    # 使用 .2E 格式化，确保小数点后有两位，并使用大写E，匹配JSON日志格式。
    # 例如：1e-10 会被格式化为 '1.00E-10'
    metric_key_to_check = f"[all_close_atol_{atol:.2E}_rtol_{rtol:.2E}]"

    # 获取 correctness 数据
    correctness_data = sample.get("correctness", {})

    # 查找并返回结果
    result = correctness_data.get(metric_key_to_check)

    return result == [1]


def fake_perf_degrad(t, fail_type, fpdb=0.1):
    """
    根据容忍度 t 和失败类型，计算 fake performance degradation。
    """
    # 适合旧版代码
    if fail_type == "accuracy":
        return fpdb + (1 - fpdb) * (1 if t >= 1 else 0)
    else:
        return fpdb + (1 - fpdb) * (1 if t >= 3 else 0)

    # if fail_type == "compiled":
    #     # 编译失败：只有 t >= 3 时才豁免（返回 1）
    #     return fpdb + (1 - fpdb) * (1 if t >= 3 else 0)
    # elif fail_type == "eager":
    #     # 执行崩溃（但编译成功）：t >= 2 时豁免
    #     return fpdb + (1 - fpdb) * (1 if t >= 2 else 0)
    # elif fail_type == "accuracy":
    #     # 精度失败（运行成功但结果错误）：t >= 1 时豁免
    #     return fpdb + (1 - fpdb) * (1 if t >= 1 else 0)


def calculate_s_scores(
    samples: list,
    folder_name: str,
    negative_speedup_penalty: float = 0,
    exec_failure_penalty: str = "0.1",
) -> dict:
    """
    使用一个标准 tolerance 来评估所有样本，并计算每个刻度上的 S(t) 分数。
    """
    s_scores = OrderedDict()
    begin = -10
    end = 5
    t_keys = list(range(begin, end + 1))

    print(
        f"\nCalculating S(t) scores for '{folder_name}' using penalty function: '{exec_failure_penalty}'..."
    )

    for t_key in t_keys:
        regularized_speedups = []
        for sample in samples:
            # 直接从原始数据字典中获取性能和失败信息
            performance_data = sample.get("performance", {})
            fail_type = performance_data.get("failure")
            speedup = performance_data.get("speedup", {}).get("e2e")
            is_correct = False

            # 如果没有严重失败，则检查精度
            if fail_type is None:
                datatype_data = performance_data.get("datatype", {})
                eager_dtypes = datatype_data.get("eager", [])
                compiled_dtypes = datatype_data.get("compiled", [])

                # 只有当 eager 和 compiled 数据类型列表一致且不为空时，才进行正确性检查
                if eager_dtypes and eager_dtypes == compiled_dtypes:
                    for dtype in eager_dtypes:
                        if get_correctness(dtype, t_key, sample):
                            is_correct = True
                            break

                if not is_correct:
                    fail_type = "accuracy"

            # 应用惩罚逻辑
            if fail_type is not None or speedup is None:
                exec_penalty = fake_perf_degrad(t_key, fail_type, fpdb=0.1)
                regularized_speedup = exec_penalty
            else:
                if speedup < 1:
                    regularized_speedup = speedup ** (negative_speedup_penalty + 1)
                else:
                    regularized_speedup = speedup

            regularized_speedups.append(regularized_speedup)

        if regularized_speedups:
            s_scores[t_key] = gmean(regularized_speedups)
            if exec_failure_penalty == "fake_perf_degrad":
                exec_penalty = fake_perf_degrad(t_key, fail_type, fpdb=0.1)
            else:
                exec_penalty = float(exec_failure_penalty)
            print(
                f"  - S(t) for tolerance={t_key}, penalty={exec_penalty:.4f}: {s_scores[t_key]:.4f}"
            )

    return s_scores


# ---------- 3. 绘图功能 ----------
def plot_results(GMRS_scores: dict, cli_args: argparse.Namespace):
    """
    绘制 S(t) 曲线
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    all_x_coords = []

    # 绘制每个文件夹的曲线
    for idx, (folder_name, s_scores) in enumerate(GMRS_scores.items()):
        plot_points = []

        for t_key, score in s_scores.items():
            try:
                all_x_coords.append(t_key)
                plot_points.append({"x": t_key, "y": score})
            except (ValueError, TypeError):
                print(f"Warning: Could not parse t_key '{t_key}'. Skipping for plot.")
                continue

        if not plot_points:
            print(f"Warning: No valid points to plot for folder '{folder_name}'.")
            continue

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

        # for p in plot_points:
        #     ax.text(p['x'], p['y'], f"{p['y']:.3f}", ha='center', va='bottom', fontsize=12, color=color)

    penalty_p = cli_args.negative_speedup_penalty
    penalty_exec = cli_args.exec_failure_penalty
    config = (
        f"Penalty Settings: exec_failure_penalty='{penalty_exec}', "
        f"negative_speedup_penalty(p)={penalty_p}"
    )
    fig.text(0.5, 0.9, config, ha="center", fontsize=16, style="italic")

    ax.set_xlabel("tolerance", fontsize=18)
    ax.set_ylabel("GMRS(t,p) (Geometric Mean of Rectified Speedup)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)

    if all_x_coords:
        x_min = int(np.floor(min(all_x_coords)))
        x_max = int(np.ceil(max(all_x_coords)))
        ax.set_xticks(np.arange(x_min, x_max + 2, 1))  # 扩展X轴范围以容纳末端文本

    ax.xaxis.grid(True, which="major", lw=0.7, ls=":", color="grey", alpha=0.5)
    ax.yaxis.grid(True, which="major", lw=0.7, ls=":", color="grey", alpha=0.5)

    ax.legend(fontsize=16, loc="best")
    plt.savefig("Eval_result.png", dpi=150)
    print("\nComparison plot saved to Eval_result.png")


# ---------- 4. 主程序入口 ----------
def main():
    """
    主执行函数。
    """
    parser = argparse.ArgumentParser(
        description="Load benchmark JSON records from multiple sub-folders, "
        "calculate aggregated S(t) scores, and plot multi-curve results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Path to the directory containing sub-folders of benchmark JSON files.",
    )
    parser.add_argument(
        "--negative-speedup-penalty",
        type=float,
        default=0.0,
        help="Penalty power (p) for negative speedup (speedup < 1). Formula: speedup**(p+1). Default: 0.0.",
    )
    parser.add_argument(
        "--exec-failure-penalty",
        type=str,
        default="0.1",
        help="Penalty for severe errors (e.g., correctness failure, crashes). \n"
        "Can be 'fake_perf_degrad' or a constant number (e.g., '0.1').",
    )
    args = parser.parse_args()

    # 1. 扫描所有子目录
    all_results = scan_all_folders(args.benchmark_path)
    if not all_results:
        print("No valid data found. Exiting.")
        return

    # 2. 分别计算每条曲线的 S 分数
    GMRS_scores = {
        folder_name: calculate_s_scores(
            samples,
            folder_name,
            negative_speedup_penalty=args.negative_speedup_penalty,
            exec_failure_penalty=args.exec_failure_penalty,
        )
        for folder_name, samples in all_results.items()
    }

    # 3. 多曲线绘图 (传入汇总数据和命令行参数)
    if any(GMRS_scores.values()):
        plot_results(GMRS_scores, args)
    else:
        print("No S(t) scores were calculated. Skipping plot generation.")


if __name__ == "__main__":
    main()
