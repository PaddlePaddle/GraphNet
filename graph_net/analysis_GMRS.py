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
def get_correctness(dtype: str, t: int, correctness_data: dict, index: int) -> bool:
    """
    根据容忍度、数据类型和输出索引，从配置中查找实际的 atol/rtol 值，获取单个输出的正确性结果。
    """
    precision_pair = get_precision(t, dtype)
    atol, rtol = precision_pair[1], precision_pair[0]

    if atol == 0 and rtol == 0:
        metric_key_to_check = "[equal]"
    else:
        # 使用 .2E 格式化，确保小数点后有两位，并使用大写E，匹配JSON日志格式
        metric_key_to_check = f"[all_close_atol_{atol:.2E}_rtol_{rtol:.2E}]"

    result = correctness_data.get(metric_key_to_check)
    if isinstance(result, list) and len(result) > index:
        return bool(result[index])
    return False


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
    fpdb: float = 0.1,
) -> tuple:
    """
    使用一个标准 tolerance 来评估所有样本，并计算每个刻度上的 S(t) 和 ES(t) 分数。
    """
    s_scores = OrderedDict()
    s_scores_fake_degrad = OrderedDict()

    begin = -10
    end = 4
    t_keys = list(range(begin, end + 1))
    total_samples = len(samples)

    print(f"\nCalculating S(t) scores for '{folder_name}'...")

    def print_stat_info(
        t_key,
        correct_count,
        acc_failure_count,
        other_failure_count,
        correct_negative_speedup_count,
        correct_speedups,
        slowdown_speedups,
    ):
        print(f"  - Details for tolerance={t_key}:")
        if total_samples > 0:
            alpha = gmean(correct_speedups) if correct_speedups else 0
            beta = gmean(slowdown_speedups) if slowdown_speedups else 0
            lambda_ = correct_count / total_samples
            if correct_count > 0:
                eta = correct_negative_speedup_count / correct_count
            else:
                eta = 0
            if t_key < 1:
                gamma = fpdb
            elif t_key < 3:
                gamma = (
                    acc_failure_count * 1
                    + (total_samples - correct_count - acc_failure_count) * fpdb
                ) / (total_samples - correct_count)
            else:
                gamma = 1

            print(
                f"    - alpha: {alpha:.3f} (Geometric mean speedup of correct samples)"
            )
            print(f"    - beta: {beta:.3f} (Geometric mean speedup of slowdown cases)")
            print(f"    - gamma: {gamma} (Average error penalty)")
            print(f"    - lambda: {lambda_:.3f} (Fraction of correct samples)")
            print(
                f"    - eta: {eta:.3f} (Fraction of slowdown cases within correct samples)"
            )
        else:
            print("    - No samples to analyze.")

    # ES曲线的阶梯状状态，初始化为'CORRECT'
    es_status = ["CORRECT"] * total_samples

    # 核心计算逻辑，处理两种惩罚模式
    for t_key in t_keys:
        regularized_speedups = []
        regularized_speedups_fake_degrad = []
        correct_count = 0
        acc_failure_count = 0
        other_failure_count = 0
        correct_negative_speedup_count = 0

        # 新增：用于计算 alpha 和 beta 的列表
        correct_speedups = []
        slowdown_speedups = []

        for idx, sample in enumerate(samples):
            performance_data = sample.get("performance", {})
            fail_type = performance_data.get("failure")
            speedup = performance_data.get("speedup", {}).get("e2e")

            # 判定当前样本的真实状态（用于统计和S曲线）
            is_correct = False
            if fail_type is None:
                datatype_data = performance_data.get("datatype", {})
                eager_dtypes = datatype_data.get("eager", [])
                compiled_dtypes = datatype_data.get("compiled", [])
                if (
                    eager_dtypes
                    and eager_dtypes == compiled_dtypes
                    and len(eager_dtypes) > 0
                ):
                    correctness_data = sample.get("correctness", {})
                    output_count = len(correctness_data.get("[equal]", []))
                    if len(eager_dtypes) == output_count:
                        is_correct = all(
                            get_correctness(eager_dtypes[i], t_key, correctness_data, i)
                            for i in range(output_count)
                        )
                    if not is_correct:
                        fail_type = "accuracy"

            # 统计信息
            if is_correct:
                correct_count += 1
                if speedup is not None:
                    correct_speedups.append(speedup)
                if speedup is not None and speedup < 1:
                    correct_negative_speedup_count += 1
                    slowdown_speedups.append(speedup)
            elif fail_type == "accuracy":
                acc_failure_count += 1
            else:
                other_failure_count += 1

            # S(t) 计算（基于原始状态）
            if fail_type is not None or speedup is None:
                regularized_speedup = fpdb
            else:
                regularized_speedup = (
                    speedup ** (negative_speedup_penalty + 1)
                    if speedup < 1
                    else speedup
                )
            regularized_speedups.append(regularized_speedup)

            # ES(t) 核心逻辑：基于状态变化
            reg_speedup_fake_degrad = 0
            if t_key < 1:
                # 在 t < 1 时，ES行为与S相同
                if fail_type is not None or speedup is None:
                    reg_speedup_fake_degrad = fpdb
                else:
                    reg_speedup_fake_degrad = (
                        speedup ** (negative_speedup_penalty + 1)
                        if speedup < 1
                        else speedup
                    )
            else:
                # 在 t >= 1 时，ES开始应用阶梯状逻辑
                if es_status[idx] == "CORRECT" and fail_type is not None:
                    es_status[idx] = fail_type

                if es_status[idx] == "accuracy":
                    reg_speedup_fake_degrad = fake_perf_degrad(t_key, "accuracy", fpdb)
                elif es_status[idx] is not None and es_status[idx] != "CORRECT":
                    reg_speedup_fake_degrad = fake_perf_degrad(
                        t_key, es_status[idx], fpdb
                    )
                else:  # Still in a "CORRECT" state
                    if speedup is None:
                        reg_speedup_fake_degrad = fpdb
                    else:
                        reg_speedup_fake_degrad = (
                            speedup ** (negative_speedup_penalty + 1)
                            if speedup < 1
                            else speedup
                        )

            regularized_speedups_fake_degrad.append(reg_speedup_fake_degrad)

        if regularized_speedups:
            s_scores[t_key] = gmean(regularized_speedups)
            s_scores_fake_degrad[t_key] = gmean(regularized_speedups_fake_degrad)
            print_stat_info(
                t_key,
                correct_count,
                acc_failure_count,
                other_failure_count,
                correct_negative_speedup_count,
                correct_speedups,
                slowdown_speedups,
            )
            print(
                f"  - S(t)={s_scores[t_key]:.3f}, ES(t)={s_scores_fake_degrad[t_key]:.3f} for tolerance={t_key}."
            )

    return s_scores, s_scores_fake_degrad


# ---------- 3. 绘图功能 ----------
def plot_S_results(s_scores: dict, cli_args: argparse.Namespace):
    """
    绘制 S(t) 曲线
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    all_x_coords = []

    for idx, (folder_name, scores_dict) in enumerate(s_scores.items()):
        plot_points = []
        for t_key, score in scores_dict.items():
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

    ax.xaxis.grid(True, which="major", lw=0.7, ls=":", color="grey", alpha=0.5)
    ax.yaxis.grid(True, which="major", lw=0.7, ls=":", color="grey", alpha=0.5)

    ax.legend(fontsize=16, loc="best")
    plt.savefig("S_result.png", dpi=150)
    print("\nComparison plot saved to S_result.png")


def plot_ES_results(s_scores: dict, cli_args: argparse.Namespace):
    """
    绘制 ES(t) 曲线
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))

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

        # 按x值排序
        plot_points.sort(key=lambda p: p["x"])

        x_vals = np.array([p["x"] for p in plot_points])
        y_vals = np.array([p["y"] for p in plot_points])

        color = colors[idx % len(colors)]

        # 找到 t=0 的索引
        zero_index = np.where(x_vals == 0)[0][0] if 0 in x_vals else None

        # 如果存在t=0的点，则分段绘制
        if zero_index is not None:
            # 绘制 t <= 0 的连续直线部分
            ax.plot(
                x_vals[: zero_index + 1],
                y_vals[: zero_index + 1],
                "o-",
                color=color,
                label=folder_name,
                linewidth=2,
                markersize=6,
            )
            # 绘制 t > 0 的阶梯状部分
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
            # 如果没有t=0，则整个曲线都使用常规直线
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
    config = f"p = {p}, gamma = fake_perf_degrad (b={cli_args.fpdb})"
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
    plt.savefig("ES_result.png", dpi=150)
    print("\nComparison plot saved to ES_result.png")


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
        "--fpdb",
        type=float,
        default=0.1,
        help="Base penalty for severe errors (e.g., correctness failure, crashes).",
    )
    args = parser.parse_args()

    # 1. 扫描所有子目录
    all_results = scan_all_folders(args.benchmark_path)
    if not all_results:
        print("No valid data found. Exiting.")
        return

    # 2. 分别计算每条曲线的 S 分数
    all_s_scores = {}
    all_s_scores_fake_degrad = {}

    for folder_name, samples in all_results.items():
        s_scores, s_scores_fake_degrad = calculate_s_scores(
            samples,
            folder_name,
            negative_speedup_penalty=args.negative_speedup_penalty,
            fpdb=args.fpdb,
        )
        all_s_scores[folder_name] = s_scores
        all_s_scores_fake_degrad[folder_name] = s_scores_fake_degrad

    # 3. 多曲线绘图 (传入汇总数据和命令行参数)
    if any(all_s_scores.values()):
        plot_S_results(all_s_scores, args)
        plot_ES_results(all_s_scores_fake_degrad, args)
    else:
        print("No S(t) scores were calculated. Skipping plot generation.")


if __name__ == "__main__":
    main()
