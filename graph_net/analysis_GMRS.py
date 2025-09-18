import os
import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
from collections import OrderedDict
from scipy.optimize import curve_fit


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
    遍历 *单个* 文件夹下的所有 .json 文件，加载所有样本数据（包括失败的）。
    返回的每条样本里额外记录 'folder_name'，用于后续多曲线区分。
    """
    if not os.path.isdir(folder_path):
        return []

    folder_name = os.path.basename(folder_path)  # 用作曲线名称
    samples = []
    print(f"  - Loading JSON files from folder: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            data = load_json_file(filepath)
            if data:  # 如果读取数据成功
                sample_data = {
                    "folder_name": folder_name,
                    "model": data.get("configuration", {}).get("model", "unknown"),
                    "compiler": data.get("configuration", {}).get(
                        "compiler", "unknown"
                    ),
                    "speedup": data.get("performance", {})
                    .get("speedup", {})
                    .get("e2e"),
                    "failed": data.get("performance", {}).get("failure") == "True",
                    "correctness_metrics": data.get("correctness", {}),
                    "performance": data.get("performance", {}),
                }
                samples.append(sample_data)

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


def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))


def get_tolerance(
    datatype: str, atol_key: str, rtol_key: str, tolerance_config: dict
) -> list:
    """
    根据标准尺度的 atol/rtol key 和数据类型，从配置中查找实际要检查的 atol/rtol 值。
    """
    precision_table = tolerance_config.get("precision_table", {}).get(datatype, {})
    atol_dict = precision_table.get("atol", {})
    rtol_dict = precision_table.get("rtol", {})

    atol_value_actual = atol_dict.get(atol_key)
    rtol_value_actual = rtol_dict.get(rtol_key)

    if atol_value_actual is not None and rtol_value_actual is not None:
        return [atol_value_actual, rtol_value_actual]
    else:
        print(
            f"Warning: No tolerance mapping found for datatype={datatype}, atol_key={atol_key}, rtol_key={rtol_key}."
        )
        return None


def get_correctness(sample: dict, atol_actual: str, rtol_actual: str) -> bool:
    """
    根据给定的实际 atol 和 rtol 值，检查样本的正确性字典。
    """
    metric_key_to_check = f"[all_close_atol_{atol_actual}_rtol_{rtol_actual}]"
    correctness_data = sample.get("correctness_metrics", {})
    return correctness_data.get(metric_key_to_check) == [1]


def calculate_s_scores(
    samples: list,
    negative_speedup_penalty: float = 0,
    tolerance_config: dict = None,
    exec_failure_penalty: str = "0.1",
) -> dict:
    """
    使用一个标准的精度“尺子”来评估所有样本，并计算每个刻度上的 S(t) 分数。
    错误惩罚 'exec_penalty' 是可配置的。
    """
    if not tolerance_config:
        print("Error: tolerance_config is required for this calculation logic.")
        return {}

    s_scores = OrderedDict()

    try:
        ruler_keys = tolerance_config["precision_table"]["float32"]["rtol"].keys()
        sorted_keys = sorted(ruler_keys, key=lambda k: float(k))
    except KeyError:
        print(
            "Error: Could not establish standard ruler from tolerance_config. Check 'float32' rtol keys."
        )
        return {}

    print(
        f"\nCalculating S(t) scores using penalty function: '{exec_failure_penalty}'..."
    )

    # 遍历“尺子”上的每一个刻度
    for rtol_key in sorted_keys:
        atol_key = rtol_key

        ruler_key = int(re.search(r"[eE][-+]?\d+", rtol_key).group(0)[1:])
        if exec_failure_penalty == "logistic":
            exec_penalty = sigmoid(np.log10(ruler_key))
        elif exec_failure_penalty == "max-tanh":
            exec_penalty = max(np.tanh(ruler_key), 0.1)
        elif exec_failure_penalty == "step":
            if ruler_key <= 0:
                exec_penalty = 0.1
            else:
                exec_penalty = 1
        else:
            exec_penalty = float(exec_failure_penalty)

        regularized_speedups = []

        # 用当前刻度去衡量每一个样本
        for sample in samples:
            is_failed = sample.get("failed", False)
            if not is_failed:
                speedup = sample.get("speedup")
                performance = sample.get("performance", {})
                eager_dtypes = performance.get("datatype", {}).get("eager", [])
                compiled_dtypes = performance.get("datatype", {}).get("compiled", [])
                if eager_dtypes == compiled_dtypes and eager_dtypes:
                    for dtype in eager_dtypes:
                        atol_actual, rtol_actual = get_tolerance(
                            dtype, atol_key, rtol_key, tolerance_config
                        )
                        if get_correctness(sample, atol_actual, rtol_actual):
                            is_failed = False
                            break
                        else:
                            is_failed = True
                else:
                    is_failed = True
            else:
                is_failed = True

            # 应用惩罚逻辑
            if is_failed:
                regularized_speedup = exec_penalty
            else:
                if speedup < 1:
                    regularized_speedup = speedup ** (negative_speedup_penalty + 1)
                else:
                    regularized_speedup = speedup
            regularized_speedups.append(regularized_speedup)

        if regularized_speedups:
            score = gmean(regularized_speedups)
            s_scores[rtol_key] = score
            t_val = -np.log10(float(rtol_key))
            print(
                f"  - S(t) for rtol={rtol_key} (t={t_val:.1f}), penalty={exec_penalty:.4f}: {score:.4f}"
            )

    return s_scores


def calculate_category_stats(all_results):
    """
    计算每个类别的汇总统计数据，用于在图上直接显示
    """
    summary_stats = {}
    print("\nCalculating summary statistics for each category...")
    for folder_name, samples in all_results.items():
        total_samples = len(samples)
        if total_samples == 0:
            continue

        correct_samples = 0
        slowdown_among_correct = 0

        for sample in samples:
            is_failed = sample.get("failed", False)
            performance = sample.get("performance", {})
            eager_dtypes = performance.get("datatype", {}).get("eager", [])
            compiled_dtypes = performance.get("datatype", {}).get("compiled", [])

            if not is_failed and (eager_dtypes == compiled_dtypes):
                correct_samples += 1
                speedup = sample.get("speedup")
                if speedup < 1:
                    slowdown_among_correct += 1

        correct_ratio = correct_samples / total_samples
        slowdown_ratio = slowdown_among_correct / correct_samples

        summary_stats[folder_name] = {
            "correct_ratio": correct_ratio,
            "slowdown_ratio": slowdown_ratio,
        }
        print(
            f"  - {folder_name}: Correctness Ratio={correct_ratio:.1%}, Slowdown Ratio (of correct)={slowdown_ratio:.1%}"
        )

    return summary_stats


# ---------- 3. 绘图功能 ----------
def plot_results(GMRS_scores: dict, summary_stats: dict, cli_args: argparse.Namespace):
    """
    绘制 S(t) 曲线，并附带显示全局参数和各曲线的统计数据。
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))  # 增加了图表尺寸以容纳更多信息

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    all_x_coords = []

    # 绘制每个文件夹的曲线
    for idx, (folder_name, s_scores) in enumerate(GMRS_scores.items()):
        plot_points = []

        for rtol_key, score in s_scores.items():
            try:
                x_val = np.log10(float(rtol_key))
                all_x_coords.append(x_val)
                plot_points.append({"x": x_val, "y": score, "label": rtol_key})
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not parse rtol_key '{rtol_key}'. Skipping for plot."
                )
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
        #     ax.text(p['x'], p['y'], f"{p['y']:.3f}", ha='center', va='bottom', fontsize=11, color=color)

        # # 在曲线末端标注统计数据
        # last_point = plot_points[-1]
        # stats = summary_stats.get(folder_name)
        # if stats:
        #     stats_label = (f"  Correct: {stats['correct_ratio']:.1%}\n"
        #                    f"  Slowdown (of correct): {stats['slowdown_ratio']:.1%}")
        #     # 使用一个带半透明背景的文本框以提高可读性
        #     ax.text(last_point['x'] + 0.2, last_point['y'], stats_label,
        #             color=color, fontsize=10, ha='left', va='center',
        #             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    penalty_p = cli_args.negative_speedup_penalty
    penalty_exec = cli_args.exec_failure_penalty
    config = (
        f"Penalty Settings: exec_failure_penalty='{penalty_exec}', "
        f"negative_speedup_penalty(p)={penalty_p}"
    )
    fig.text(0.5, 0.9, config, ha="center", fontsize=12, style="italic")

    ax.set_xlabel("tolerance (log10 rtol)", fontsize=18)
    ax.set_ylabel("GMRS(t,p) (Geometric Mean of Rectified Speedup)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)

    if all_x_coords:
        x_min = int(np.floor(min(all_x_coords)))
        x_max = int(np.ceil(max(all_x_coords)))
        ax.set_xticks(np.arange(x_min, x_max + 2, 1))  # 扩展X轴范围以容纳末端文本

    ax.xaxis.grid(True, which="major", lw=0.7, ls=":", color="grey", alpha=0.5)
    ax.yaxis.grid(True, which="major", lw=0.7, ls=":", color="grey", alpha=0.5)

    ax.legend(title="Category", fontsize=12, loc="best")
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
        "Can be 'logistic', 'max-tanh' or a constant number (e.g., '0.1').",
    )
    args = parser.parse_args()

    # 1. 扫描所有子目录
    all_results = scan_all_folders(args.benchmark_path)
    if not all_results:
        print("No valid data found. Exiting.")
        return

    # 2. 分别计算每条曲线的 S 分数
    tolerance_config = load_json_file(
        "/work/GraphNet/graph_net/config/datatype_accuracy.json"
    )
    GMRS_scores = {
        folder_name: calculate_s_scores(
            samples,
            tolerance_config=tolerance_config,
            negative_speedup_penalty=args.negative_speedup_penalty,
            exec_failure_penalty=args.exec_failure_penalty,
        )
        for folder_name, samples in all_results.items()
    }
    # 计算每个类别的汇总统计数据
    summary_stats = calculate_category_stats(all_results)

    # 3. 多曲线绘图 (传入汇总数据和命令行参数)
    if any(GMRS_scores.values()):
        plot_results(GMRS_scores, summary_stats, args)
    else:
        print("No S(t) scores were calculated. Skipping plot generation.")


if __name__ == "__main__":
    main()
