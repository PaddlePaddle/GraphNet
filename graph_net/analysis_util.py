import os
import json
import numpy as np
from scipy.stats import gmean
from collections import OrderedDict, defaultdict
from graph_net.config.datatype_tolerance_config import get_precision


def extract_speedup_data_from_subdirs(benchmark_path: str) -> dict:
    """
    Reads speedup data from JSON files within each immediate subdirectory of the benchmark_path.
    Each subdirectory is treated as a separate category.
    Returns a dictionary mapping {subdir_name: [speedup_values]}.
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
        # Using scan_all_folders and load_one_folder could be an alternative,
        # but os.walk is also robust for nested directories if needed in the future.
        for root, _, files in os.walk(current_dir_path):
            for file in files:
                if file.endswith(".json"):
                    json_file = os.path.join(root, file)
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                            performance = data.get("performance", {})
                            if not performance:
                                continue

                            speedup_data = performance.get("speedup")
                            if isinstance(speedup_data, dict):
                                # Prioritize 'e2e' speedup, fallback to 'gpu'
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


def load_json_file(filepath: str) -> dict:
    """
    Safely load a JSON file and return data, return an empty dictionary if loading fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    Warning: Could not process file {filepath}. Error: {e}")
        return {}


def load_one_folder(folder_path: str) -> list:
    """
    Traverse all .json files in a *single* folder and load all raw data.
    Returns a list of raw data dictionaries.
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
    Unified entry point:
      - If there are .json files directly under benchmark_path → treat them as a single curve (curve name is the directory name).
      - Otherwise, fallback to the old logic where subdirectories represent curves.
    Returns dict[folder_name] -> list_of_samples
    """
    if not os.path.isdir(benchmark_path):
        print(f"Error: Provided path '{benchmark_path}' is not a valid directory.")
        return {}

    print(f"Scanning '{benchmark_path}' ...")

    # Try flat structure, directly read JSON
    flat_samples = load_one_folder(benchmark_path)
    if flat_samples:  # ≥1 JSON loaded successfully
        folder_name = os.path.basename(benchmark_path) or "benchmark"
        print(
            f"  - Detected flat structure → 1 curve '{folder_name}' "
            f"with {len(flat_samples)} samples."
        )
        return {folder_name: flat_samples}

    # Fall back to subdirectories as curves logic
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


def get_correctness(dtype: str, t: int, correctness_data: dict, index: int) -> bool:
    """
    Based on tolerance, data type, and output index, find the actual atol/rtol values from the config and get the correctness result for a single output.
    """
    precision_pair = get_precision(t, dtype)
    atol, rtol = precision_pair[1], precision_pair[0]

    if atol == 0 and rtol == 0:
        metric_key_to_check = "[equal]"
    else:
        # Use .2E format to ensure two decimal places and use uppercase E to match JSON log format
        metric_key_to_check = f"[all_close_atol_{atol:.2E}_rtol_{rtol:.2E}]"

    result = correctness_data.get(metric_key_to_check)
    if isinstance(result, list) and len(result) > index:
        return bool(result[index])
    return False


def fake_perf_degrad(t, error_code, fpdb=0.1):
    """
    Calculate fake performance degradation based on tolerance t and error code.
    """
    if error_code == "accuracy":
        return fpdb if t < 1 else 1
    else:
        return fpdb if t < 3 else 1

    # if error_code == "compiled":
    #     # Compilation failure: only exempt if t >= 3 (return 1)
    #     return fpdb + (1 - fpdb) * (1 if t >= 3 else 0)
    # elif error_code == "eager":
    #     # Execution crash (but compilation succeeded): exempt if t >= 2
    #     return fpdb + (1 - fpdb) * (1 if t >= 2 else 0)
    # elif error_code == "accuracy":
    #     # Accuracy failure (execution succeeded but result wrong): exempt if t >= 1
    #     return fpdb + (1 - fpdb) * (1 if t >= 1 else 0)


def calculate_s_scores(
    samples: list,
    folder_name: str,
    negative_speedup_penalty: float = 0,
    fpdb: float = 0.1,
) -> tuple:
    """
    Use a standard tolerance to evaluate all samples and calculate S(t) and ES(t) scores for each tolerance level.
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
        pi,
        correct_negative_speedup_count,
        correct_speedups,
        slowdown_speedups,
    ):
        print(f"  - Details for tolerance={t_key}:")
        if total_samples > 0:
            alpha = gmean(correct_speedups) if correct_speedups else 1
            beta = gmean(slowdown_speedups) if slowdown_speedups else 1
            lambda_ = correct_count / total_samples if total_samples > 0 else 0
            eta = (
                correct_negative_speedup_count / correct_count
                if correct_count > 0
                else 0
            )
            indicator = [1 if t_key < 1 else 0, 1 if t_key < 3 else 0]
            gamma = (
                fpdb ** sum(pi[i] * indicator[i] for i in range(len(pi)))
                if t_key >= 1
                else fpdb
            )

            expected_s = (
                alpha**lambda_
                * beta ** (lambda_ * eta * negative_speedup_penalty)
                * fpdb ** (1 - lambda_)
            )
            expected_es = (
                alpha**lambda_
                * beta ** (lambda_ * eta * negative_speedup_penalty)
                * gamma ** (1 - lambda_)
            )

            print(
                f"    - alpha: {alpha:.3f} (Geometric mean speedup of correct samples)"
            )
            print(f"    - beta: {beta:.3f} (Geometric mean speedup of slowdown cases)")
            print(f"    - gamma: {gamma:.3f} (Average error penalty)")
            print(f"    - lambda: {lambda_:.3f} (Fraction of correct samples)")
            print(
                f"    - eta: {eta:.3f} (Fraction of slowdown cases within correct samples)"
            )
        else:
            print("    - No samples to analyze.")

        return expected_s, expected_es

    # pi is a list of constants for t > 0 for each group
    pi = [0, 0]

    is_correct_at_t1 = [False] * total_samples
    speedup_at_t1 = [None] * total_samples
    fail_type_at_t1 = ["CORRECT"] * total_samples

    final_correct_count = 0
    final_correct_negative_speedup_count = 0
    final_correct_speedups = []
    final_slowdown_speedups = []

    for t_key in t_keys:
        rectified_speedups = []
        rectified_speedups_fake_degrad = []
        correct_count = 0
        acc_failure_count = 0
        correct_negative_speedup_count = 0
        correct_speedups = []
        slowdown_speedups = []

        for idx, sample in enumerate(samples):
            performance_data = sample.get("performance", {})
            fail_type = performance_data.get("failure")
            speedup = performance_data.get("speedup", {}).get("e2e")

            # Determine the true state of the current sample (for statistics and S curve)
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

            # Collect statistics
            if is_correct:
                correct_count += 1
                if speedup is not None:
                    correct_speedups.append(speedup)
                if speedup is not None and speedup < 1:
                    correct_negative_speedup_count += 1
                    slowdown_speedups.append(speedup)

            if fail_type == "accuracy":
                acc_failure_count += 1

            if t_key == 1:
                is_correct_at_t1[idx] = is_correct
                speedup_at_t1[idx] = speedup
                fail_type_at_t1[idx] = fail_type if fail_type is not None else "CORRECT"

            # S(t) calculation
            if fail_type is not None or speedup is None:
                regularized_speedup = fpdb
            else:
                regularized_speedup = (
                    speedup ** (negative_speedup_penalty + 1)
                    if speedup < 1
                    else speedup
                )
            rectified_speedups.append(regularized_speedup)

            # ES(t) calculation: based on state change
            rec_speedup_fake_degrad = 0
            if t_key < 1:
                if fail_type is not None or speedup is None:
                    rec_speedup_fake_degrad = fpdb
                else:
                    rec_speedup_fake_degrad = (
                        speedup ** (negative_speedup_penalty + 1)
                        if speedup < 1
                        else speedup
                    )
            else:
                if not is_correct_at_t1[idx] or speedup_at_t1[idx] is None:
                    fail_type_frozen = fail_type_at_t1[idx]
                    rec_speedup_fake_degrad = fake_perf_degrad(
                        t_key, fail_type_frozen, fpdb
                    )
                else:
                    rec_speedup_fake_degrad = (
                        speedup_at_t1[idx] ** (negative_speedup_penalty + 1)
                        if speedup_at_t1[idx] < 1
                        else speedup_at_t1[idx]
                    )
            rectified_speedups_fake_degrad.append(rec_speedup_fake_degrad)

        if t_key == 1:
            if total_samples == correct_count:
                pi[0] = 0
                pi[1] = 0
            else:
                pi[0] = acc_failure_count / (total_samples - correct_count)
                pi[1] = 1 - pi[0]
            final_correct_count = correct_count
            final_correct_negative_speedup_count = correct_negative_speedup_count
            final_correct_speedups = correct_speedups
            final_slowdown_speedups = slowdown_speedups

        if rectified_speedups:
            s_scores[t_key] = gmean(rectified_speedups)
            s_scores_fake_degrad[t_key] = gmean(rectified_speedups_fake_degrad)
            print(
                f"  - S(t)={s_scores[t_key]:.3f}, ES(t)={s_scores_fake_degrad[t_key]:.3f} for tolerance={t_key} from micro level."
            )
            if t_key < 1:
                expected_s, expected_es = print_stat_info(
                    t_key,
                    correct_count,
                    acc_failure_count,
                    pi,
                    correct_negative_speedup_count,
                    correct_speedups,
                    slowdown_speedups,
                )
            else:
                expected_s, expected_es = print_stat_info(
                    t_key,
                    final_correct_count,
                    acc_failure_count,
                    pi,
                    final_correct_negative_speedup_count,
                    final_correct_speedups,
                    final_slowdown_speedups,
                )
            print(
                f"  - S(t)={expected_s:.3f}, ES(t)={expected_es:.3f} for tolerance={t_key} from macro level."
            )

    print(f"    - pi: {pi}")

    return s_scores, s_scores_fake_degrad
