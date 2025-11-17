import os
import json
import re
import numpy as np
from scipy.stats import gmean
from collections import OrderedDict, defaultdict
from graph_net.config.datatype_tolerance_config import get_precision
from graph_net import samples_statistics
from graph_net.samples_statistics import get_errno_from_error_type


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
                if not file.endswith(".json"):
                    continue

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
                                data_by_subdir[subdir_name].append(speedup_data["e2e"])
                            elif "gpu" in speedup_data:
                                data_by_subdir[subdir_name].append(speedup_data["gpu"])
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


def detect_sample_error_code(log_text: str) -> str:
    """
    Detect the error code for a single sample from log text.

    This function is used for bug subgraph detection. It analyzes log text
    (which can be generated from a single sample) and returns an error code.

    Args:
        log_text: Log text content (can be a string or list of lines)

    Returns:
        Error code string. Possible values:
        - "correct": Sample executed successfully
        - "eager_fail": Eager model execution failed
        - "compile_fail": Compiled model compilation failed
        - "runtime_fail": Runtime error during execution
        - "unknown": Unable to determine error type
    """
    if isinstance(log_text, str):
        lines = log_text.split("\n")
    else:
        lines = log_text

    # Define regex patterns for error detection
    patterns = {
        "result_status": re.compile(r"\[Result\] status: (.+)"),
        "failure": re.compile(r"\[Fail due to (.+)\.\]"),
    }

    # Error type mapping based on failure reason keywords
    error_keywords = {
        "eager": "eager_fail",
        "compiled": "compile_fail",
    }

    for i, line in enumerate(lines):
        result_status_match = patterns["result_status"].search(line)
        if not result_status_match:
            continue

        status = result_status_match.group(1).strip()
        if status == "success":
            return "correct"

        if status != "failed":
            continue

        # Check the next line for failure reason
        if (i + 1) >= len(lines):
            return "runtime_fail"

        error_reason_match = patterns["failure"].search(lines[i + 1])
        if not error_reason_match:
            return "runtime_fail"

        reason = error_reason_match.group(1).lower()
        # Check for specific error keywords
        for keyword, error_code in error_keywords.items():
            if keyword in reason:
                return error_code

        return "runtime_fail"

    return "unknown"


def parse_logs_to_data(log_file: str) -> list:
    """
    Parse a structured log file generated by the benchmark script and
    return a list of data dictionaries (one per model-compiler run).

    This function directly parses log files without generating intermediate JSON files.
    It automatically handles both Paddle (with subgraph) and PyTorch (without subgraph) samples.

    Args:
        log_file: Path to the benchmark log file

    Returns:
        List of data dictionaries, each containing configuration, correctness,
        performance, and result information for a single model-compiler run.
    """
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file}'")
        return []
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []

    # Dictionary to hold parsed data for all runs
    all_runs_data = {}
    current_run_key = None

    # Define regex patterns for each type of log line
    patterns = {
        "processing": re.compile(r"\[Processing\] (.+)"),
        "config": re.compile(r"\[Config\] (\S+): (.+)"),
        "performance": re.compile(r"\[Performance\]\[(\w+)\]: (.+)"),
        "datatype": re.compile(r"\[Datatype\]\[(\w+)\]: (.+)"),
        "correctness": re.compile(r"\[Correctness\](\[.+\]): (.+)"),
        "result_status": re.compile(r"\[Result\] status: (.+)"),
        "failure": re.compile(r"\[Fail due to (.+)\.\]"),
        "speedup": re.compile(r"\[Speedup\]\[(\w+)\]: (.+)"),
    }

    for i, line in enumerate(lines):
        # Check for the start of a new model run
        processing_match = patterns["processing"].search(line)
        if processing_match:
            current_run_key = processing_match.group(1).strip()
            # Initialize a nested dictionary structure for this new run
            all_runs_data[current_run_key] = {
                "configuration": {},
                "correctness": {},
                "performance": {
                    "eager": {},
                    "compiled": {},
                    "datatype": {},
                    "speedup": {},
                },
                "result": {
                    "status": "unknown",
                },
            }
            continue

        # If we haven't identified a run yet, skip the line
        if not current_run_key:
            continue

        # Get the data dictionary for the current run
        data = all_runs_data[current_run_key]

        # Try to match other patterns
        config_match = patterns["config"].search(line)
        if config_match:
            key, value = config_match.groups()
            data["configuration"][key.strip()] = value.strip()
            continue

        performance_match = patterns["performance"].search(line)
        if performance_match:
            key, value_str = performance_match.groups()
            # The performance value is a JSON string, so we load it
            data["performance"][key.strip()] = json.loads(value_str)
            continue

        datatype_match = patterns["datatype"].search(line)
        if datatype_match:
            key, value_str = datatype_match.groups()
            # The datatype value is a space-separated string
            data["performance"]["datatype"][key.strip()] = value_str.strip().split()
            continue

        correctness_match = patterns["correctness"].search(line)
        if correctness_match:
            key, value_str = correctness_match.groups()
            values = []
            for v in value_str.strip().split():
                try:
                    # Try to convert to int if it's a whole number, else float
                    values.append(int(v) if "." not in v else float(v))
                except ValueError:
                    # Handle non-numeric values like 'nan'
                    values.append(float(v))
            data["correctness"][key.strip()] = values
            continue

        # Check for speedup
        speedup_match = patterns["speedup"].search(line)
        if speedup_match:
            key, value_str = speedup_match.groups()
            data["performance"]["speedup"][key.strip()] = float(value_str)
            continue

        # Look for the status, and if it's "failed", look ahead to the next line.
        result_status_match = patterns["result_status"].search(line)
        if not result_status_match:
            continue

        status = result_status_match.group(1).strip()
        data["result"]["status"] = status
        if status != "failed" or (i + 1) >= len(lines):
            continue

        error_reason_match = patterns["failure"].search(lines[i + 1])
        if not error_reason_match:
            continue

        reason = error_reason_match.group(1).lower()
        if "eager" in reason:
            data["performance"]["failure"] = "eager"
            data["result"]["status"] = "eager_fail"
        elif "compiled" in reason:
            data["performance"]["failure"] = "compiled"
            data["result"]["status"] = "compile_fail"
        else:
            data["performance"]["failure"] = "other"
            data["result"]["status"] = "runtime_fail"
        continue

    # After parsing all lines, process the results
    if not all_runs_data:
        print("No processable log entries found in the file.")
        return []

    samples = []
    for run_key, data in all_runs_data.items():
        try:
            speedup_dict = data["performance"].get("speedup", {})

            # Build result field with status and speedup (for compatibility with log2json output format)
            if data["result"]["status"] == "success" and speedup_dict:
                speedup_data = {}
                for key in ["e2e", "gpu"]:
                    if key in speedup_dict:
                        speedup_data[key] = {"mean": speedup_dict[key]}
                if speedup_data:
                    data["result"]["speedup"] = speedup_data

            # Ensure performance.speedup.e2e/gpu are direct values (not nested dict)
            # This is required by calculate_s_scores which uses performance_data.get("speedup", {}).get("e2e")
            for key in ["e2e", "gpu"]:
                if key in speedup_dict:
                    val = speedup_dict[key]
                    if isinstance(val, dict) and "mean" in val:
                        speedup_dict[key] = val["mean"]

            samples.append(data)

        except KeyError as e:
            print(f"Warning: Could not process run '{run_key}' due to missing key: {e}")
        except Exception as e:
            print(
                f"Warning: An unexpected error occurred while processing run '{run_key}': {e}"
            )

    print(f"Successfully parsed {len(samples)} samples from log file: {log_file}")
    return samples


def scan_all_folders(benchmark_path: str) -> dict:
    """
    Unified entry point that supports log files and directories:
      - If benchmark_path is a log file (.log or .txt) → parse it directly and return data as a single curve.
      - If benchmark_path is a directory → scan for .log and .txt files in the directory,
        each log file becomes a curve.
    Returns dict[curve_name] -> list_of_samples
    """
    # Handle single log file
    if os.path.isfile(benchmark_path):
        print(f"Detected log file: '{benchmark_path}'")
        samples = parse_logs_to_data(benchmark_path)
        if not samples:
            print(f"  - No valid data found in log file.")
            return {}

        folder_name = (
            os.path.splitext(os.path.basename(benchmark_path))[0] or "benchmark"
        )
        print(
            f"  - Parsed log file → 1 curve '{folder_name}' "
            f"with {len(samples)} samples."
        )
        return {folder_name: samples}

    # Check if it's a directory
    if not os.path.isdir(benchmark_path):
        print(
            f"Error: Provided path '{benchmark_path}' is neither a valid file nor directory."
        )
        return {}

    print(f"Scanning '{benchmark_path}' ...")

    # Find .log and .txt files in the directory
    log_files = sorted(
        [
            f
            for f in os.listdir(benchmark_path)
            if os.path.isfile(os.path.join(benchmark_path, f))
            and f.endswith((".log", ".txt"))
        ]
    )

    if not log_files:
        print("  - No log files (.log or .txt) found in directory.")
        return {}

    # Process log files, each becomes a curve
    all_results = {}
    print(f"  - Found {len(log_files)} log file(s) → each becomes a curve.")
    for log_file in log_files:
        log_file_path = os.path.join(benchmark_path, log_file)
        samples = parse_logs_to_data(log_file_path)
        if not samples:
            continue

        curve_name = os.path.splitext(log_file)[0] or "benchmark"
        all_results[curve_name] = samples
        print(f"    - Curve '{curve_name}': {len(samples)} samples.")

    if not all_results:
        print("  - No valid data found in any log file.")
        return {}

    print(f"Total curves loaded: {len(all_results)}")
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


def check_sample_correctness(sample: dict, t_key: int) -> tuple[bool, str]:
    """
    Check if a sample is correct at the given tolerance level.

    Args:
        sample: Sample data dictionary
        t_key: Tolerance level

    Returns:
        Tuple of (is_correct, fail_type)
        - is_correct: True if sample is correct at this tolerance
        - fail_type: Error type if not correct, None if correct
    """
    performance_data = sample.get("performance", {})
    fail_type = performance_data.get("failure")

    # If there's already a failure type, return it
    if fail_type is not None:
        return False, fail_type

    # Check correctness based on datatype and tolerance
    datatype_data = performance_data.get("datatype", {})
    eager_dtypes = datatype_data.get("eager", [])
    compiled_dtypes = datatype_data.get("compiled", [])

    # Check if datatypes match and are valid
    if not (eager_dtypes and eager_dtypes == compiled_dtypes and len(eager_dtypes) > 0):
        return False, "accuracy"

    correctness_data = sample.get("correctness", {})
    output_count = len(correctness_data.get("[equal]", []))

    if len(eager_dtypes) != output_count:
        return False, "accuracy"

    # Check all outputs for correctness
    is_correct = all(
        get_correctness(eager_dtypes[i], t_key, correctness_data, i)
        for i in range(output_count)
    )

    return is_correct, None if is_correct else "accuracy"


def calculate_rectified_speedup(
    speedup: float, fail_type: str, negative_speedup_penalty: float, fpdb: float
) -> float:
    """
    Calculate rectified speedup for S(t) calculation.

    Args:
        speedup: Original speedup value
        fail_type: Error type or None if correct
        negative_speedup_penalty: Penalty power p for negative speedup
        fpdb: Base penalty for failures

    Returns:
        Rectified speedup value
    """
    if fail_type is not None or speedup is None:
        return fpdb

    if speedup < 1:
        return speedup ** (negative_speedup_penalty + 1)
    return speedup


def calculate_es_rectified_speedup(
    speedup: float,
    fail_type: str,
    t_key: int,
    is_correct_at_t1: bool,
    speedup_at_t1: float,
    fail_type_at_t1: str,
    negative_speedup_penalty: float,
    fpdb: float,
) -> float:
    """
    Calculate rectified speedup for ES(t) calculation.

    Args:
        speedup: Current speedup value
        fail_type: Current error type
        t_key: Current tolerance level
        is_correct_at_t1: Whether sample was correct at t=1
        speedup_at_t1: Speedup value at t=1
        fail_type_at_t1: Error type at t=1
        negative_speedup_penalty: Penalty power p
        fpdb: Base penalty for failures

    Returns:
        Error-aware rectified speedup value
    """
    if t_key < 1:
        # For t < 1, ES(t) = S(t)
        return calculate_rectified_speedup(
            speedup, fail_type, negative_speedup_penalty, fpdb
        )

    # For t >= 1, use frozen state from t=1
    if not is_correct_at_t1 or speedup_at_t1 is None:
        return fake_perf_degrad(t_key, fail_type_at_t1, fpdb)

    if speedup_at_t1 < 1:
        return speedup_at_t1 ** (negative_speedup_penalty + 1)
    return speedup_at_t1


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
    # Store aggregated-level calculation results for cross-validation
    s_scores._aggregated_results = OrderedDict()
    s_scores_fake_degrad._aggregated_results = OrderedDict()

    begin = -10
    end = 4
    t_keys = list(range(begin, end + 1))
    total_samples = len(samples)

    print(f"\nCalculating S(t) scores for '{folder_name}'...")

    def print_stat_info(
        t_key,
        correct_count,
        errno2count,
        pi,
        correct_negative_speedup_count,
        correct_speedups,
    ):
        """
        Calculate and print aggregated statistics for a given tolerance level.

        Uses the samples_statistics module for all parameter calculations.
        """
        print(f"  - Details for tolerance={t_key}:")
        if total_samples > 0:
            # Calculate all aggregated parameters using the dedicated module
            aggregated_params = samples_statistics.calculate_all_aggregated_parameters(
                total_samples=total_samples,
                correct_speedups=correct_speedups,
                errno2count=errno2count,
                t_key=t_key,
                negative_speedup_penalty=negative_speedup_penalty,
                fpdb=fpdb,
                pi=pi,
            )

            alpha = aggregated_params["alpha"]
            beta = aggregated_params["beta"]
            lambda_ = aggregated_params["lambda"]
            eta = aggregated_params["eta"]
            gamma = aggregated_params["gamma"]
            expected_s = aggregated_params["s_t"]
            expected_es = aggregated_params["es_t"]

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
            expected_s = fpdb
            expected_es = fpdb

        return expected_s, expected_es

    # pi is a tuple of constants for t > 0 for each group: (pi[0], pi[1])
    # Calculated at t=1, used for all t >= 1
    pi = (0.0, 0.0)

    is_correct_at_t1 = [False] * total_samples
    speedup_at_t1 = [None] * total_samples
    fail_type_at_t1 = ["CORRECT"] * total_samples

    final_correct_count = 0
    final_correct_negative_speedup_count = 0
    final_correct_speedups = []
    final_errno2count = {}  # Store error type counts at t=1 (using errno)

    for t_key in t_keys:
        rectified_speedups = []
        rectified_speedups_fake_degrad = []
        correct_count = 0
        errno2count = {}  # Dictionary to count errors by errno
        correct_negative_speedup_count = 0
        correct_speedups = []

        # Process all samples using helper functions to reduce nesting
        for idx, sample in enumerate(samples):
            performance_data = sample.get("performance", {})
            speedup = performance_data.get("speedup", {}).get("e2e")

            # Check correctness using dedicated function
            is_correct, fail_type = check_sample_correctness(sample, t_key)

            # Collect statistics
            if is_correct:
                correct_count += 1
                if speedup is not None:
                    correct_speedups.append(speedup)
                if speedup is not None and speedup < 1:
                    correct_negative_speedup_count += 1

            # Count errors by errno (convert error type string to errno)
            if fail_type is not None:
                errno = get_errno_from_error_type(fail_type)
                errno2count[errno] = errno2count.get(errno, 0) + 1

            # Store state at t=1 for ES(t) calculation
            if t_key == 1:
                is_correct_at_t1[idx] = is_correct
                speedup_at_t1[idx] = speedup
                fail_type_at_t1[idx] = fail_type if fail_type is not None else "CORRECT"

            # Calculate rectified speedups using dedicated functions
            regularized_speedup = calculate_rectified_speedup(
                speedup, fail_type, negative_speedup_penalty, fpdb
            )
            rectified_speedups.append(regularized_speedup)

            rec_speedup_fake_degrad = calculate_es_rectified_speedup(
                speedup,
                fail_type,
                t_key,
                is_correct_at_t1[idx],
                speedup_at_t1[idx],
                fail_type_at_t1[idx],
                negative_speedup_penalty,
                fpdb,
            )
            rectified_speedups_fake_degrad.append(rec_speedup_fake_degrad)

        if t_key == 1:
            # Calculate pi at t=1 using the dedicated function
            pi = samples_statistics.calculate_pi(
                errno2count, total_samples, correct_speedups
            )
            final_correct_count = correct_count
            final_correct_negative_speedup_count = correct_negative_speedup_count
            final_correct_speedups = correct_speedups
            final_errno2count = errno2count.copy()  # Save for t >= 1

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
                    errno2count,
                    pi,
                    correct_negative_speedup_count,
                    correct_speedups,
                )
            else:
                # For t >= 1, use errno2count from t=1 (frozen state)
                expected_s, expected_es = print_stat_info(
                    t_key,
                    final_correct_count,
                    final_errno2count,  # Use the frozen errno2count from t=1
                    pi,
                    final_correct_negative_speedup_count,
                    final_correct_speedups,
                )
            print(
                f"  - S(t)={expected_s:.3f}, ES(t)={expected_es:.3f} for tolerance={t_key} from aggregated level."
            )
            # Store aggregated results for cross-validation
            s_scores._aggregated_results[t_key] = expected_s
            s_scores_fake_degrad._aggregated_results[t_key] = expected_es

    print(f"    - pi: {dict(sorted(pi.items()))}")

    return s_scores, s_scores_fake_degrad
