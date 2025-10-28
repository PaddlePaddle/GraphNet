import argparse
import json
import os
import re
import numpy as np
from collections import defaultdict
import glob
import paddle
import io
import sys
import contextlib
from graph_net import test_compiler_util


def get_cmp_equal(expected_out, compiled_out):
    """Callback for test_compiler_util.check_equal."""

    def convert(x):
        if not isinstance(x, paddle.Tensor):
            return x
        if x.dtype in [paddle.float16, paddle.bfloat16]:
            return x.astype("float32")
        elif x.dtype in [paddle.uint8, paddle.int8, paddle.int16]:
            return x.astype("int32")
        return x

    results = []
    if len(expected_out) != len(compiled_out):
        print(f"ERROR: Output list length mismatch for get_cmp_equal", file=sys.stderr)
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append(str(int(a is b)))
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append(str(int(a == b)))
        elif a.shape != b.shape:
            print(f"WARNING: Shape mismatch in get_cmp_equal", file=sys.stderr)
            results.append("0")
        else:
            try:
                results.append(str(int(paddle.equal_all(convert(a), convert(b)))))
            except Exception as e:
                print(f"ERROR: during paddle.equal_all: {e}", file=sys.stderr)
                results.append("error")
    return " ".join(results)


def get_cmp_all_close(expected_out, compiled_out, atol, rtol):
    """Callback for test_compiler_util.check_allclose."""
    results = []
    if len(expected_out) != len(compiled_out):
        print(
            f"ERROR: Output list length mismatch for get_cmp_all_close", file=sys.stderr
        )
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append(str(int(a is b)))
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append(str(int(a == b)))
        elif a.shape != b.shape:
            print(f"WARNING: Shape mismatch in get_cmp_all_close", file=sys.stderr)
            results.append("0")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            print(
                f"WARNING: Non-floating point tensor in get_cmp_all_close (types: {a.dtype}, {b.dtype}). Comparing with equal_all.",
                file=sys.stderr,
            )
            try:
                results.append(str(int(paddle.equal_all(a, b))))
            except Exception as e:
                print(
                    f"ERROR: during fallback equal_all in allclose: {e}",
                    file=sys.stderr,
                )
                results.append("error")
        else:
            try:
                results.append(str(int(paddle.allclose(a, b, atol=atol, rtol=rtol))))
            except Exception as e:
                print(
                    f"ERROR: during paddle.allclose (atol={atol}, rtol={rtol}): {e}",
                    file=sys.stderr,
                )
                results.append("error")
    return " ".join(results)


def get_format_str(f):
    """Formats a float for consistent output."""
    try:
        f_float = float(f)
        if (abs(f_float) > 1e5 or abs(f_float) < 1e-5) and abs(f_float) != 0.0:
            return str(f"{f_float:.5E}")
        else:
            return str(f"{f_float:.5f}")
    except (ValueError, TypeError):
        print(f"WARNING: Could not format value {f} as float.", file=sys.stderr)
        return str(f)


def get_cmp_max_diff(expected_out, compiled_out):
    """Callback for test_compiler_util.check_allclose."""
    results = []
    if len(expected_out) != len(compiled_out):
        print(
            f"ERROR: Output list length mismatch for get_cmp_max_diff", file=sys.stderr
        )
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append("N/A")
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append("N/A")
        elif a.shape != b.shape:
            print(f"WARNING: Shape mismatch in get_cmp_max_diff", file=sys.stderr)
            results.append("shape_mismatch")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            print(
                f"WARNING: Non-floating point tensor in get_cmp_max_diff (types: {a.dtype}, {b.dtype})",
                file=sys.stderr,
            )
            results.append("non_float")
        else:
            try:
                a_f32 = (
                    a.astype("float32")
                    if a.dtype not in [paddle.float32, paddle.float64]
                    else a
                )
                b_f32 = (
                    b.astype("float32")
                    if b.dtype not in [paddle.float32, paddle.float64]
                    else b
                )
                results.append(
                    get_format_str(paddle.max(paddle.abs(a_f32 - b_f32)).item())
                )
            except Exception as e:
                print(f"ERROR: during max_diff calculation: {e}", file=sys.stderr)
                results.append("error")
    return " ".join(results)


def get_cmp_mean_diff(expected_out, compiled_out):
    """Callback for test_compiler_util.check_allclose."""
    results = []
    if len(expected_out) != len(compiled_out):
        print(
            f"ERROR: Output list length mismatch for get_cmp_mean_diff", file=sys.stderr
        )
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append("N/A")
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append("N/A")
        elif a.shape != b.shape:
            print(f"WARNING: Shape mismatch in get_cmp_mean_diff", file=sys.stderr)
            results.append("shape_mismatch")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            print(
                f"WARNING: Non-floating point tensor in get_cmp_mean_diff (types: {a.dtype}, {b.dtype})",
                file=sys.stderr,
            )
            results.append("non_float")
        else:
            try:
                a_f32 = (
                    a.astype("float32")
                    if a.dtype not in [paddle.float32, paddle.float64]
                    else a
                )
                b_f32 = (
                    b.astype("float32")
                    if b.dtype not in [paddle.float32, paddle.float64]
                    else b
                )
                results.append(
                    get_format_str(paddle.mean(paddle.abs(a_f32 - b_f32)).item())
                )
            except Exception as e:
                print(f"ERROR: during mean_diff calculation: {e}", file=sys.stderr)
                results.append("error")
    return " ".join(results)


def get_cmp_max_relative_diff(expected_out, compiled_out):
    """Callback for test_compiler_util.check_allclose."""
    epsilon = 1e-8
    results = []
    if len(expected_out) != len(compiled_out):
        print(
            f"ERROR: Output list length mismatch for get_cmp_max_relative_diff",
            file=sys.stderr,
        )
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append("N/A")
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append("N/A")
        elif a.shape != b.shape:
            print(
                f"WARNING: Shape mismatch in get_cmp_max_relative_diff", file=sys.stderr
            )
            results.append("shape_mismatch")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            print(
                f"WARNING: Non-floating point tensor in get_cmp_max_relative_diff (types: {a.dtype}, {b.dtype})",
                file=sys.stderr,
            )
            results.append("non_float")
        else:
            try:
                a_f32 = (
                    a.astype("float32")
                    if a.dtype not in [paddle.float32, paddle.float64]
                    else a
                )
                b_f32 = (
                    b.astype("float32")
                    if b.dtype not in [paddle.float32, paddle.float64]
                    else b
                )
                denom = paddle.abs(a_f32) + epsilon
                relative_diff = paddle.abs(a_f32 - b_f32) / denom
                max_val = paddle.max(
                    paddle.where(
                        paddle.isnan(relative_diff) | paddle.isinf(relative_diff),
                        paddle.to_tensor(0.0),
                        relative_diff,
                    )
                ).item()
                results.append(
                    get_format_str(max_val) if not np.isnan(max_val) else "NaN/Inf"
                )
            except Exception as e:
                print(
                    f"ERROR: during max_relative_diff calculation: {e}", file=sys.stderr
                )
                results.append("error")
    return " ".join(results)


def get_cmp_mean_relative_diff(expected_out, compiled_out):
    """Callback for test_compiler_util.check_allclose."""
    epsilon = 1e-8
    results = []
    if len(expected_out) != len(compiled_out):
        print(
            f"ERROR: Output list length mismatch for get_cmp_mean_relative_diff",
            file=sys.stderr,
        )
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append("N/A")
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append("N/A")
        elif a.shape != b.shape:
            print(
                f"WARNING: Shape mismatch in get_cmp_mean_relative_diff",
                file=sys.stderr,
            )
            results.append("shape_mismatch")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            print(
                f"WARNING: Non-floating point tensor in get_cmp_mean_relative_diff (types: {a.dtype}, {b.dtype})",
                file=sys.stderr,
            )
            results.append("non_float")
        else:
            try:
                a_f32 = (
                    a.astype("float32")
                    if a.dtype not in [paddle.float32, paddle.float64]
                    else a
                )
                b_f32 = (
                    b.astype("float32")
                    if b.dtype not in [paddle.float32, paddle.float64]
                    else b
                )
                denom = paddle.abs(a_f32) + epsilon
                relative_diff = paddle.abs(a_f32 - b_f32) / denom
                valid_diff = paddle.where(
                    paddle.isnan(relative_diff) | paddle.isinf(relative_diff),
                    paddle.to_tensor(0.0),
                    relative_diff,
                )
                mean_val = paddle.mean(valid_diff).item()
                results.append(
                    get_format_str(mean_val) if not np.isnan(mean_val) else "NaN/Inf"
                )
            except Exception as e:
                print(
                    f"ERROR: during mean_relative_diff calculation: {e}",
                    file=sys.stderr,
                )
                results.append("error")
    return " ".join(results)


def standardize_model_path(raw_path: str) -> str:
    """Standardizes the model path by removing common prefixes."""
    graphnet_pattern = re.compile(r".*GraphNet/(.+)")
    match = graphnet_pattern.search(raw_path)
    if match:
        return f"GraphNet/{match.group(1)}"

    print(
        f"WARNING: Unknown model path prefix for standardization: {raw_path}. Using part after the last '/'.",
        file=sys.stderr,
    )
    parts = raw_path.split("/")
    if len(parts) > 1 and parts[-1].startswith("subgraph_"):
        return "/".join(parts[-2:])
    elif len(parts) > 0:
        return parts[-1]
    else:
        return raw_path


def parse_single_log_file(log_filepath: str) -> dict:
    """Parses performance and config data from a single log file."""
    if not os.path.exists(log_filepath):
        print(f"ERROR: Log file not found: {log_filepath}", file=sys.stderr)
        return None

    try:
        with open(log_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"ERROR: Error reading log file {log_filepath}: {e}", file=sys.stderr)
        return None

    run_data = {"config": {}, "results": {}, "original_path": ""}
    model_path_found = None
    model_run_failed = False

    patterns = {
        "model_path": re.compile(
            r"\[?\d*\]?\s*fixed_random_seed_device_runner, model_path: (.+)"
        ),
        "config": re.compile(r"\[Config\] (\S+): (.+)"),
        "result": re.compile(r"\[Result\] (\S+): (.+)"),
        "error": re.compile(r"Run model failed:"),
    }

    for line in lines:
        line = line.strip()
        if model_run_failed:
            continue

        if not model_path_found:
            model_path_match = patterns["model_path"].search(line)
            if model_path_match:
                model_path_found = model_path_match.group(1).strip()
                run_data["original_path"] = model_path_found

        if model_path_found:
            if patterns["error"].search(line):
                print(
                    f"WARNING: Detected failure for model: {model_path_found} in {log_filepath}."
                )
                model_run_failed = True
                run_data["results"]["failure"] = "run_model_failed"

            config_match = patterns["config"].search(line)
            if config_match:
                key, value = config_match.groups()
                run_data["config"][key.strip()] = value.strip()
                continue

            result_match = patterns["result"].search(line)
            if result_match:
                key, value_str = result_match.groups()
                key = key.strip()
                value_str = value_str.strip()
                if key == "model_path":
                    if value_str != model_path_found:
                        print(
                            f"WARNING: Result block model path '{value_str}' does not match header '{model_path_found}' in {log_filepath}. Using header path."
                        )
                elif key in ["e2e_mean", "e2e_std"]:
                    try:
                        run_data["results"][key] = float(value_str)
                    except ValueError:
                        print(
                            f"WARNING: Could not parse float value '{value_str}' for key '{key}' in {log_filepath}"
                        )
                else:
                    run_data["results"][key] = value_str

    if not model_path_found:
        print(f"WARNING: No model_path found in {log_filepath}. Skipping.")
        return None

    if "e2e_mean" not in run_data["results"] and not model_run_failed:
        print(
            f"WARNING: Incomplete data parsed (missing e2e_mean) from {log_filepath} and not marked as failed. Skipping."
        )
        return None

    return run_data


def load_outputs(output_filepath: str) -> list:
    """Loads outputs from a JSON file and converts to Paddle Tensors."""
    if not os.path.exists(output_filepath):
        print(f"WARNING: Output file not found: {output_filepath}", file=sys.stderr)
        return []
    try:
        with open(output_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(
                f"ERROR: Unexpected data format in {output_filepath}. Expected a list.",
                file=sys.stderr,
            )
            return []

        tensors = []
        for item in data:
            try:
                if item is None:
                    tensors.append(None)
                else:
                    tensors.append(paddle.to_tensor(item, dtype="float32"))
            except Exception as e:
                print(
                    f"ERROR: Error converting item to tensor in {output_filepath}: {item} - {e}",
                    file=sys.stderr,
                )
                tensors.append(None)
        return tensors
    except json.JSONDecodeError:
        print(f"ERROR: Error decoding JSON from {output_filepath}", file=sys.stderr)
        return []
    except Exception as e:
        print(
            f"ERROR: Error loading or processing {output_filepath}: {e}",
            file=sys.stderr,
        )
        return []


def perform_correctness_checks(baseline_tensors: list, target_tensors: list) -> dict:
    """
    Performs correctness checks using test_compiler_util and returns results.
    """
    MockArgs = argparse.Namespace(log_prompt="[Correctness]")
    stderr_capture = io.StringIO()
    correctness_results = {}

    try:
        with contextlib.redirect_stderr(stderr_capture):
            baseline_dtypes = [
                str(t.dtype).replace("paddle.", "") if t is not None else "none"
                for t in baseline_tensors
            ]
            target_dtypes = [
                str(t.dtype).replace("paddle.", "") if t is not None else "none"
                for t in target_tensors
            ]

            type_match = test_compiler_util.check_output_datatype(
                MockArgs, baseline_dtypes, target_dtypes
            )

            def transfer_to_float(origin_outputs):
                outputs = []
                for item in origin_outputs:
                    if (
                        item is not None
                        and isinstance(item, paddle.Tensor)
                        and item.dtype not in [paddle.float32, paddle.float64]
                        and paddle.is_floating_point(item)
                    ):
                        try:
                            item = item.astype("float32")
                        except Exception as e:
                            print(
                                f"WARNING: Could not cast tensor to float32: {e}",
                                file=sys.stderr,
                            )
                    outputs.append(item)
                return outputs

            if len(baseline_tensors) != len(target_tensors):
                print(
                    "WARNING: Output tensor list lengths differ, skipping detailed comparison."
                )
                correctness_results["error"] = "Output count mismatch"
                output_str = stderr_capture.getvalue()
            elif not baseline_tensors and not target_tensors:
                print(
                    "WARNING: Both output tensor lists are empty. Skipping correctness checks."
                )
                correctness_results["info"] = "Empty outputs"
                output_str = stderr_capture.getvalue()
            else:
                if type_match:
                    test_compiler_util.check_equal(
                        MockArgs,
                        baseline_tensors,
                        target_tensors,
                        cmp_equal_func=get_cmp_equal,
                    )

                    baseline_out_fp32 = transfer_to_float(baseline_tensors)
                    target_out_fp32 = transfer_to_float(target_tensors)

                    test_compiler_util.check_allclose(
                        MockArgs,
                        baseline_out_fp32,
                        target_out_fp32,
                        cmp_all_close_func=get_cmp_all_close,
                        cmp_max_diff_func=get_cmp_max_diff,
                        cmp_mean_diff_func=get_cmp_mean_diff,
                        cmp_max_relative_diff_func=get_cmp_max_relative_diff,
                        cmp_mean_relative_diff_func=get_cmp_mean_relative_diff,
                    )
                else:
                    print(
                        f"WARNING: Skipping correctness checks due to type mismatch: {baseline_dtypes} vs {target_dtypes}"
                    )
                    correctness_results["error"] = "Datatype mismatch"
                    correctness_results["eager_dtypes"] = baseline_dtypes
                    correctness_results["compiled_dtypes"] = target_dtypes

                output_str = stderr_capture.getvalue()

        correctness_pattern = re.compile(r"\[Correctness\](\[.+?\]): (.+)")
        datatype_pattern = re.compile(r"\[Datatype\]\[(\w+)\]: (.+)")

        if "datatype" not in correctness_results:
            correctness_results["datatype"] = {}

        for line in output_str.splitlines():
            corr_match = correctness_pattern.search(line)
            data_match = datatype_pattern.search(line)

            if corr_match:
                key = corr_match.group(1).strip()
                value_str = corr_match.group(2).strip()
                value_list = value_str.split()

                if key.startswith("[all_close") or key == "[equal]":
                    try:
                        correctness_results[key] = [
                            int(v) if v.isdigit() else v for v in value_list
                        ]
                    except ValueError:
                        correctness_results[key] = value_list
                elif key in [
                    "[max_diff]",
                    "[mean_diff]",
                    "[max_relative_diff]",
                    "[mean_relative_diff]",
                ]:
                    converted_list = []
                    for v in value_list:
                        try:
                            converted_list.append(float(v))
                        except (ValueError, TypeError):
                            converted_list.append(v)
                    correctness_results[key] = converted_list
                else:
                    correctness_results[key] = value_list

            elif data_match:
                key = data_match.group(1).strip()  # eager or compiled
                value_str = data_match.group(2).strip()
                correctness_results["datatype"][key] = value_str.split()

    except ImportError:
        print(
            "ERROR: PaddlePaddle or graph_net.test_compiler_util not found. Cannot perform correctness checks.",
            file=sys.stderr,
        )
        correctness_results = {
            "error": "PaddlePaddle or test_compiler_util not installed"
        }
    except Exception as e:
        print(f"ERROR: Error during correctness check: {e}", file=sys.stderr)
        correctness_results = {"error": f"Correctness check failed: {e}"}
        output_str = stderr_capture.getvalue()
        if output_str:
            correctness_results["captured_stderr"] = output_str

    if (
        not correctness_results
        or (
            len(correctness_results) == 1
            and ("error" in correctness_results or "info" in correctness_results)
        )
        or (
            len(correctness_results) == 2
            and ("error" in correctness_results and "datatype" in correctness_results)
        )
    ):
        if "error" in correctness_results:
            print(
                f"WARNING: Populating correctness section with empty placeholders due to error: {correctness_results['error']}"
            )
        elif "info" in correctness_results:
            print(
                f"WARNING: Populating correctness section with empty placeholders: {correctness_results['info']}"
            )
        else:
            print(
                "WARNING: Populating correctness section with empty placeholders (no results captured)."
            )

        placeholder_keys = [
            "[equal]",
            "[max_diff]",
            "[mean_diff]",
            "[max_relative_diff]",
            "[mean_relative_diff]",
        ]

        try:
            allclose_tolerances = test_compiler_util.calculate_tolerance_pair(-10, 5)
            for pair in allclose_tolerances:
                atol, rtol = pair["atol"], pair["rtol"]
                placeholder_keys.append(f"[all_close_atol_{atol:.2E}_rtol_{rtol:.2E}]")
        except Exception as e:
            print(
                f"ERROR: Failed to use test_compiler_util.calculate_tolerance_pair for placeholders: {e}.",
                file=sys.stderr,
            )
            allclose_tolerances = []

        num_outputs = max(1, len(baseline_tensors), len(target_tensors))
        if correctness_results.get("error") == "Output count mismatch":
            num_outputs = 1

        placeholder_list = []
        # placeholder_list = [1] * num_outputs

        for key in placeholder_keys:
            if key not in correctness_results:
                correctness_results[key] = placeholder_list

        if "datatype" not in correctness_results:
            correctness_results["datatype"] = {
                "eager": ["N/A"] * len(baseline_tensors) if baseline_tensors else [],
                "compiled": ["N/A"] * len(target_tensors) if target_tensors else [],
            }

    return correctness_results


def find_log_output_pairs(log_dir: str) -> dict:
    """Finds matching .log and _outputs.json files in a directory.
    Also includes logs where the output file is missing."""
    pairs = {}
    if not os.path.isdir(log_dir):
        print(
            f"ERROR: Provided log path is not a directory: {log_dir}", file=sys.stderr
        )
        return pairs

    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    print(f"INFO: Scanning {len(log_files)} log files in {log_dir}...")

    for log_file in log_files:
        base_name = os.path.basename(log_file)[:-4]  # Remove .log extension
        output_file = os.path.join(log_dir, f"{base_name}_outputs.json")

        if os.path.exists(output_file):
            pairs[base_name] = {"log": log_file, "outputs": output_file}
        else:
            print(
                f"WARNING: {base_name}: Log file '{log_file}' found but missing corresponding output file '{output_file}'."
            )
            pairs[base_name] = {
                "log": log_file,
                "outputs": None,
            }  # Store with None output

    print(f"INFO: Found {len(pairs)} log entries in {log_dir}")
    return pairs


def generate_comparison_json(target_pairs: dict, baseline_pairs: dict, output_dir: str):
    """
    Processes pairs of logs/outputs, performs comparisons, and writes JSON reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_count = 0

    all_base_names = set(target_pairs.keys()) | set(baseline_pairs.keys())
    print(f"INFO: Found {len(all_base_names)} unique model entries to process.")

    for base_name in all_base_names:
        target_files = target_pairs.get(base_name)
        baseline_files = baseline_pairs.get(base_name)

        target_run_data = None
        baseline_run_data = None
        target_outputs = None
        baseline_outputs = None

        failure_reason = None
        correctness_results = {}

        target_device_name = "target_device"
        baseline_device_name = "baseline_device"

        # --- Process Target Device Data ---
        if target_files:
            target_run_data = parse_single_log_file(target_files["log"])
            if target_run_data:
                target_device_name = target_run_data.get("config", {}).get(
                    "device", "target_device"
                )

            if not target_run_data:
                failure_reason = f"{target_device_name}_log_parse_fail"
                print(
                    f"WARNING: {base_name}: Failed to parse log for {target_device_name}."
                )
            elif target_run_data.get("results", {}).get("failure"):
                failure_reason = f"{target_device_name}_run_model_failed"
                print(
                    f"WARNING: {base_name}: Log indicates {target_device_name} run failed."
                )
            elif target_files["outputs"] is None:
                print(
                    f"WARNING: {base_name}: Missing output file for {target_device_name}."
                )
                failure_reason = f"{target_device_name}_output_missing"
            else:
                target_outputs = load_outputs(target_files["outputs"])
                if not target_outputs and os.path.exists(target_files["outputs"]):
                    print(
                        f"WARNING: {base_name}: Failed to load/parse output file for {target_device_name}."
                    )
                    failure_reason = f"{target_device_name}_output_load_fail"
        else:
            print(f"WARNING: {base_name}: No data found for target device.")
            failure_reason = "target_data_missing_flag"

        # --- Process Baseline (CUDA) Data ---
        if baseline_files:
            baseline_run_data = parse_single_log_file(baseline_files["log"])
            if baseline_run_data:
                baseline_device_name = baseline_run_data.get("config", {}).get(
                    "device", "baseline_device"
                )

            if not baseline_run_data:
                if not failure_reason:
                    failure_reason = f"{baseline_device_name}_log_parse_fail"
                print(
                    f"WARNING: {base_name}: Failed to parse log for {baseline_device_name}."
                )
            elif baseline_run_data.get("results", {}).get("failure"):
                if not failure_reason:
                    failure_reason = f"{baseline_device_name}_run_model_failed"
                print(
                    f"WARNING: {base_name}: Log indicates {baseline_device_name} run failed."
                )
            elif baseline_files["outputs"] is None:
                if not failure_reason:
                    failure_reason = f"{baseline_device_name}_output_missing"
                print(
                    f"WARNING: {base_name}: Missing output file for {baseline_device_name}."
                )
            else:
                baseline_outputs = load_outputs(baseline_files["outputs"])
                if not baseline_outputs and os.path.exists(baseline_files["outputs"]):
                    if not failure_reason:
                        failure_reason = f"{baseline_device_name}_output_load_fail"
                    print(
                        f"WARNING: {base_name}: Failed to load/parse output file for {baseline_device_name}."
                    )
        else:
            print(f"WARNING: {base_name}: No data found for baseline device.")
            if not failure_reason:
                failure_reason = f"{baseline_device_name}_data_missing"

        # --- Refine device names and failure reasons ---
        if target_run_data and not baseline_run_data:
            baseline_device_name = "cuda"  # Guess name
            if failure_reason == f"{baseline_device_name}_log_parse_fail":
                pass
            elif not baseline_files:
                failure_reason = f"{baseline_device_name}_data_missing"

        if not target_run_data and baseline_run_data:
            target_device_name = "target_device"  # Guess name
            if failure_reason == f"{target_device_name}_log_parse_fail":
                pass
            elif failure_reason == "target_data_missing_flag":
                failure_reason = f"{target_device_name}_data_missing"

        if failure_reason == "target_data_missing_flag":
            failure_reason = f"{target_device_name}_data_missing"

        # --- Skip if both failed to parse (no config) ---
        if not target_run_data and not baseline_run_data:
            print(
                f"ERROR: Skipping '{base_name}': Both log files failed to parse or are missing.",
                file=sys.stderr,
            )
            continue

        # --- Perform Correctness Checks ---
        if failure_reason:
            correctness_results = {}  # Empty dict for failures
        elif target_outputs and baseline_outputs:
            try:
                correctness_results = perform_correctness_checks(
                    baseline_outputs, target_outputs
                )
            except Exception as e:
                print(
                    f"ERROR: {base_name}: Error during correctness check: {e}",
                    file=sys.stderr,
                )
                correctness_results = {"error": f"Correctness check failed: {e}"}
                failure_reason = "correctness_check_exception"
        else:
            correctness_results = {
                "error": "Output data missing or invalid for comparison"
            }
            if not failure_reason:
                failure_reason = "output_data_mismatch"

        # --- Configuration Section ---
        target_conf = target_run_data.get("config", {}) if target_run_data else {}
        baseline_conf = baseline_run_data.get("config", {}) if baseline_run_data else {}

        primary_conf = target_conf if target_run_data else baseline_conf

        name_parts = base_name.split("_")
        model_name_extracted = "UnknownModel"
        if len(name_parts) >= 2:
            if name_parts[-1].startswith("subgraph"):
                model_name_extracted = "_".join(name_parts[1:-1])
            else:
                model_name_extracted = "_".join(name_parts[1:])
            if not model_name_extracted:
                model_name_extracted = name_parts[0]
        elif len(name_parts) == 1:
            model_name_extracted = name_parts[0]

        combined_config = {
            "model": model_name_extracted,
            "device": target_conf.get("device", "N/A"),
            "hardware": target_conf.get("hardware", "N/A"),
            "compiler": primary_conf.get("compiler", "N/A"),
            "warmup": primary_conf.get("warmup", "N/A"),
            "trials": primary_conf.get("trials", "N/A"),
            "compile_framework_version": primary_conf.get("framework_version", "N/A"),
            "baseline_device": baseline_conf.get("device", "N/A"),
            "baseline_hardware": baseline_conf.get("hardware", "N/A"),
            "baseline_framework_version": baseline_conf.get("framework_version", "N/A"),
        }

        # --- Calculate Speedup ---
        speedup_val = None
        baseline_mean = (
            baseline_run_data.get("results", {}).get("e2e_mean")
            if baseline_run_data
            else None
        )
        target_mean = (
            target_run_data.get("results", {}).get("e2e_mean")
            if target_run_data
            else None
        )

        if baseline_mean is not None and target_mean is not None and target_mean > 1e-9:
            try:
                speedup_val = baseline_mean / target_mean
            except ZeroDivisionError:
                print(
                    f"WARNING: {base_name}: Division by zero for speedup calculation."
                )
                speedup_val = float("inf")
        elif not failure_reason:
            print(
                f"WARNING: {base_name}: Could not calculate speedup (baseline={baseline_mean}, target={target_mean})."
            )

        # --- Performance Section ---
        performance_data = {
            "eager": {
                "e2e": {
                    "mean": baseline_run_data.get("results", {}).get("e2e_mean"),
                    "std": baseline_run_data.get("results", {}).get("e2e_std"),
                }
                if baseline_run_data
                and "e2e_mean" in baseline_run_data.get("results", {})
                else {},
            },
            "compiled": {
                "e2e": {
                    "mean": target_run_data.get("results", {}).get("e2e_mean"),
                    "std": target_run_data.get("results", {}).get("e2e_std"),
                }
                if target_run_data and "e2e_mean" in target_run_data.get("results", {})
                else {},
            },
            "datatype": {
                "eager": correctness_results.get("datatype", {}).get("eager", []),
                "compiled": correctness_results.get("datatype", {}).get("compiled", []),
            },
            "speedup": {"e2e": speedup_val} if speedup_val is not None else {},
        }

        if not performance_data["eager"].get("e2e"):
            performance_data["eager"] = {}
        if not performance_data["compiled"].get("e2e"):
            performance_data["compiled"] = {}

        if "datatype" in correctness_results:
            del correctness_results["datatype"]

        if failure_reason:
            performance_data["failure"] = failure_reason
            correctness_results = {}

        output_data = {
            "configuration": combined_config,
            "correctness": correctness_results,
            "performance": performance_data,
        }

        filename = f"{base_name}.json"
        filename = (
            "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in filename)
            + ".json"
        )
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, cls=NumpyEncoder)
            report_count += 1
        except Exception as e:
            print(
                f"ERROR: {base_name}: Error writing JSON report to {filepath}: {e}",
                file=sys.stderr,
            )

    print(f"INFO: Finished generating {report_count} JSON reports in '{output_dir}'.")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle potential numpy types from Paddle."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, paddle.Tensor):
            print(
                f"WARNING: Attempting to JSON serialize a Paddle Tensor. Converting to list.",
                file=sys.stderr,
            )
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Parse benchmark logs and output files from two directories, compare performance and correctness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-device-log-path",
        type=str,
        required=True,
        help="Path to the DIRECTORY containing log and output files from the target device.",
    )
    parser.add_argument(
        "--baseline-device-log-path",
        type=str,
        required=True,
        help="Path to the DIRECTORY containing log and output files from the baseline (e.g., CUDA) device.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the generated JSON reports.",
    )
    args = parser.parse_args()

    print(f"INFO: Scanning target device directory: {args.target_device_log_path}")
    target_pairs = find_log_output_pairs(args.target_device_log_path)
    print(f"INFO: Scanning baseline device directory: {args.baseline_device_log_path}")
    baseline_pairs = find_log_output_pairs(args.baseline_device_log_path)

    if not target_pairs and not baseline_pairs:
        print(
            "WARNING: Both log directories appear to be empty or contain no valid pairs. Exiting."
        )
        return

    print("INFO: Generating comparison JSON reports...")
    generate_comparison_json(target_pairs, baseline_pairs, args.output_dir)


if __name__ == "__main__":
    main()
