import argparse
import json
import os
import re
import numpy as np
from collections import defaultdict
import logging
import hashlib
import glob
import paddle  # <<< Required for correctness checks
import io       # <<< For capturing stderr
import sys      # <<< For stderr redirection
import contextlib # <<< For stderr redirection
from graph_net import test_compiler_util


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# --- Correctness Helper Functions (Copied from user request) ---
# 这些函数现在作为回调传递给 test_compiler_util

def get_cmp_equal(expected_out, compiled_out):
    def convert(x):
        if not isinstance(x, paddle.Tensor): return x # Handle potential None
        if x.dtype in [paddle.float16, paddle.bfloat16]:
            return x.astype("float32")
        elif x.dtype in [paddle.uint8, paddle.int8, paddle.int16]:
            return x.astype("int32")
        return x

    results = []
    if len(expected_out) != len(compiled_out):
        logging.error("Output list length mismatch for get_cmp_equal")
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append(str(int(a is b))) # None equals None
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append(str(int(a == b))) # Fallback for non-tensors
        elif a.shape != b.shape:
            logging.warning("Shape mismatch in get_cmp_equal")
            results.append("0") # Not equal if shapes differ
        else:
            try:
                results.append(str(int(paddle.equal_all(convert(a), convert(b)))))
            except Exception as e:
                logging.error(f"Error during paddle.equal_all: {e}")
                results.append("error")
    return " ".join(results)


def get_cmp_all_close(expected_out, compiled_out, atol, rtol):
    results = []
    if len(expected_out) != len(compiled_out):
        logging.error("Output list length mismatch for get_cmp_all_close")
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append(str(int(a is b)))
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append(str(int(a == b))) # Fallback, might not be appropriate
        elif a.shape != b.shape:
            logging.warning("Shape mismatch in get_cmp_all_close")
            results.append("0") # Not close if shapes differ
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            logging.warning(f"Non-floating point tensor encountered in get_cmp_all_close (types: {a.dtype}, {b.dtype}). Comparing with equal_all.")
            try:
                # Fallback to equality check for non-float types
                results.append(str(int(paddle.equal_all(a, b))))
            except Exception as e:
                logging.error(f"Error during fallback equal_all in allclose: {e}")
                results.append("error")
        else:
            try:
                results.append(str(int(paddle.allclose(a, b, atol=atol, rtol=rtol))))
            except Exception as e:
                logging.error(f"Error during paddle.allclose (atol={atol}, rtol={rtol}): {e}")
                results.append("error")
    return " ".join(results)


def get_format_str(f):
    try:
        f_float = float(f)
        if (abs(f_float) > 1e5 or abs(f_float) < 1e-5) and abs(f_float) != 0.0:
            return str(f"{f_float:.5E}")
        else:
            return str(f"{f_float:.5f}")
    except (ValueError, TypeError):
        logging.warning(f"Could not format value {f} as float.")
        return str(f) # Return original string if conversion fails


def get_cmp_max_diff(expected_out, compiled_out):
    results = []
    if len(expected_out) != len(compiled_out):
        logging.error("Output list length mismatch for get_cmp_max_diff")
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append("N/A")
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append("N/A")
        elif a.shape != b.shape:
            logging.warning("Shape mismatch in get_cmp_max_diff")
            results.append("shape_mismatch")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            logging.warning(f"Non-floating point tensor in get_cmp_max_diff (types: {a.dtype}, {b.dtype})")
            results.append("non_float")
        else:
            try:
                # Ensure float32 for comparison if needed
                a_f32 = a.astype("float32") if a.dtype not in [paddle.float32, paddle.float64] else a
                b_f32 = b.astype("float32") if b.dtype not in [paddle.float32, paddle.float64] else b
                results.append(get_format_str(paddle.max(paddle.abs(a_f32 - b_f32)).item()))
            except Exception as e:
                logging.error(f"Error during max_diff calculation: {e}")
                results.append("error")
    return " ".join(results)


def get_cmp_mean_diff(expected_out, compiled_out):
    results = []
    if len(expected_out) != len(compiled_out):
        logging.error("Output list length mismatch for get_cmp_mean_diff")
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append("N/A")
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append("N/A")
        elif a.shape != b.shape:
            logging.warning("Shape mismatch in get_cmp_mean_diff")
            results.append("shape_mismatch")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            logging.warning(f"Non-floating point tensor in get_cmp_mean_diff (types: {a.dtype}, {b.dtype})")
            results.append("non_float")
        else:
            try:
                a_f32 = a.astype("float32") if a.dtype not in [paddle.float32, paddle.float64] else a
                b_f32 = b.astype("float32") if b.dtype not in [paddle.float32, paddle.float64] else b
                results.append(get_format_str(paddle.mean(paddle.abs(a_f32 - b_f32)).item()))
            except Exception as e:
                logging.error(f"Error during mean_diff calculation: {e}")
                results.append("error")
    return " ".join(results)


def get_cmp_max_relative_diff(expected_out, compiled_out):
    epsilon = 1e-8
    results = []
    if len(expected_out) != len(compiled_out):
        logging.error("Output list length mismatch for get_cmp_max_relative_diff")
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append("N/A")
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append("N/A")
        elif a.shape != b.shape:
            logging.warning("Shape mismatch in get_cmp_max_relative_diff")
            results.append("shape_mismatch")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            logging.warning(f"Non-floating point tensor in get_cmp_max_relative_diff (types: {a.dtype}, {b.dtype})")
            results.append("non_float")
        else:
            try:
                a_f32 = a.astype("float32") if a.dtype not in [paddle.float32, paddle.float64] else a
                b_f32 = b.astype("float32") if b.dtype not in [paddle.float32, paddle.float64] else b
                denom = paddle.abs(a_f32) + epsilon
                relative_diff = paddle.abs(a_f32 - b_f32) / denom
                max_val = paddle.max(paddle.where(paddle.isnan(relative_diff) | paddle.isinf(relative_diff), paddle.to_tensor(0.0), relative_diff)).item()
                results.append(get_format_str(max_val) if not np.isnan(max_val) else "NaN/Inf")
            except Exception as e:
                logging.error(f"Error during max_relative_diff calculation: {e}")
                results.append("error")
    return " ".join(results)

def get_cmp_mean_relative_diff(expected_out, compiled_out):
    epsilon = 1e-8
    results = []
    if len(expected_out) != len(compiled_out):
        logging.error("Output list length mismatch for get_cmp_mean_relative_diff")
        return "length_mismatch"
    for a, b in zip(expected_out, compiled_out):
        if a is None or b is None:
            results.append("N/A")
        elif not isinstance(a, paddle.Tensor) or not isinstance(b, paddle.Tensor):
            results.append("N/A")
        elif a.shape != b.shape:
            logging.warning("Shape mismatch in get_cmp_mean_relative_diff")
            results.append("shape_mismatch")
        elif not paddle.is_floating_point(a) or not paddle.is_floating_point(b):
            logging.warning(f"Non-floating point tensor in get_cmp_mean_relative_diff (types: {a.dtype}, {b.dtype})")
            results.append("non_float")
        else:
            try:
                a_f32 = a.astype("float32") if a.dtype not in [paddle.float32, paddle.float64] else a
                b_f32 = b.astype("float32") if b.dtype not in [paddle.float32, paddle.float64] else b
                denom = paddle.abs(a_f32) + epsilon
                relative_diff = paddle.abs(a_f32 - b_f32) / denom
                valid_diff = paddle.where(paddle.isnan(relative_diff) | paddle.isinf(relative_diff), paddle.to_tensor(0.0), relative_diff)
                mean_val = paddle.mean(valid_diff).item()
                results.append(get_format_str(mean_val) if not np.isnan(mean_val) else "NaN/Inf")
            except Exception as e:
                logging.error(f"Error during mean_relative_diff calculation: {e}")
                results.append("error")
    return " ".join(results)

# --- End Correctness Helpers ---

def standardize_model_path(raw_path: str) -> str:
    """Standardizes the model path by removing common prefixes."""
    graphnet_pattern = re.compile(r".*GraphNet/(.+)")
    match = graphnet_pattern.search(raw_path)
    if match:
        return f"GraphNet/{match.group(1)}"

    logging.warning(f"Unknown model path prefix for standardization: {raw_path}. Using part after the last '/'.")
    parts = raw_path.split('/')
    if len(parts) > 1 and parts[-1].startswith("subgraph_"):
        return "/".join(parts[-2:])
    elif len(parts) > 0:
        return parts[-1]
    else:
        return raw_path

def parse_single_log_file(log_filepath: str) -> dict:
    """ Parses performance and config data from a single log file."""
    if not os.path.exists(log_filepath):
        logging.error(f"Log file not found: {log_filepath}")
        return None

    try:
        with open(log_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logging.error(f"Error reading log file {log_filepath}: {e}")
        return None

    run_data = {"config": {}, "results": {}, "original_path": ""}
    model_path_found = None
    model_run_failed = False

    patterns = {
        # Match index potentially present, then the path
        "model_path": re.compile(r"\[?\d*\]?\s*fixed_random_seed_device_runner, model_path: (.+)"),
        "config": re.compile(r"\[Config\] (\S+): (.+)"),
        "result": re.compile(r"\[Result\] (\S+): (.+)"),
        "error": re.compile(r"Run model failed:"),
    }

    for line in lines:
        line = line.strip()
        if model_run_failed: # Skip rest of file if error detected
            continue

        # Find model path first
        if not model_path_found:
            model_path_match = patterns["model_path"].search(line)
            if model_path_match:
                model_path_found = model_path_match.group(1).strip()
                run_data["original_path"] = model_path_found # Store original path
                # Don't skip, continue parsing the same line for config if needed

        if model_path_found:
            if patterns["error"].search(line):
                logging.warning(f"Detected failure for model: {model_path_found} in {log_filepath}. Skipping file.")
                model_run_failed = True
                return None # Indicate failure

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
                     # Verify model path in results matches the one found earlier
                     if value_str != model_path_found:
                         logging.warning(f"Result block model path '{value_str}' does not match header '{model_path_found}' in {log_filepath}. Using header path.")
                elif key in ["e2e_mean", "e2e_std"]:
                    try:
                        run_data["results"][key] = float(value_str)
                    except ValueError:
                        logging.warning(f"Could not parse float value '{value_str}' for key '{key}' in {log_filepath}")
                else:
                    run_data["results"][key] = value_str

    # Final check for completeness
    if not model_path_found or "e2e_mean" not in run_data["results"]:
        logging.warning(f"Incomplete data parsed from {log_filepath}. Skipping.")
        return None

    return run_data

def load_outputs(output_filepath: str) -> list:
    """Loads outputs from a JSON file and converts to Paddle Tensors."""
    if not os.path.exists(output_filepath):
        logging.warning(f"Output file not found: {output_filepath}")
        return []
    try:
        with open(output_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
             logging.error(f"Unexpected data format in {output_filepath}. Expected a list.")
             return []

        tensors = []
        for item in data:
            try:
                # Convert to tensor, defaulting to float32 as per example
                # Handle None values explicitly
                if item is None:
                    tensors.append(None)
                else:
                    # Attempt conversion, assuming float32 is a reasonable default
                    tensors.append(paddle.to_tensor(item, dtype='float32'))
            except Exception as e:
                logging.error(f"Error converting item to tensor in {output_filepath}: {item} - {e}")
                tensors.append(None) # Add placeholder on error
        return tensors
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {output_filepath}")
        return []
    except Exception as e:
        logging.error(f"Error loading or processing {output_filepath}: {e}")
        return []

def perform_correctness_checks(expected_tensors: list, compiled_tensors: list) -> dict:
    """
    Performs correctness checks using test_compiler_util and returns results.

    Args:
        expected_tensors: List of Paddle tensors from the baseline (CUDA).
        compiled_tensors: List of Paddle tensors from the new hardware.

    Returns:
        A dictionary containing the results of various correctness checks.
    """
    # The log_prompt will prefix the captured output
    MockArgs = argparse.Namespace(log_prompt="[Correctness]")

    # Capture stderr
    stderr_capture = io.StringIO()
    correctness_results = {}

    try:
        with contextlib.redirect_stderr(stderr_capture):
            # --- Type Check ---
            eager_dtypes = [str(t.dtype).replace("paddle.", "") if t is not None else "none" for t in expected_tensors]
            compiled_dtypes = [str(t.dtype).replace("paddle.", "") if t is not None else "none" for t in compiled_tensors]

            # This function *prints* the [Datatype] lines
            type_match = test_compiler_util.check_output_datatype(MockArgs, eager_dtypes, compiled_dtypes)

            # --- Float Conversion ---
            def transfer_to_float(origin_outputs):
                outputs = []
                for item in origin_outputs:
                    if (
                        item is not None
                        and isinstance(item, paddle.Tensor)
                        and item.dtype not in [paddle.float32, paddle.float64]
                        and paddle.is_floating_point(item) # Only convert floating types
                    ):
                        try:
                            item = item.astype("float32")
                        except Exception as e:
                            logging.warning(f"Could not cast tensor to float32: {e}")
                    outputs.append(item)
                return outputs

            # --- Run Checks ---
            if len(expected_tensors) != len(compiled_tensors):
                logging.warning("Output tensor list lengths differ, skipping detailed comparison.")
                correctness_results["error"] = "Output count mismatch"
                # Still log datatype info
                output_str = stderr_capture.getvalue()
            else:
                # Run checks only if lengths match
                if type_match: # As per user's example logic
                    test_compiler_util.check_equal(
                        MockArgs,
                        expected_tensors,
                        compiled_tensors,
                        cmp_equal_func=get_cmp_equal,
                    )

                    expected_out_fp32 = transfer_to_float(expected_tensors)
                    compiled_out_fp32 = transfer_to_float(compiled_tensors)

                    test_compiler_util.check_allclose(
                        MockArgs,
                        expected_out_fp32,
                        compiled_out_fp32,
                        cmp_all_close_func=get_cmp_all_close,
                        cmp_max_diff_func=get_cmp_max_diff,
                        cmp_mean_diff_func=get_cmp_mean_diff,
                        cmp_max_relative_diff_func=get_cmp_max_relative_diff,
                        cmp_mean_relative_diff_func=get_cmp_mean_relative_diff,
                    )
                else:
                     logging.warning(f"Skipping correctness checks due to type mismatch: {eager_dtypes} vs {compiled_dtypes}")
                     correctness_results["error"] = "Datatype mismatch"
                     correctness_results["eager_dtypes"] = eager_dtypes
                     correctness_results["compiled_dtypes"] = compiled_dtypes

                output_str = stderr_capture.getvalue()


        # --- Parse Captured Output ---
        correctness_pattern = re.compile(r"\[Correctness\](\[.+?\]): (.+)")
        datatype_pattern = re.compile(r"\[Datatype\]\[(\w+)\]: (.+)")

        for line in output_str.splitlines():
            corr_match = correctness_pattern.search(line)
            data_match = datatype_pattern.search(line)

            if corr_match:
                key = corr_match.group(1).strip()
                value_str = corr_match.group(2).strip()
                value_list = value_str.split()

                # Convert values to numbers if possible, matching output format
                if key.startswith("[all_close") or key == "[equal]":
                    try:
                        correctness_results[key] = [int(v) if v.isdigit() else v for v in value_list]
                    except ValueError:
                        correctness_results[key] = value_list # Keep as list of strings on error
                elif key in ["[max_diff]", "[mean_diff]", "[max_relative_diff]", "[mean_relative_diff]"]:
                    converted_list = []
                    for v in value_list:
                        try:
                           converted_list.append(float(v))
                        except (ValueError, TypeError):
                           converted_list.append(v)
                    correctness_results[key] = converted_list
                else:
                    correctness_results[key] = value_list # Default to list of strings
            
            elif data_match: # <<< MODIFIED: Removed 'and "datatype" not in correctness_results'
                if "datatype" not in correctness_results:
                    correctness_results["datatype"] = {}
                key = data_match.group(1).strip() # eager or compiled
                value_str = data_match.group(2).strip()
                correctness_results["datatype"][key] = value_str.split()


    except ImportError:
         logging.error("PaddlePaddle or graph_net.test_compiler_util not found. Cannot perform correctness checks.")
         correctness_results = {"error": "PaddlePaddle or test_compiler_util not installed"}
    except Exception as e:
         logging.error(f"Error during correctness check: {e}")
         correctness_results = {"error": f"Correctness check failed: {e}"}
         # Include any output captured before the crash
         output_str = stderr_capture.getvalue()
         if output_str:
             correctness_results["captured_stderr"] = output_str

    # --- Fill placeholders if checks were skipped/failed ---
    if not correctness_results or (len(correctness_results) == 1 and "error" in correctness_results):
        logging.warning("Populating correctness section with empty placeholders due to error or missing data.")
        placeholder_keys = [
            "[equal]", "[max_diff]", "[mean_diff]", "[max_relative_diff]", "[mean_relative_diff]"
        ]
        allclose_tolerances = [ # Re-define for placeholder generation
            (1e-10, 1e-06), (1e-10, 2.56e-04), (1e-10, 1.69e-12),
            (1e-14, 1e-14), (1e-09, 3.98e-06), (1e-09, 5.85e-04),
            (1e-09, 2.54e-11), (2.51e-13, 2.51e-13), (1e-08, 1.58e-05),
            (1e-08, 1.34e-03), (1e-08, 3.82e-10), (6.31e-12, 6.31e-12),
            (1e-07, 6.31e-05), (1e-07, 3.06e-03), (1e-07, 5.75e-09),
            (1.58e-10, 1.58e-10), (1e-06, 2.51e-04), (1e-06, 7.00e-03),
            (1e-06, 8.65e-08), (3.98e-09, 3.98e-09), (1e-05, 1.00e-03),
            (1e-05, 1.60e-02), (1e-05, 1.30e-06), (1e-07, 1.00e-07),
            (1e-04, 3.98e-03), (1e-04, 3.66e-02), (1e-04, 1.96e-05),
            (2.51e-06, 2.51e-06), (1e-03, 1.58e-02), (1e-03, 8.36e-02),
            (1e-03, 2.94e-04), (6.31e-05, 6.31e-05), (1e-02, 6.31e-02),
            (1e-02, 1.91e-01), (1e-02, 4.42e-03), (1.58e-03, 1.58e-03),
            (1e-01, 2.51e-01), (1e-01, 4.37e-01), (1e-01, 6.65e-02),
            (3.98e-02, 3.98e-02), (1.00e+00, 1.00e+00), (1.00e+01, 3.98e+00),
            (1.00e+01, 2.29e+00), (1.00e+01, 1.50e+01), (2.51e+01, 2.51e+01),
            (1.00e+02, 1.58e+01), (1.00e+02, 5.23e+00), (1.00e+02, 2.26e+02),
            (6.31e+02, 6.31e+02), (1.00e+03, 6.31e+01), (1.00e+03, 1.20e+01),
            (1.00e+03, 3.40e+03), (1.58e+04, 1.58e+04), (1.00e+04, 2.51e+02),
            (1.00e+04, 2.73e+01), (1.00e+04, 5.11e+04), (3.98e+05, 3.98e+05),
            (1.00e+05, 1.00e+03), (1.00e+05, 6.25e+01), (1.00e+05, 7.69e+05),
            (1.00e+07, 1.00e+07)
        ]
        for atol, rtol in allclose_tolerances:
            placeholder_keys.append(f"[all_close_atol_{atol:.2E}_rtol_{rtol:.2E}]")
        
        num_outputs = max(1, len(expected_tensors), len(compiled_tensors))
        if correctness_results.get("error") == "Output count mismatch":
             num_outputs = 1 # Force single placeholder if counts mismatch

        # Option 1: Fill with empty lists
        placeholder_list = []
        # Option 2: Fill with 1s (uncomment to use)
        # placeholder_list = [1] * num_outputs 

        for key in placeholder_keys:
            if key not in correctness_results:
                correctness_results[key] = placeholder_list
        
        if "datatype" not in correctness_results:
             correctness_results["datatype"] = {
                 "eager": ["N/A"] * len(expected_tensors) if expected_tensors else [],
                 "compiled": ["N/A"] * len(compiled_tensors) if compiled_tensors else []
             }


    return correctness_results


def find_log_output_pairs(log_dir: str) -> dict:
    """Finds matching .log and _outputs.json files in a directory."""
    pairs = {}
    if not os.path.isdir(log_dir):
        logging.error(f"Provided log path is not a directory: {log_dir}")
        return pairs

    log_files = glob.glob(os.path.join(log_dir, '*.log'))

    for log_file in log_files:
        base_name = os.path.basename(log_file)[:-4] # Remove .log extension
        output_file = os.path.join(log_dir, f"{base_name}_outputs.json")

        if os.path.exists(output_file):
            pairs[base_name] = {"log": log_file, "outputs": output_file}
            logging.debug(f"Found pair for '{base_name}' in {log_dir}")
        else:
            logging.warning(f"Log file '{log_file}' found but missing corresponding output file '{output_file}'.")

    logging.info(f"Found {len(pairs)} log/output pairs in {log_dir}")
    return pairs

def generate_comparison_json_new(new_hw_pairs: dict, cuda_pairs: dict, output_dir: str):
    """
    Processes pairs of logs/outputs, performs comparisons, and writes JSON reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_count = 0
    processed_keys = set()

    for base_name, new_hw_files in new_hw_pairs.items():
        if base_name in cuda_pairs:
            cuda_files = cuda_pairs[base_name]
            processed_keys.add(base_name) 

            logging.info(f"Processing comparison for: {base_name}")

            new_hw_run_data = parse_single_log_file(new_hw_files["log"])
            cuda_run_data = parse_single_log_file(cuda_files["log"])

            if not new_hw_run_data or not cuda_run_data:
                logging.warning(f"Skipping '{base_name}' due to parsing errors in log files.")
                continue

            new_hw_outputs = load_outputs(new_hw_files["outputs"])
            cuda_outputs = load_outputs(cuda_files["outputs"])

            correctness_results = perform_correctness_checks(cuda_outputs, new_hw_outputs)


            new_hw_conf = new_hw_run_data.get("config", {})
            cuda_conf = cuda_run_data.get("config", {})
            
            name_parts = base_name.split('_')
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
                "device": new_hw_conf.get("device", "N/A"),
                "hardware": new_hw_conf.get("hardware", "N/A"),
                "compiler": new_hw_conf.get("compiler", "N/A"),
                "warmup": new_hw_conf.get("warmup", "N/A"),
                "trials": new_hw_conf.get("trials", "N/A"),
                "compile_framework_version": new_hw_conf.get("framework_version", "N/A"),
                "baseline_device": cuda_conf.get("device", "N/A"),
                "baseline_hardware": cuda_conf.get("hardware", "N/A"),
                "baseline_framework_version": cuda_conf.get("framework_version", "N/A"),
            }

            speedup_val = None
            cuda_mean = cuda_run_data.get("results", {}).get("e2e_mean")
            new_hw_mean = new_hw_run_data.get("results", {}).get("e2e_mean")

            if cuda_mean is not None and new_hw_mean is not None and new_hw_mean > 1e-9:
                try:
                    speedup_val = cuda_mean / new_hw_mean
                except ZeroDivisionError:
                    logging.warning(f"Division by zero for speedup calculation in '{base_name}'")
                    speedup_val = float('inf')
            else:
                logging.warning(f"Could not calculate speedup for '{base_name}'.")

            performance_data = {
                "eager": { 
                    "e2e": {
                        "mean": cuda_run_data.get("results", {}).get("e2e_mean"),
                        "std": cuda_run_data.get("results", {}).get("e2e_std"),
                    },
                },
                "compiled": { 
                    "e2e": {
                        "mean": new_hw_run_data.get("results", {}).get("e2e_mean"),
                        "std": new_hw_run_data.get("results", {}).get("e2e_std"),
                    },
                },
                "datatype": {
                    "eager": correctness_results.get("datatype", {}).get("eager", []),
                    "compiled": correctness_results.get("datatype", {}).get("compiled", [])
                },
                "speedup": {
                    "e2e": speedup_val
                }
            }
            if "datatype" in correctness_results:
                del correctness_results["datatype"]


            output_data = {
                "configuration": combined_config,
                "correctness": correctness_results,
                "performance": performance_data,
            }

            filename = f"{base_name}.json"
            filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in filename) + ".json"
            filepath = os.path.join(output_dir, filename)

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=4, cls=NumpyEncoder)
                logging.debug(f"Successfully generated report: {filepath}")
                report_count += 1
            except Exception as e:
                logging.error(f"Error writing JSON report for '{base_name}' to {filepath}: {e}")

        else:
            logging.warning(f"No corresponding CUDA data found for new hardware run: {base_name}")

    for base_name in cuda_pairs:
        if base_name not in processed_keys:
            logging.warning(f"No corresponding new hardware data found for CUDA run: {base_name}")

    logging.info(f"Finished generating {report_count} JSON reports in '{output_dir}'.")


# Custom JSON encoder to handle potential numpy types from Paddle
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, paddle.Tensor):
             logging.warning("Attempting to JSON serialize a Paddle Tensor. Converting to list.")
             return obj.tolist() # Fallback
        return super(NumpyEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(
        description="Parse benchmark logs and output files from two directories, compare performance and correctness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Updated Arguments ---
    parser.add_argument(
        "--new-hardware-device-log-path",
        type=str,
        required=True,
        help="Path to the DIRECTORY containing log and output files from the new hardware device.",
    )
    parser.add_argument(
        "--cuda-device-log-path",
        type=str,
        required=True,
        help="Path to the DIRECTORY containing log and output files from the CUDA device (baseline).",
    )
    # --- End Updated Arguments ---
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the generated JSON reports.",
    )
    args = parser.parse_args()

    # Find pairs in both directories
    logging.info(f"Scanning new hardware directory: {args.new_hardware_device_log_path}")
    new_hw_pairs = find_log_output_pairs(args.new_hardware_device_log_path)
    logging.info(f"Scanning CUDA directory: {args.cuda_device_log_path}")
    cuda_pairs = find_log_output_pairs(args.cuda_device_log_path)

    if not new_hw_pairs or not cuda_pairs:
        logging.warning("No matching log/output pairs found in one or both directories. No reports will be generated.")
        return

    logging.info("Generating comparison JSON reports...")
    generate_comparison_json_new(new_hw_pairs, cuda_pairs, args.output_dir)

if __name__ == "__main__":
    main()

