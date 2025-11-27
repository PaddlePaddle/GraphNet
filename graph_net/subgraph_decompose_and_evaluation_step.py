import os
import sys
import json
import base64
import shutil
import argparse
import subprocess
import glob
from typing import List, Set, Dict, Any, Union
import graph_net
from graph_net.analysis_util import get_incorrect_models
from graph_net import path_utils


def get_rectfied_model_path(model_path):
    graphnet_root = path_utils.get_graphnet_root()
    return os.path.join(graphnet_root, model_path.split("GraphNet/")[-1])


def count_samples(samples_dir):
    num_samples = 0
    for root, dirs, files in os.walk(samples_dir):
        if path_utils.is_single_model_dir(root):
            num_samples += 1
    return num_samples


def determine_current_pass_id(output_dir: str) -> int:
    """Scans the output directory to determine the next pass ID."""
    if not os.path.exists(output_dir):
        return 0
    existing_passes = glob.glob(os.path.join(output_dir, "pass_*"))
    valid_ids = []
    for p in existing_passes:
        basename = os.path.basename(p)
        parts = basename.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            valid_ids.append(int(parts[1]))
    return max(valid_ids) + 1 if valid_ids else 0


def load_prev_config(pass_id: int, output_dir: str) -> Dict[str, Any]:
    """Loads the configuration file from the previous pass."""
    prev_dir = os.path.join(output_dir, f"pass_{pass_id - 1}")
    config_path = os.path.join(prev_dir, "decompose_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def save_current_config(
    work_dir: str,
    current_max_size: int,
    incorrect_models: Union[List[str], Set[str]],
    failed_models: List[str],
):
    """Saves the current state."""
    config = {
        "current_max_subgraph_size": current_max_size,
        "incorrect_models": list(incorrect_models),
        "failed_extraction_models": list(failed_models),
    }
    config_path = os.path.join(work_dir, "decompose_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] State saved to: {config_path}")


def run_decomposer(
    framework: str,
    model_path: str,
    output_dir: str,
    max_subgraph_size: int,
    decorator_config: Dict[str, Any],
) -> bool:
    """Decomposes a single model."""
    # 1. Calculate dynamic split positions
    upper_bound = 4096
    split_positions = list(range(max_subgraph_size, upper_bound, max_subgraph_size))

    # 2. Deep copy the template
    final_decorator_config = json.loads(json.dumps(decorator_config))

    # 3. Locate the nested dictionary to inject values
    decorator_cfg = final_decorator_config["decorator_config"]
    decorator_cfg["name"] = os.path.basename(model_path.rstrip("/"))

    if "custom_extractor_config" not in decorator_cfg:
        decorator_cfg["custom_extractor_config"] = {}

    custom_cfg = decorator_cfg["custom_extractor_config"]
    custom_cfg["output_dir"] = output_dir
    custom_cfg["split_positions"] = split_positions

    # 4. Encode and Run
    decorator_config_json = json.dumps(final_decorator_config)
    decorator_config_b64 = base64.b64encode(decorator_config_json.encode()).decode()

    cmd = [
        sys.executable,
        "-m",
        f"graph_net.{framework}.run_model",
        "--model-path",
        model_path,
        "--decorator-config",
        decorator_config_b64,
    ]

    print(
        f"- [Decomposing] {os.path.basename(model_path)} (max_subgraph_size={max_subgraph_size}, split_positions={split_positions})"
    )

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] Decomposition failed for {model_path}\n{result.stderr}")
        return False
    print(result.stdout)
    return True


def run_evaluation(test_cmd_b64: str, work_dir: str, log_path: str) -> int:
    """Executes the test command on the batch directory."""
    json_str = base64.b64decode(test_cmd_b64).decode("utf-8")
    cmd_config = json.loads(json_str)

    # Check if the config follows the new structure: {"module_name": "...", "arguments": {...}}
    if "module_name" in cmd_config and "arguments" in cmd_config:
        target_module = cmd_config["module_name"]
        args_dict = cmd_config["arguments"]
    else:
        # Fallback for old format (flat dictionary), assuming default compiler test
        target_module = "graph_net.torch.test_compiler"
        args_dict = cmd_config

    cmd = [sys.executable, "-m", target_module]

    for key, value in args_dict.items():
        if key == "model_path":
            continue
        cmd.append(f"--{key}")
        cmd.append(str(value))

    cmd.append("--model-path")
    cmd.append(work_dir)

    print(f"      [Batch Testing] Logging to: {log_path}")
    print(f"      [Command] {' '.join(cmd)}")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


# ==========================================================
# Main Execution Flow
# ==========================================================
def main(args):
    base_output_dir = os.path.abspath(args.output_dir)
    current_pass_id = determine_current_pass_id(base_output_dir)

    # Parse the Decorator Configuration Template
    tpl_str = base64.b64decode(args.decorator_config).decode("utf-8")
    decorator_template = json.loads(tpl_str)

    print("=" * 80)
    print(f" GraphNet Auto-Debugger | ROUND: PASS_{current_pass_id}")
    print("=" * 80)

    # --- Step 1: Initialize State ---
    target_models = []
    current_max_size = 2048

    if current_pass_id == 0:
        print(f"[Init] Pass 0: Reading from log file: {args.log_file}")
        current_max_size = args.max_subgraph_size
        target_models = get_incorrect_models(args.tolerance, args.log_file)
    else:
        print(f"[Init] Resuming from Pass {current_pass_id - 1}...")
        prev_config = load_prev_config(current_pass_id, base_output_dir)
        target_models = prev_config.get("incorrect_models", [])

        prev_size = prev_config.get("current_max_subgraph_size", 2048)
        current_max_size = max(1, prev_size // 2)

        if not target_models:
            print(f"[FINISHED] Debugging completed.")
            sys.exit(0)

    print(f"[INFO] current max_subgraph_size: {current_max_size}")
    print(f"[INFO] number of incorrect models: {len(target_models)}")
    for model_path in target_models:
        print(f"- {model_path}")

    # --- Step 2: Prepare Workspace ---
    pass_work_dir = os.path.join(base_output_dir, f"pass_{current_pass_id}")
    if os.path.exists(pass_work_dir):
        shutil.rmtree(pass_work_dir)
    os.makedirs(pass_work_dir, exist_ok=True)

    # --- Step 3: Decomposition ---
    print("\n--- Phase 1: Decomposition ---", flush=True)
    need_decompose = True
    while need_decompose:
        failed_extraction = []

        for idx, model_path in enumerate(target_models):
            rectied_model_path = get_rectfied_model_path(model_path)
            assert os.path.exists(
                rectied_model_path
            ), f"{rectied_model_path} does not exist."

            model_name = os.path.basename(rectied_model_path.rstrip("/"))
            model_out_dir = os.path.join(pass_work_dir, model_name)
            os.makedirs(model_out_dir, exist_ok=True)

            success = run_decomposer(
                args.framework,
                rectied_model_path,
                model_out_dir,
                current_max_size,
                decorator_template,
            )
            if not success:
                failed_extraction.append(rectied_model_path)
        num_decomposed_samples = count_samples(pass_work_dir)
        print(
            f"- number of graphs: {len(target_models)} -> {num_decomposed_samples}",
            flush=True,
        )
        if failed_extraction:
            print(f"[WARN] {len(failed_extraction)} models failed to decompose.")

        if num_decomposed_samples == len(target_models):
            need_decompose = True
            shutil.rmtree(pass_work_dir)
            os.makedirs(pass_work_dir, exist_ok=True)
            shutil.rmtree("/tmp")
            os.makedirs("/tmp", exist_ok=True)
            current_max_size = max(1, current_max_size // 2)
        else:
            need_decompose = False

    # --- Step 4: Testing ---
    print("\n--- Phase 2: Batch Testing ---")
    pass_log_path = os.path.join(pass_work_dir, "batch_test_result.log")
    run_evaluation(args.test_config, pass_work_dir, pass_log_path)

    # --- Step 5: Analysis ---
    print("\n--- Phase 3: Analysis ---")
    next_round_models = set()
    try:
        next_round_models = set(get_incorrect_models(args.tolerance, pass_log_path))
        print(f"      [Result] Found {len(next_round_models)} incorrect subgraphs.")
    except Exception as e:
        print(f"      [ERROR] Log analysis failed: {e}")

    # --- Step 6: Save State ---
    save_current_config(
        pass_work_dir, current_max_size, next_round_models, failed_extraction
    )

    print("\n" + "=" * 80)
    if next_round_models and current_max_size > 1:
        print(f">>> [SUGGESTION] Issues remain (Count: {len(next_round_models)}).")
        print(">>> Please start next round decomposition test (Run this script again).")
    elif next_round_models and current_max_size <= 1:
        print(f">>> [FAILURE] Minimal granularity reached, but errors persist.")
    else:
        print(f">>> [SUCCESS] Debugging converged.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--framework", type=str, required=True)
    parser.add_argument(
        "--test-config", type=str, required=True, help="Base64 encoded test config"
    )
    parser.add_argument(
        "--decorator-config",
        type=str,
        required=True,
        help="Base64 encoded decorator config template",
    )
    parser.add_argument(
        "--tolerance", type=float, required=True, help="Tolerance level range [-10, 5)"
    )
    parser.add_argument("--max-subgraph-size", type=int, default=4096)

    args = parser.parse_args()
    main(args)
