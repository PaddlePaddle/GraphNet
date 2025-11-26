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
    current_min_interval: int,
    incorrect_paths: Union[List[str], Set[str]],
    active_models_map: Dict[str, str],
    used_split_positions_map: Dict[str, List[int]],
    failed_extraction_models: Union[List[str], Set[str]],
):
    """Saves the current state to a JSON file."""
    config = {
        "current_min_interval": current_min_interval,
        "incorrect_models": list(incorrect_paths),
        "active_models_map": active_models_map,
        "used_split_positions_map": used_split_positions_map,
        "failed_extraction_models": list(failed_extraction_models),
    }
    config_path = os.path.join(work_dir, "decompose_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] State saved to: {config_path}")


def run_decomposer(
    original_model_path: str,
    output_dir: str,
    split_positions: List[int],  # Receives the specific list of split points
    decorator_config: Dict[str, Any],
) -> bool:
    """Decomposes a single model using specific split positions."""

    # 1. Deep copy the template
    final_decorator_config = json.loads(json.dumps(decorator_config))

    # 2. Locate the nested dictionary to inject values
    decorator_cfg = final_decorator_config["decorator_config"]
    decorator_cfg["name"] = os.path.basename(original_model_path.rstrip("/"))

    if "custom_extractor_config" not in decorator_cfg:
        decorator_cfg["custom_extractor_config"] = {}

    custom_cfg = decorator_cfg["custom_extractor_config"]
    custom_cfg["output_dir"] = output_dir
    custom_cfg["split_positions"] = split_positions  # Use the provided list

    # 3. Encode and Run
    decorator_config_json = json.dumps(final_decorator_config)
    decorator_config_b64 = base64.b64encode(decorator_config_json.encode()).decode()

    cmd = [
        sys.executable,
        "-m",
        "graph_net.torch.run_model",
        "--model-path",
        original_model_path,
        "--decorator-config",
        decorator_config_b64,
    ]

    print(f"      [Decomposing] {os.path.basename(original_model_path)}")
    print(f"      [Strategy] Splits: {split_positions}")

    # Capture output for debugging
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"[ERROR] Decomposition failed for {original_model_path}")
        print("-" * 20 + " ERROR LOG " + "-" * 20)
        print(proc.stderr)
        print("-" * 50)
        return False
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


def reconstruct_intervals(split_positions: List[int]) -> List[tuple]:
    """Reconstructs intervals based on sorted split positions."""
    # Ensure 0 is included to generate complete interval mapping
    full_splits = sorted(list(set(split_positions + [0])))

    intervals = []
    for i in range(len(full_splits) - 1):
        intervals.append((full_splits[i], full_splits[i + 1]))

    return intervals


def main(args):
    base_output_dir = os.path.abspath(args.output_dir)
    current_pass_id = determine_current_pass_id(base_output_dir)

    # Parse the Decorator Configuration Template
    tpl_str = base64.b64decode(args.decorator_config).decode("utf-8")
    decorator_template = json.loads(tpl_str)

    print("=" * 80)
    print(f" GraphNet Auto-Debugger | ROUND: PASS_{current_pass_id}")
    print("=" * 80)

    # --- Step 1: Initialize Strategy ---
    # tasks_map: { "ModelName": { "original_path": str, "split_positions": Set[int] } }
    tasks_map = {}
    active_models_map_for_save = {}
    current_min_interval = 2048

    if current_pass_id == 0:
        print(f"[Init] Pass 0: Reading from log file: {args.log_file}")
        initial_failures = get_incorrect_models(args.tolerance, args.log_file)

        # [Strategy] Pass 0 Fixed Split: [0, 2048, 4096]
        # This will generate subgraph_0 (0-2048) and subgraph_1 (2048-4096)
        initial_splits = [0, 2048, 4096]
        current_min_interval = 2048

        for path in initial_failures:
            name = os.path.basename(path.rstrip("/"))
            active_models_map_for_save[name] = path
            tasks_map[name] = {
                "original_path": path,
                "split_positions": set(initial_splits),
            }

    else:
        prev_pass_dir = os.path.join(base_output_dir, f"pass_{current_pass_id - 1}")
        print(
            f"[Init] Resuming from Pass {current_pass_id - 1} (Dir: {prev_pass_dir})..."
        )
        prev_config = load_prev_config(current_pass_id, base_output_dir)
        prev_min_interval = prev_config.get("current_min_interval", 2048)
        prev_map = prev_config.get("active_models_map", {})
        prev_used_splits = prev_config.get("used_split_positions_map", {})
        prev_incorrect_subgraphs = prev_config.get("incorrect_models", [])

        # Reduce interval size by half for this pass
        current_min_interval = max(1, prev_min_interval // 2)

        if not prev_incorrect_subgraphs:
            print(f"[FINISHED] Debugging completed.")
            sys.exit(0)

        print(f"[Analysis] Refining splits based on failures...")

        for sub_path in prev_incorrect_subgraphs:
            parts = sub_path.rstrip("/").split("/")
            if len(parts) < 2:
                continue

            subgraph_dirname = parts[-1]
            model_name = parts[-2]

            # Retrieve original model path
            if model_name in prev_map:
                active_models_map_for_save[model_name] = prev_map[model_name]
                if model_name not in tasks_map:
                    tasks_map[model_name] = {
                        "original_path": prev_map[model_name],
                        "split_positions": set(),  # Start fresh for this round
                    }
            else:
                continue

            try:
                sub_idx = int(subgraph_dirname.split("_")[-1])
            except ValueError:
                continue

            # 1. Reconstruct previous intervals to locate the failing segment
            old_splits = sorted(prev_used_splits.get(model_name, []))
            intervals = reconstruct_intervals(old_splits)

            if sub_idx >= len(intervals):
                print(f"[WARN] Index {sub_idx} out of bounds for {model_name}")
                continue

            # 2. Get the specific failing interval [Start, End]
            fail_start, fail_end = intervals[sub_idx]

            # 3. Calculate Midpoint
            mid_point = fail_start + (fail_end - fail_start) // 2

            # 4. Update split positions: Add Start, Mid, End
            # This ensures the decomposer only focuses on [Start, Mid] and [Mid, End]
            if mid_point > fail_start and mid_point < fail_end:
                tasks_map[model_name]["split_positions"].update(
                    [int(fail_start), int(mid_point), int(fail_end)]
                )
            else:
                # Cannot split further, keep as is
                tasks_map[model_name]["split_positions"].update(
                    [int(fail_start), int(fail_end)]
                )

    if not tasks_map:
        print(f"[FINISHED] No models need processing.")
        sys.exit(0)

    print(f"[INFO] Current Min Interval: {current_min_interval}")
    print(f"[INFO] Models to Process: {len(tasks_map)}")

    # --- Step 2: Prepare Workspace ---
    pass_work_dir = os.path.join(base_output_dir, f"pass_{current_pass_id}")
    if os.path.exists(pass_work_dir):
        shutil.rmtree(pass_work_dir)
    os.makedirs(pass_work_dir, exist_ok=True)

    # --- Step 3: Decomposition ---
    print("\n--- Phase 1: Decomposition ---")
    failed_extraction = []
    final_used_splits_map = {}  # To save into config

    for model_name, task_info in tasks_map.items():
        original_path = task_info["original_path"]

        # Sort and deduplicate
        split_positions = sorted(list(task_info["split_positions"]))

        final_used_splits_map[model_name] = split_positions

        if not os.path.exists(original_path):
            continue

        model_out_dir = os.path.join(pass_work_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        success = run_decomposer(
            original_path, model_out_dir, split_positions, decorator_template
        )
        if not success:
            failed_extraction.append(model_name)

    if failed_extraction:
        print(f"\n[WARN] {len(failed_extraction)} models failed to decompose.")

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
        pass_work_dir,
        current_min_interval,
        next_round_models,
        active_models_map_for_save,
        final_used_splits_map,
        failed_extraction,
    )

    print("\n" + "=" * 80)
    if next_round_models and current_min_interval > 1:
        print(f">>> [SUGGESTION] Issues remain (Count: {len(next_round_models)}).")
        print(">>> Please start next round decomposition test (Run this script again).")
    elif next_round_models and current_min_interval <= 1:
        print(f">>> [FAILURE] Minimal granularity reached, but errors persist.")
    else:
        print(f">>> [SUCCESS] Debugging converged.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
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
