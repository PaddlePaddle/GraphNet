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


def get_decompose_config_path(output_dir: str) -> str:
    """Returns the full path to the decompose configuration file."""
    return os.path.join(output_dir, "decompose_config.json")


def load_decompose_config(pass_id: int, output_dir: str) -> Dict[str, Any]:
    """Loads the configuration file from the previous pass."""
    prev_dir = os.path.join(output_dir, f"pass_{pass_id - 1}")
    config_path = get_decompose_config_path(prev_dir)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def save_decompose_config(
    work_dir: str,
    max_subgraph_size: int,
    incorrect_paths: Union[List[str], Set[str]],
    active_models_map: Dict[str, str],
    split_positions_map: Dict[str, List[int]],
    failed_decomposition_models: Union[List[str], Set[str]],
):
    """Saves the current state to a JSON file."""
    config = {
        "max_subgraph_size": max_subgraph_size,
        "incorrect_models": list(incorrect_paths),
        "active_models_map": active_models_map,
        "split_positions_map": split_positions_map,
        "failed_decomposition_models": list(failed_decomposition_models),
    }
    config_path = get_decompose_config_path(work_dir)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] State saved to: {config_path}")


def get_decomposer_config() -> Dict[str, Any]:
    """Constructs the decomposer configuration internally."""
    graph_net_root = os.path.dirname(graph_net.__file__)

    return {
        "decorator_path": os.path.join(graph_net_root, "torch/extractor.py"),
        "decorator_config": {
            "name": "PLACEHOLDER_NAME",
            "custom_extractor_path": os.path.join(
                graph_net_root, "torch/naive_graph_decomposer.py"
            ),
            "custom_extractor_config": {
                "output_dir": "PLACEHOLDER_DIR",
                "split_positions": [],
                "group_head_and_tail": False,
                "chain_style": False,
                "filter_path": os.path.join(
                    graph_net_root, "torch/naive_subgraph_filter.py"
                ),
                "filter_config": {},
            },
        },
    }


def run_decomposer(
    model_path: str,
    output_dir: str,
    split_positions: List[int],
    decorator_config: Dict[str, Any],
) -> bool:
    """Decomposes a single model using specific split positions."""
    final_decorator_config = json.loads(json.dumps(decorator_config))
    decorator_cfg = final_decorator_config["decorator_config"]
    decorator_cfg["name"] = os.path.basename(model_path.rstrip("/"))

    custom_cfg = decorator_cfg.get("custom_extractor_config", {})
    custom_cfg["output_dir"] = output_dir
    custom_cfg["split_positions"] = split_positions

    decorator_config_json = json.dumps(final_decorator_config)
    decorator_config_b64 = base64.b64encode(decorator_config_json.encode()).decode()

    cmd = [
        sys.executable,
        "-m",
        "graph_net.torch.run_model",
        "--model-path",
        model_path,
        "--decorator-config",
        decorator_config_b64,
    ]

    print(f"[Decomposing] {model_path}")
    print(f"[Strategy] split_positions: {split_positions}")

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"[ERROR] Decomposition failed for {model_path}")
        print("-" * 20 + " ERROR LOG " + "-" * 20)
        print(proc.stderr)
        print("-" * 50)
        return False
    return True


def run_evaluation(test_cmd_b64: str, work_dir: str, log_path: str) -> int:
    """Executes the test command on the batch directory."""
    json_str = base64.b64decode(test_cmd_b64).decode("utf-8")
    cmd_config = json.loads(json_str)

    assert "module_name" in cmd_config, "Test config must contain 'module_name'"
    assert "arguments" in cmd_config, "Test config must contain 'arguments'"

    target_module = cmd_config["module_name"]
    args_dict = cmd_config["arguments"]

    cmd = [sys.executable, "-m", target_module]

    for key, value in args_dict.items():
        if key != "model_path":
            cmd.extend([f"--{key}", str(value)])

    cmd.extend(["--model-path", work_dir])

    print(f"[Batch Testing] Logging to: {log_path}")
    print(f"[Command] {' '.join(cmd)}")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def reconstruct_subgraph_size(split_positions: List[int]) -> List[tuple]:
    """Reconstructs subgraph size based on sorted split positions."""
    full_splits = sorted(list(set(split_positions)))

    subgraph_size = []
    # Needs at least 2 points to form an subgraph size
    if len(full_splits) < 2:
        return []

    for i in range(len(full_splits) - 1):
        subgraph_size.append((full_splits[i], full_splits[i + 1]))

    return subgraph_size


def calculate_current_subgraph_size(
    tasks_map: Dict[str, Dict], fallback_size: int
) -> int:
    """Calculates the current subgraph size from generated tasks."""
    current_subgraph_size = float("inf")
    found_splits = False

    for _, info in tasks_map.items():
        splits = sorted(list(info["split_positions"]))

        if len(splits) < 2:
            continue

        found_splits = True
        for i in range(len(splits) - 1):
            diff = splits[i + 1] - splits[i]
            if diff > 0:
                current_subgraph_size = min(current_subgraph_size, diff)

    return (
        int(current_subgraph_size)
        if found_splits and current_subgraph_size != float("inf")
        else fallback_size
    )


def main(args):
    base_output_dir = os.path.abspath(args.output_dir)
    current_pass_id = determine_current_pass_id(base_output_dir)

    decorator_template = get_decomposer_config()

    print("=" * 80)
    print(f" GraphNet Auto-Debugger | ROUND: PASS_{current_pass_id}")
    print("=" * 80)

    tasks_map = {}
    active_models_map_for_save = {}
    kMaxGraphSize = 4096

    # Initialize using the argument passed from bash
    max_subgraph_size = args.max_subgraph_size

    if current_pass_id == 0:
        print(f"[Init] Pass 0: Reading from log file: {args.log_file}")
        initial_failures = get_incorrect_models(args.tolerance, args.log_file)

        # Dynamic generation based on step size (args.max_subgraph_size)
        initial_splits = list(range(0, kMaxGraphSize + 1, max_subgraph_size))

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

        prev_config = load_decompose_config(current_pass_id, base_output_dir)
        prev_map = prev_config.get("active_models_map", {})

        prev_used_splits = prev_config.get("split_positions_map", {})
        prev_incorrect_subgraphs = prev_config.get("incorrect_models", [])

        # Load previous max size as fallback for calculation
        prev_max_size = prev_config.get("max_subgraph_size", args.max_subgraph_size)
        max_subgraph_size = prev_max_size

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

            if model_name in prev_map:
                active_models_map_for_save[model_name] = prev_map[model_name]
                if model_name not in tasks_map:
                    tasks_map[model_name] = {
                        "original_path": prev_map[model_name],
                        "split_positions": set(),
                    }
            else:
                continue

            try:
                sub_idx = int(subgraph_dirname.split("_")[-1])
            except ValueError:
                continue

            # 1. Reconstruct previous subgraph size to locate the failing segment
            old_split_position = sorted(prev_used_splits.get(model_name, []))
            subgraph_size = reconstruct_subgraph_size(old_split_position)

            if sub_idx >= len(subgraph_size):
                print(
                    f"[WARN] Index {sub_idx} out of bounds for {model_name} (old split position: {old_split_position})"
                )
                continue

            # 2. Get the specific failing subgraph size [Start, End]
            fail_start, fail_end = subgraph_size[sub_idx]

            # though intervals logic usually handles this via float('inf') replacement if used.
            if fail_end == float("inf"):
                fail_end = kMaxGraphSize

            # Dynamic step calculation
            subgraph_size_len = fail_end - fail_start
            new_step = subgraph_size_len // 2

            if new_step < 1:
                new_step = subgraph_size_len

            # 3. Calculate Midpoint
            mid_point = fail_start + new_step

            # 4. Add split positions
            if mid_point > fail_start and mid_point < fail_end:
                tasks_map[model_name]["split_positions"].update(
                    [int(fail_start), int(mid_point), int(fail_end)]
                )
            else:
                tasks_map[model_name]["split_positions"].update(
                    [int(fail_start), int(fail_end)]
                )

    if not tasks_map:
        print(f"[FINISHED] No models need processing.")
        sys.exit(0)

    # Recalculate based on current map to ensure log accuracy
    real_subgraph_size = calculate_current_subgraph_size(tasks_map, max_subgraph_size)
    print(f"[INFO] Current Subgraph Size: {real_subgraph_size}")
    print(f"[INFO] Models to Process: {len(tasks_map)}")

    # --- Step 2: Prepare Workspace ---
    pass_work_dir = os.path.join(base_output_dir, f"pass_{current_pass_id}")
    if os.path.exists(pass_work_dir):
        shutil.rmtree(pass_work_dir)
    os.makedirs(pass_work_dir, exist_ok=True)

    # --- Step 3: Decomposition ---
    print("\n--- Phase 1: Decomposition ---")
    failed_decomposition = []
    final_used_splits_map = {}

    for model_name, task_info in tasks_map.items():
        original_path = task_info["original_path"]
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
            failed_decomposition.append(model_name)

    if failed_decomposition:
        print(f"\n[WARN] {len(failed_decomposition)} models failed to decompose.")

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
    save_decompose_config(
        pass_work_dir,
        real_subgraph_size,
        next_round_models,
        active_models_map_for_save,
        final_used_splits_map,
        failed_decomposition,
    )

    print("\n" + "=" * 80)
    if next_round_models and real_subgraph_size > 1:
        print(f">>> [SUGGESTION] Issues remain (Count: {len(next_round_models)}).")
        print(">>> Please start next round decomposition test (Run this script again).")
    elif next_round_models and real_subgraph_size <= 1:
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
        "--tolerance", type=float, required=True, help="Tolerance level range [-10, 5)"
    )
    parser.add_argument("--max-subgraph-size", type=int, default=2048)

    args = parser.parse_args()
    main(args)
