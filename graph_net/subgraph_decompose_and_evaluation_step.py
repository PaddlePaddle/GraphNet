import os
import sys
import re
import json
import base64
import argparse
import subprocess
import glob
from typing import List, Set, Dict, Any, Union
from graph_net.analysis_util import get_incorrect_models
from graph_net import path_utils

kMaxGraphSize = 4096


def convert_b64_string_to_json(b64str):
    return json.loads(base64.b64decode(b64str).decode("utf-8"))


class TaskController:
    def __init__(self, args):
        self.root_output_dir = os.path.abspath(args.output_dir)
        self.test_config = convert_b64_string_to_json(args.test_config)
        assert "test_module_name" in self.test_config

        self.test_module_name = self.test_config["test_module_name"]
        max_pass_id = self._determine_max_pass_id(self.root_output_dir)
        self.current_pass_id = (
            max_pass_id
            if self.test_module_name == "test_target_device"
            else max_pass_id + 1
        )

        self._init_task_scheduler(self.test_module_name)
        self._print()

    def _determine_max_pass_id(self, output_dir: str) -> int:
        """Scans the output directory to determine the next pass ID."""
        if not os.path.exists(output_dir):
            return -1
        existing_passes = glob.glob(os.path.join(output_dir, "pass_*"))
        valid_ids = []
        for p in existing_passes:
            basename = os.path.basename(p)
            parts = basename.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                valid_ids.append(int(parts[1]))
        return max(valid_ids) if valid_ids else -1

    def _init_task_scheduler(self, test_module_name):
        assert test_module_name in [
            "test_compiler",
            "test_reference_device",
            "test_target_device",
        ]
        if test_module_name == "test_compiler":
            self.task_scheduler = {
                "run_decomposer": True,
                "run_evaluation": True,
                "post_analysis": True,
            }
        elif test_module_name == "test_reference_device":
            self.task_scheduler = {
                "run_decomposer": True,
                "run_evaluation": True,
                "post_analysis": False,
            }
        elif test_module_name == "test_target_device":
            self.task_scheduler = {
                "run_decomposer": False,
                "run_evaluation": True,
                "post_analysis": True,
            }

    def _print(self):
        print(
            f"[TaskController] test_module_name: {self.test_module_name}, current_pass_id: {self.current_pass_id}",
            flush=True,
        )
        print(f"[TaskController] task_scheduler: {self.task_scheduler}", flush=True)
        print()


def get_rectfied_model_path(model_path):
    graphnet_root = path_utils.get_graphnet_root()
    return os.path.join(graphnet_root, model_path.split("GraphNet/")[-1])


def count_samples(samples_dir):
    num_samples = 0
    for root, dirs, files in os.walk(samples_dir):
        if path_utils.is_single_model_dir(root):
            num_samples += 1
    return num_samples


def get_decompose_config_path(output_dir: str) -> str:
    """Returns the full path to the decompose configuration file."""
    return os.path.join(output_dir, "decompose_config.json")


def get_decompose_workspace_path(output_dir, pass_id):
    return os.path.join(output_dir, f"pass_{pass_id}")


def load_decompose_config(work_dir: str) -> Dict[str, Any]:
    """Loads the configuration file from the previous pass."""
    config_path = get_decompose_config_path(work_dir)

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
    print(f"[INFO] Save state to: {config_path}")


def get_model_name_with_subgraph_tag(model_path):
    fields = model_path.rstrip("/").split(os.sep)
    pattern = r"^subgraph(_\d+)?$"
    return f"{fields[-2]}_{fields[-1]}" if re.match(pattern, fields[-1]) else fields[-1]


def run_decomposer(
    framework: str,
    model_path: str,
    output_dir: str,
    split_positions: List[int],
) -> bool:
    """Decomposes a single model."""

    graphnet_root = path_utils.get_graphnet_root()
    model_name = get_model_name_with_subgraph_tag(model_path)
    decorator_config = {
        "decorator_path": f"{graphnet_root}/graph_net/{framework}/extractor.py",
        "decorator_config": {
            "name": model_name,
            "custom_extractor_path": f"{graphnet_root}/graph_net/{framework}/naive_graph_decomposer.py",
            "custom_extractor_config": {
                "output_dir": output_dir,
                "split_positions": split_positions,
                "group_head_and_tail": False,
                "chain_style": False,
            },
        },
    }
    decorator_config_b64 = base64.b64encode(
        json.dumps(decorator_config).encode()
    ).decode()

    print(f"[Decomposing] {model_path}")
    print(f"[Strategy] split_positions: {split_positions}")

    cmd = [
        sys.executable,
        "-m",
        f"graph_net.{framework}.run_model",
        "--model-path",
        model_path,
        "--decorator-config",
        decorator_config_b64,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        print(
            f"[ERROR] Decomposition failed for {model_path}\n{result.stderr}",
            flush=True,
        )
        return False
    # print(result.stdout)
    return True


def run_evaluation(
    framework: str, test_cmd_b64: str, work_dir: str, log_path: str
) -> int:
    """Executes the test command on the batch directory."""

    test_config = convert_b64_string_to_json(test_cmd_b64)
    test_module_name = test_config["test_module_name"]
    test_module_arguments = test_config[f"{test_module_name}_arguments"]
    test_module_arguments["model-path"] = work_dir
    if test_module_name in ["test_reference_device", "test_target_device"]:
        test_module_arguments["reference-dir"] = os.path.join(
            work_dir, "reference_device_outputs"
        )

    cmd = [sys.executable, "-m", f"graph_net.{framework}.{test_module_name}"] + [
        item
        for key, value in test_module_arguments.items()
        for item in (f"--{key}", str(value))
    ]

    print(f"[Batch Testing] Logging to: {log_path}")
    print(f"[Command] {' '.join(cmd)}")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        with open(log_path, "r") as f:
            content = f.read()
            print(f"[ERROR] test failed for {work_dir}\n{content}", flush=True)
        sys.exit(proc.returncode)


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


def calculate_split_positions_for_subgraph(subgraph_size):
    assert isinstance(subgraph_size, (list, tuple)) and len(subgraph_size) == 2

    # Get the specific failing subgraph size [Start, End]
    fail_start, fail_end = subgraph_size

    # though intervals logic usually handles this via float('inf') replacement if used.
    if fail_end == float("inf"):
        fail_end = kMaxGraphSize

    # Dynamic step calculation
    subgraph_size_len = fail_end - fail_start
    new_step = subgraph_size_len // 2

    if new_step < 1:
        new_step = subgraph_size_len

    # Calculate Midpoint
    mid_point = fail_start + new_step

    # Add split positions
    if mid_point > fail_start and mid_point < fail_end:
        split_positions = [int(fail_start), int(mid_point), int(fail_end)]
    else:
        split_positions = [int(fail_start), int(fail_end)]
    return split_positions


def main(args):
    task_controller = TaskController(args)
    base_output_dir = task_controller.root_output_dir
    current_pass_id = task_controller.current_pass_id

    print("=" * 80)
    print(f" GraphNet Auto-Debugger | ROUND: PASS_{current_pass_id}")
    print("=" * 80)

    tasks_map = {}
    active_models_map_for_save = {}

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
        prev_pass_dir = get_decompose_workspace_path(
            base_output_dir, current_pass_id - 1
        )
        print(
            f"[Init] Resuming from Pass_{current_pass_id - 1} (Dir: {prev_pass_dir})..."
        )

        prev_config = load_decompose_config(prev_pass_dir)
        prev_active_models_map = prev_config.get("active_models_map", {})
        prev_used_splits = prev_config.get("split_positions_map", {})
        prev_incorrect_subgraphs = prev_config.get("incorrect_models", [])

        # Load previous max size as fallback for calculation
        prev_max_size = prev_config.get("max_subgraph_size", args.max_subgraph_size)
        max_subgraph_size = prev_max_size

        if not prev_incorrect_subgraphs:
            print("[FINISHED] Debugging completed.")
            sys.exit(0)

        print("[Analysis] Refining splits based on previous incorrect models ...")

        for subgraph_path in prev_incorrect_subgraphs:
            print(f"- subgraph_path: {subgraph_path}")
            model_name_with_subgraph_idx = subgraph_path.rstrip("/").split(os.sep)[-1]
            model_name = "_".join(model_name_with_subgraph_idx.split("_")[:-1])
            subgraph_idx = int(model_name_with_subgraph_idx.split("_")[-1])
            print(f"- model_name: {model_name}, subgraph_idx: {subgraph_idx}")

            assert model_name in prev_active_models_map
            active_models_map_for_save[model_name] = prev_active_models_map[model_name]

            # Reconstruct previous subgraph size to locate the failing segment
            prev_split_positions = sorted(prev_used_splits.get(model_name, []))
            subgraph_size = reconstruct_subgraph_size(prev_split_positions)
            assert subgraph_idx < len(
                subgraph_size
            ), f"subgraph_idx {subgraph_idx} is out of bounds for {model_name} (previous split_positions: {prev_split_positions})"

            split_positions = calculate_split_positions_for_subgraph(
                subgraph_size[subgraph_idx]
            )
            if model_name not in tasks_map:
                tasks_map[model_name] = {
                    "subgraph_path": subgraph_path,
                    "original_path": prev_active_models_map[model_name],
                    "subgraph_size": subgraph_size[subgraph_idx],
                    "split_positions": split_positions,
                }
            else:
                continue

    # Recalculate based on current map to ensure log accuracy
    real_subgraph_size = calculate_current_subgraph_size(tasks_map, max_subgraph_size)
    print(f"[INFO] Current Subgraph Size: {real_subgraph_size}")
    print(f"[INFO] Models to Process: {len(tasks_map)}")
    for model_name, task_info in tasks_map.items():
        original_path = task_info["original_path"]
        print(f"- {original_path}")

    if not tasks_map:
        print("[FINISHED] No models need processing.")
        sys.exit(0)

    # --- Step 2: Prepare Workspace ---
    pass_work_dir = get_decompose_workspace_path(base_output_dir, current_pass_id)
    if not os.path.exists(pass_work_dir):
        os.makedirs(pass_work_dir, exist_ok=True)

    # --- Step 3: Decomposition ---
    need_decompose = (
        True
        if task_controller.task_scheduler["run_decomposer"] and len(tasks_map) > 0
        else False
    )
    if need_decompose:
        print("\n--- Phase 1: Decomposition ---", flush=True)

    failed_decomposition = []
    final_used_splits_map = {}
    if need_decompose:
        decomposed_samples_dir = os.path.join(
            pass_work_dir, "samples" if args.framework == "torch" else "paddle_samples"
        )
        os.makedirs(decomposed_samples_dir, exist_ok=True)
        print(f"decomposed_samples_dir: {decomposed_samples_dir}")

        for model_name, task_info in tasks_map.items():
            original_path = task_info["original_path"]
            split_positions = sorted(list(task_info["split_positions"]))

            final_used_splits_map[model_name] = split_positions

            rectied_model_path = get_rectfied_model_path(original_path)
            print(f"original_path: {original_path}")
            print(f"rectied_model_path: {rectied_model_path}")
            assert os.path.exists(
                rectied_model_path
            ), f"{rectied_model_path} does not exist."

            success = run_decomposer(
                args.framework,
                rectied_model_path,
                decomposed_samples_dir,
                split_positions,
            )
            if not success:
                failed_decomposition.append(rectied_model_path)

        num_decomposed_samples = count_samples(decomposed_samples_dir)
        print(
            f"- number of graphs: {len(tasks_map)} -> {num_decomposed_samples}",
            flush=True,
        )
        if failed_decomposition:
            print(f"[WARN] {len(failed_decomposition)} models failed to decompose.")

    # --- Step 4: Testing ---
    if task_controller.task_scheduler["run_evaluation"]:
        print("\n--- Phase 2: Batch Testing ---")
        pass_log_path = os.path.join(pass_work_dir, "batch_test_result.log")
        run_evaluation(args.framework, args.test_config, pass_work_dir, pass_log_path)

    # --- Step 5: Analysis ---
    next_round_models = set()
    if task_controller.task_scheduler["post_analysis"]:
        print("\n--- Phase 3: Analysis ---")
        next_round_models = set(get_incorrect_models(args.tolerance, pass_log_path))
        print(f"[Result] Found {len(next_round_models)} incorrect subgraphs.")

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
        print(">>> [FAILURE] Minimal granularity reached, but errors persist.")
    else:
        print(">>> [SUCCESS] Debugging converged.")
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
        "--tolerance", type=int, required=True, help="Tolerance level range [-10, 5)"
    )
    parser.add_argument("--max-subgraph-size", type=int, default=4096)
    args = parser.parse_args()
    print(args)
    main(args)
