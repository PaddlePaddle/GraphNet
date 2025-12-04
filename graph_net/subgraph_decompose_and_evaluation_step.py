import os
import sys
import re
import json
import base64
import shutil
import argparse
import subprocess
import glob
from typing import List, Set, Dict, Any, Union
from graph_net.analysis_util import get_incorrect_models
from graph_net import path_utils

kMaxGraphSize = 4096


def convert_b64_string_to_json(b64str):
    return json.loads(base64.b64decode(b64str).decode("utf-8"))


def get_ranged_incorrect_models(tolerance_args: List[int], log_path: str) -> set:
    if not os.path.exists(log_path):
        return set()

    t_start = tolerance_args[0]
    models_start = set(get_incorrect_models(t_start, log_path))

    if len(tolerance_args) == 1:
        return models_start

    t_end = tolerance_args[1]
    models_end = set(get_incorrect_models(t_end, log_path))

    print(f"[Filter] Tolerance Range: {t_start} -> {t_end}")
    print(
        f"[Filter] Fail({t_start}): {len(models_start)}, Fail({t_end}): {len(models_end)}"
    )

    diff_set = models_start - models_end

    return diff_set


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
    workspace: str,
    max_subgraph_size: int,
    tasks_map: Dict[str, Union[int, str, list, dict]],
    incorrect_paths: Union[List[str], Set[str]],
    failed_decomposition_models: Union[List[str], Set[str]],
):
    """Saves the current state to a JSON file."""

    tasks_map_copy = {}
    for model_name, task_info in tasks_map.items():
        tasks_map_copy[model_name] = {}
        for key in ["original_path", "split_positions"]:
            tasks_map_copy[model_name][key] = task_info[key]

    config = {
        "max_subgraph_size": max_subgraph_size,
        "incorrect_models": list(incorrect_paths),
        "tasks_map": tasks_map_copy,
        "failed_decomposition_models": list(failed_decomposition_models),
    }
    config_path = get_decompose_config_path(workspace)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"\n[INFO] Save state to: {config_path}")


def get_model_name_with_subgraph_tag(model_path):
    fields = model_path.rstrip("/").split(os.sep)
    pattern = r"^subgraph(_\d+)?$"
    return f"{fields[-2]}_{fields[-1]}" if re.match(pattern, fields[-1]) else fields[-1]


def run_decomposer_for_single_model(
    framework: str,
    model_path: str,
    output_dir: str,
    split_positions: List[int],
    log_path: str,
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

    print(
        f"[Decomposition] model_path: {model_path}, split_positions: {split_positions}"
    )
    cmd = [
        sys.executable,
        "-m",
        f"graph_net.{framework}.run_model",
        "--model-path",
        model_path,
        "--decorator-config",
        decorator_config_b64,
    ]
    with open(log_path, "a") as f:
        result = subprocess.run(cmd, stdout=f, stderr=f, text=True)
    return result.returncode == 0


def run_decomposer_for_multi_models(
    framework, tasks_map, decomposed_samples_dir, max_subgraph_size, log_path
):
    failed_decomposition = []

    print(
        f"[Decomposition] max_subgraph_size: {max_subgraph_size}, log_path: {log_path}"
    )
    for model_name, task_info in tasks_map.items():
        original_path = task_info["original_path"]

        split_positions = task_info["split_positions"]
        if isinstance(split_positions, set):
            split_positions = sorted(list(split_positions))

        rectified_model_path = get_rectfied_model_path(original_path)
        assert os.path.exists(
            rectified_model_path
        ), f"{rectified_model_path} does not exist."

        success = run_decomposer_for_single_model(
            framework,
            rectified_model_path,
            decomposed_samples_dir,
            split_positions,
            log_path,
        )
        if not success:
            failed_decomposition.append(rectified_model_path)
    return tasks_map, failed_decomposition


def run_evaluation(
    framework: str, test_cmd_b64: str, samples_dir: str, log_path: str
) -> int:
    """Executes the test command on the batch directory."""

    test_config = convert_b64_string_to_json(test_cmd_b64)
    test_module_name = test_config["test_module_name"]
    test_module_arguments = test_config[f"{test_module_name}_arguments"]
    test_module_arguments["model-path"] = samples_dir
    if test_module_name in ["test_reference_device", "test_target_device"]:
        test_module_arguments["reference-dir"] = os.path.join(
            samples_dir, "reference_device_outputs"
        )

    cmd = [sys.executable, "-m", f"graph_net.{framework}.{test_module_name}"] + [
        item
        for key, value in test_module_arguments.items()
        for item in (f"--{key}", str(value))
    ]

    print(f"[Evaluation] Logging to: {log_path}")
    print(f"[Evaluation] command: {' '.join(cmd)}")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=f, text=True)
    assert (
        result.returncode == 0
    ), f"[ERROR] test failed for {samples_dir}, please check the log."


def reconstruct_subgraph_size(split_positions: List[int]) -> List[list]:
    """Reconstructs subgraph size based on sorted split positions."""
    deduplicated_splits = list(dict.fromkeys(split_positions))

    subgraph_size = [
        [deduplicated_splits[i], deduplicated_splits[i + 1]]
        for i in range(len(deduplicated_splits) - 1)
    ]
    return subgraph_size


def calculate_split_positions_for_subgraph(subgraph_size, max_subgraph_size):
    assert isinstance(subgraph_size, (list, tuple)) and len(subgraph_size) == 2

    # subgraph_size: the start and end position in original model.
    start_pos, end_pos = subgraph_size
    end_pos = kMaxGraphSize if end_pos == float("inf") else end_pos

    split_positions = list(range(start_pos, end_pos + 1, max_subgraph_size))
    deduplicated_splits = list(dict.fromkeys(split_positions))
    return deduplicated_splits


def generate_initial_tasks(args):
    """Generates tasks for Pass 0 based on the initial log file."""
    print(f"[Init] Pass 0: Reading from log file: {args.log_file}")
    initial_failures = get_ranged_incorrect_models(args.tolerance, args.log_file)

    tasks_map = {}
    max_subgraph_size = args.max_subgraph_size

    for model_path in initial_failures:
        model_name = get_model_name_with_subgraph_tag(model_path)

        initial_range = [0, kMaxGraphSize]
        initial_splits = calculate_split_positions_for_subgraph(
            initial_range, max_subgraph_size
        )

        tasks_map[model_name] = {
            "subgraph_path": model_path,
            "original_path": model_path,
            "split_positions": set(initial_splits),
        }

    for task in tasks_map.values():
        task["split_positions"] = sorted(list(task["split_positions"]))

    return tasks_map, max_subgraph_size


def generate_refined_tasks(base_output_dir, current_pass_id):
    """Generates tasks for Pass > 0 based on previous pass results."""
    prev_pass_dir = get_decompose_workspace_path(base_output_dir, current_pass_id - 1)
    print(f"[Init] Resuming from Pass_{current_pass_id - 1} (Dir: {prev_pass_dir})...")

    prev_config = load_decompose_config(prev_pass_dir)
    prev_incorrect_subgraphs = prev_config.get("incorrect_models", [])
    prev_tasks_map = prev_config.get("tasks_map", {})

    prev_max_subgraph_size = prev_config.get("max_subgraph_size")
    max_subgraph_size = prev_max_subgraph_size // 2

    if not prev_incorrect_subgraphs:
        return {}, max_subgraph_size

    tasks_map = {}
    for subgraph_path in prev_incorrect_subgraphs:
        # Parse model name and subgraph index
        model_name_with_subgraph_idx = subgraph_path.rstrip("/").split(os.sep)[-1]
        model_name = "_".join(model_name_with_subgraph_idx.split("_")[:-1])
        subgraph_idx = int(model_name_with_subgraph_idx.split("_")[-1])

        assert model_name in prev_tasks_map
        pre_task_for_model = prev_tasks_map[model_name]

        prev_split_positions = pre_task_for_model.get("split_positions", [])
        subgraph_ranges = reconstruct_subgraph_size(prev_split_positions)

        assert subgraph_idx < len(
            subgraph_ranges
        ), f"subgraph_idx {subgraph_idx} is out of bounds for {model_name} (previous split_positions: {prev_split_positions})"

        current_fail_range = subgraph_ranges[subgraph_idx]

        new_splits = calculate_split_positions_for_subgraph(
            current_fail_range, max_subgraph_size
        )

        if model_name not in tasks_map:
            tasks_map[model_name] = {
                "subgraph_path": subgraph_path,
                "original_path": pre_task_for_model["original_path"],
                "split_positions": set(new_splits),
            }
        else:
            tasks_map[model_name]["split_positions"].update(new_splits)

    for task in tasks_map.values():
        task["split_positions"] = sorted(list(task["split_positions"]))

    return tasks_map, max_subgraph_size


def prepare_tasks_and_verify(args, current_pass_id, base_output_dir):
    if current_pass_id == 0:
        tasks_map, max_subgraph_size = generate_initial_tasks(args)
    else:
        tasks_map, max_subgraph_size = generate_refined_tasks(
            base_output_dir, current_pass_id
        )

    print(f"[INFO] initial max_subgraph_size: {max_subgraph_size}")
    print(f"[INFO] number of incorrect models: {len(tasks_map)}")
    for model_name, task_info in tasks_map.items():
        original_path = task_info["original_path"]
        print(f"- {original_path}")

    if not tasks_map:
        print("[FINISHED] No models need processing.")
        sys.exit(0)

    if max_subgraph_size <= 0:
        print(
            f"[FINISHED] Cannot decompose with max_subgraph_size {max_subgraph_size}."
        )
        sys.exit(0)

    return tasks_map, max_subgraph_size


def execute_decomposition_phase(max_subgraph_size, tasks_map, framework, workspace):
    """Executes the decomposition phase."""

    failed_decomposition = []
    need_decompose = True if len(tasks_map) > 0 else False

    while need_decompose:
        decomposed_samples_dir = os.path.join(
            workspace, "samples" if framework == "torch" else "paddle_samples"
        )
        if not os.path.exists(decomposed_samples_dir):
            os.makedirs(decomposed_samples_dir, exist_ok=True)
            print(f"[Decomposition] decomposed_samples_dir: {decomposed_samples_dir}")

        log_path = os.path.join(
            workspace, f"log_decompose-max_subgraph_size_{max_subgraph_size}.txt"
        )
        tasks_map, failed_decomposition = run_decomposer_for_multi_models(
            framework, tasks_map, decomposed_samples_dir, max_subgraph_size, log_path
        )
        num_decomposed_samples = count_samples(decomposed_samples_dir)
        print(
            f"[Decomposition] number of graphs: {len(tasks_map)} -> {num_decomposed_samples}",
            flush=True,
        )
        if (
            not failed_decomposition
            and num_decomposed_samples == len(tasks_map)
            and max_subgraph_size > 1
        ):
            need_decompose = True
            shutil.rmtree(decomposed_samples_dir)
            os.makedirs(decomposed_samples_dir, exist_ok=True)
            max_subgraph_size = max(1, max_subgraph_size // 2)
            for model_name, task_info in tasks_map.items():
                splits = task_info["split_positions"]
                if not splits or len(splits) < 2:
                    continue
                if isinstance(splits, set):
                    splits = sorted(list(splits))
                start_pos = splits[0]
                first_segment_end = splits[1]
                new_splits = list(
                    range(start_pos, first_segment_end + 1, max_subgraph_size)
                )

                if new_splits[-1] != first_segment_end:
                    new_splits.append(first_segment_end)

                task_info["split_positions"] = sorted(list(set(new_splits)))
        else:
            need_decompose = False
        print()

    if failed_decomposition:
        print(f"[WARN] {len(failed_decomposition)} models failed to decompose.")

    return tasks_map, failed_decomposition, max_subgraph_size


def print_summary_and_suggestion(next_round_models, max_subgraph_size):
    """Print suggestion/result."""
    print("\n" + "=" * 80)
    if next_round_models and max_subgraph_size > 1:
        print(f">>> [SUGGESTION] Issues remain (Count: {len(next_round_models)}).")
        print(">>> Please start next round decomposition test (Run this script again).")
    elif next_round_models and max_subgraph_size <= 1:
        print(">>> [FAILURE] Minimal granularity reached, but errors persist.")
    else:
        print(">>> [SUCCESS] Debugging converged.")
    print("=" * 80)


def main(args):
    task_controller = TaskController(args)
    base_output_dir = task_controller.root_output_dir
    current_pass_id = task_controller.current_pass_id

    print("=" * 80)
    print(f" GraphNet Auto-Debugger | ROUND: PASS_{current_pass_id}")
    print("=" * 80)

    # --- Step 1: Prepare Tasks and Workspace ---
    tasks_map, max_subgraph_size = prepare_tasks_and_verify(
        args, current_pass_id, base_output_dir
    )
    pass_work_dir = get_decompose_workspace_path(base_output_dir, current_pass_id)
    if not os.path.exists(pass_work_dir):
        os.makedirs(pass_work_dir, exist_ok=True)

    # --- Step 2: Decomposition ---
    failed_decomposition = []
    if task_controller.task_scheduler["run_decomposer"]:
        print("\n--- Phase 1: Decomposition ---", flush=True)
        (
            tasks_map,
            failed_decomposition,
            max_subgraph_size,
        ) = execute_decomposition_phase(
            max_subgraph_size, tasks_map, args.framework, pass_work_dir
        )
    else:
        config = load_decompose_config(pass_work_dir)
        max_subgraph_size = config["max_subgraph_size"]
        failed_decomposition = config["failed_decomposition_models"]
        tasks_map = config.get("tasks_map", {})

    # --- Step 3: Evaluation ---
    pass_log_path = os.path.join(pass_work_dir, "batch_test_result.log")
    if task_controller.task_scheduler["run_evaluation"]:
        print("\n--- Phase 2: Evaluation ---")
        run_evaluation(args.framework, args.test_config, pass_work_dir, pass_log_path)

    # --- Step 4: Analysis ---
    next_round_models = set()
    if task_controller.task_scheduler["post_analysis"]:
        print("\n--- Phase 3: Analysis ---")
        analysis_tolerance = (
            args.tolerance[0] if isinstance(args.tolerance, list) else args.tolerance
        )
        next_round_models = get_incorrect_models(analysis_tolerance, pass_log_path)

        print(f"[Analysis] Found {len(next_round_models)} incorrect subgraphs.\n")
        if len(next_round_models) > 0:
            print("[DEBUG] List of detected incorrect models:")
            for idx, model_path in enumerate(sorted(list(next_round_models))):
                print(f"  [{idx}] {model_path}")
        else:
            print("[DEBUG] No incorrect models detected.")
        print_summary_and_suggestion(next_round_models, max_subgraph_size)

    # --- Step 5: Save States ---
    save_decompose_config(
        pass_work_dir,
        max_subgraph_size,
        tasks_map,
        next_round_models,
        failed_decomposition,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--framework", type=str, required=True)
    parser.add_argument(
        "--test-config", type=str, required=True, help="Base64 encoded test config"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        nargs="+",
        required=True,
        help="Tolerance level range [-10, 5)",
    )
    parser.add_argument("--max-subgraph-size", type=int, default=4096)
    args = parser.parse_args()
    print(args)
    main(args)
