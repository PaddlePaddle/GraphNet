import os
import sys
import re
import json
import base64
import shutil
import argparse
import subprocess
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Union
from graph_net.analysis_util import get_incorrect_models
from graph_net import path_utils

kMaxGraphSize = 4096


def convert_b64_string_to_json(b64str):
    return json.loads(base64.b64decode(b64str).decode("utf-8"))


def convert_json_to_b64_string(config):
    return base64.b64encode(json.dumps(config).encode()).decode()


def get_pass_name(pass_id):
    return f"pass_{pass_id}"


def get_ranged_incorrect_models(tolerance_args: List[int], log_path: str) -> set:
    assert os.path.exists(log_path)

    t_start = tolerance_args[0]
    models_start = set(get_incorrect_models(t_start, log_path))

    if len(tolerance_args) == 1:
        return models_start

    t_end = tolerance_args[1]
    models_end = set(get_incorrect_models(t_end, log_path))

    print(
        f"[Init] number of incorrect models: {len(models_start)} (tolerance={t_start}) - {len(models_end)} (tolerance={t_end})"
    )
    return models_start - models_end


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


@dataclass
class DecomposeConfig:
    method: str
    tolerance: int | List[int]
    max_subgraph_size: int = -1
    tasks_map: Dict[str, Union[int, str, list, dict]] = field(default_factory=dict)
    running_states: Dict[str, Union[int, str, list, dict]] = field(default_factory=dict)

    def save(self, work_dir):
        """Save the current config to a JSON file."""
        config_path = self.get_config_path(work_dir)

        with open(config_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"\n[INFO] Save state to: {config_path}")

    @classmethod
    def load(self, work_dir):
        """Initialize a object from a JSON file."""
        config_path = self.get_config_path(work_dir)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing configuration file: {config_path}")

        with open(config_path, "r") as f:
            data = json.load(f)
        return self(**data)

    @classmethod
    def get_config_path(self, work_dir) -> str:
        return os.path.join(work_dir, "decompose_config.json")

    def get_incorrect_models(self, pass_id):
        pass_key = get_pass_name(pass_id)
        assert pass_key in self.running_states
        return self.running_states[pass_key]["incorrect_models"]

    def update_running_states(self, pass_id, **kwargs):
        pass_key = get_pass_name(pass_id)
        if self.running_states.get(pass_key, None) is None:
            self.running_states[pass_key] = {}

        for key, value in kwargs.items():
            assert key in [
                "num_incorrect_models",
                "incorrect_models",
                "failed_decomposition_models",
            ]
            self.running_states[pass_key][key] = value


def get_rectfied_model_path(model_path):
    graphnet_root = path_utils.get_graphnet_root()
    return os.path.join(graphnet_root, model_path.split("GraphNet/")[-1])


def count_samples(samples_dir):
    num_samples = 0
    for root, dirs, files in os.walk(samples_dir):
        if path_utils.is_single_model_dir(root):
            num_samples += 1
    return num_samples


def get_decompose_workspace_path(output_dir, pass_id):
    return os.path.join(output_dir, f"pass_{pass_id}")


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
            "custom_extractor_path": f"{graphnet_root}/graph_net/{framework}/graph_decomposer.py",
            "custom_extractor_config": {
                "output_dir": output_dir,
                "split_positions": split_positions,
                "group_head_and_tail": False,
                "chain_style": False,
            },
        },
    }
    decorator_config_b64 = convert_json_to_b64_string(decorator_config)

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
        split_positions = sorted(list(task_info["split_positions"]))

        method = "fixed-start"
        if method == "fixed-start":
            assert len(split_positions) >= 3, f"{split_positions=}"
            split_positions = [0, split_positions[1]]

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


def reconstruct_split_positions_for_subgraphs(
    split_positions, subgraph_idxs, max_subgraph_size
):
    subgraph_idxs = [subgraph_idxs] if isinstance(subgraph_idxs, int) else subgraph_idxs

    new_split_positions = []
    for subgraph_idx in subgraph_idxs:
        assert (
            subgraph_idx < len(split_positions) - 1
        ), f"subgraph_idx {subgraph_idx} is out of bounds of split_positions: {split_positions}."

        start_pos, end_pos = split_positions[subgraph_idx : subgraph_idx + 2]
        new_split_positions = new_split_positions + list(
            range(start_pos, end_pos + max_subgraph_size, max_subgraph_size)
        )
    return sorted(list(set(new_split_positions)))


def generate_initial_tasks(args):
    """Generates tasks for Pass 0 based on the initial log file."""
    print(f"[Init] Pass 0: Reading from log file: {args.log_file}")
    initial_failures = get_ranged_incorrect_models(args.tolerance, args.log_file)

    tasks_map = {}
    max_subgraph_size = min(args.max_subgraph_size, kMaxGraphSize // 2)

    initial_split_positions = reconstruct_split_positions_for_subgraphs(
        [0, kMaxGraphSize], 0, max_subgraph_size
    )
    for model_path in initial_failures:
        model_name = get_model_name_with_subgraph_tag(model_path)
        tasks_map[model_name] = {
            "original_path": model_path,
            "split_positions": initial_split_positions,
        }

    running_states = {
        "pass_0": {
            "num_incorrect_models": len(initial_failures),
            "incorrect_models": list(sorted(initial_failures)),
        }
    }
    return tasks_map, max_subgraph_size, running_states


def extract_model_name_and_subgraph_idx(subgraph_path):
    # Parse model name and subgraph index
    model_name_with_subgraph_idx = subgraph_path.rstrip("/").split(os.sep)[-1]
    model_name = "_".join(model_name_with_subgraph_idx.split("_")[:-1])
    subgraph_idx = int(model_name_with_subgraph_idx.split("_")[-1])
    return model_name, subgraph_idx


def collect_incorrect_subgraph_idxs(args, target_model_names, incorrect_models):
    model_name2subgraph_idxs = {}
    for subgraph_path in sorted(incorrect_models):
        model_name, subgraph_idx = extract_model_name_and_subgraph_idx(subgraph_path)
        print(f"{subgraph_path=}")
        print(f"{model_name=}, {subgraph_idx=}")
        assert model_name in target_model_names, f"{model_name=}, {subgraph_idx=}"

        if model_name not in model_name2subgraph_idxs:
            model_name2subgraph_idxs[model_name] = []
        model_name2subgraph_idxs[model_name].append(subgraph_idx)

    if args.method == "fixed-start":
        print(model_name2subgraph_idxs)
        for model_name in target_model_names:
            if model_name not in model_name2subgraph_idxs:
                model_name2subgraph_idxs[model_name] = [1]
            else:
                assert len(
                    model_name2subgraph_idxs[model_name]
                ) == 1 and model_name2subgraph_idxs[model_name] == [0]
    return model_name2subgraph_idxs


def generate_successor_tasks(args, base_output_dir, current_pass_id):
    """Generates tasks for Pass > 0 based on previous pass results."""
    prev_pass_dir = get_decompose_workspace_path(base_output_dir, current_pass_id - 1)
    print(f"[Init] Resuming from Pass_{current_pass_id - 1} (Dir: {prev_pass_dir})...")

    prev_config = DecomposeConfig.load(prev_pass_dir)
    max_subgraph_size = prev_config.max_subgraph_size // 2
    incorrect_models = prev_config.get_incorrect_models(current_pass_id)
    if args.method != "fixed-start" and not incorrect_models:
        return {}, max_subgraph_size, prev_config.running_states

    tasks_map = {}
    prev_tasks_map = prev_config.tasks_map

    target_model_names = list(prev_tasks_map.keys())
    model_name2subgraph_idxs = collect_incorrect_subgraph_idxs(
        args, target_model_names, incorrect_models
    )

    for model_name, subgraph_idxs in model_name2subgraph_idxs.items():
        pre_task_for_model = prev_tasks_map[model_name]

        prev_split_positions = pre_task_for_model.get("split_positions", [])
        split_positions = reconstruct_split_positions_for_subgraphs(
            prev_split_positions, subgraph_idxs, max_subgraph_size
        )

        tasks_map[model_name] = {
            "original_path": pre_task_for_model["original_path"],
            "split_positions": split_positions,
        }
        print(f"{tasks_map=}")

    return tasks_map, max_subgraph_size, prev_config.running_states


def prepare_tasks_and_verify(args, current_pass_id, base_output_dir):
    if current_pass_id == 0:
        tasks_map, max_subgraph_size, running_states = generate_initial_tasks(args)
    else:
        tasks_map, max_subgraph_size, running_states = generate_successor_tasks(
            args, base_output_dir, current_pass_id
        )

    print(f"[Init] initial max_subgraph_size: {max_subgraph_size}")
    print(f"[Init] number of incorrect models: {len(tasks_map)}")
    for idx, (model_name, task_info) in enumerate(tasks_map.items()):
        original_path = task_info["original_path"]
        print(f"- [{idx}] {original_path}")

    if not tasks_map:
        print("[FINISHED] No models need processing.")
        sys.exit(0)

    if max_subgraph_size <= 0:
        print(
            f"[FINISHED] Cannot decompose with max_subgraph_size {max_subgraph_size}."
        )
        sys.exit(0)

    return tasks_map, max_subgraph_size, running_states


def execute_decomposition_phase(max_subgraph_size, tasks_map, framework, workspace):
    """Executes the decomposition phase."""

    failed_decomposition = []
    need_decompose = True if len(tasks_map) > 0 else False
    method = "fixed-start"

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
            and method != "fixed-start"
        ):
            need_decompose = True
            shutil.rmtree(decomposed_samples_dir)
            os.makedirs(decomposed_samples_dir, exist_ok=True)
            max_subgraph_size = max(1, max_subgraph_size // 2)
            for model_name, task_info in tasks_map.items():
                split_positions = task_info["split_positions"]
                if not split_positions or len(split_positions) < 2:
                    continue
                new_split_positions = reconstruct_split_positions_for_subgraphs(
                    split_positions, 0, max_subgraph_size
                )
                task_info["split_positions"] = new_split_positions
        else:
            need_decompose = False
        print()

    if failed_decomposition:
        print(f"[WARN] {len(failed_decomposition)} models failed to decompose.")

    return tasks_map, failed_decomposition, max_subgraph_size


def count_unique_original_models(incorrect_models):
    original_model_paths = set(
        model_name
        for subgraph_path in incorrect_models
        for model_name, _ in [extract_model_name_and_subgraph_idx(subgraph_path)]
    )
    return len(original_model_paths)


def print_summary_and_suggestion(args, next_round_models, max_subgraph_size):
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
    tasks_map, max_subgraph_size, running_states = prepare_tasks_and_verify(
        args, current_pass_id, base_output_dir
    )
    decompose_config = DecomposeConfig(
        method=args.method,
        tolerance=args.tolerance,
        max_subgraph_size=max_subgraph_size,
        tasks_map=tasks_map,
        running_states=running_states,
    )
    work_dir = get_decompose_workspace_path(base_output_dir, current_pass_id)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # --- Step 2: Decomposition ---
    if task_controller.task_scheduler["run_decomposer"]:
        print("\n--- Phase 1: Decomposition ---", flush=True)
        (
            tasks_map,
            failed_decomposition,
            max_subgraph_size,
        ) = execute_decomposition_phase(
            max_subgraph_size, tasks_map, args.framework, work_dir
        )
        decompose_config.update_running_states(
            current_pass_id, failed_decomposition_models=list(failed_decomposition)
        )
    else:
        print("\n--- Phase 1: Decomposition (skipped) ---", flush=True)
        decompose_config = DecomposeConfig.load(work_dir)

    # --- Step 3: Evaluation ---
    log_path = os.path.join(work_dir, f"log_{task_controller.test_module_name}.txt")
    if task_controller.task_scheduler["run_evaluation"]:
        print(f"\n--- Phase 2: Evaluation ({task_controller.test_module_name}) ---")
        run_evaluation(args.framework, args.test_config, work_dir, log_path)

    # --- Step 4: Analysis ---
    if task_controller.task_scheduler["post_analysis"]:
        tolerance = (
            args.tolerance[0] if isinstance(args.tolerance, list) else args.tolerance
        )
        print(f"\n--- Phase 3: Analysis (torlance={tolerance}) ---")
        next_pass_incorrect_models = sorted(get_incorrect_models(tolerance, log_path))
        num_original_models = count_unique_original_models(next_pass_incorrect_models)
        decompose_config.update_running_states(
            current_pass_id + 1,
            num_incorrect_models=num_original_models,
            incorrect_models=list(next_pass_incorrect_models),
        )

        print(
            f"[Analysis] Found {len(next_pass_incorrect_models)} incorrect subgraphs ({num_original_models} original models)."
        )
        for idx, model_path in enumerate(next_pass_incorrect_models):
            print(f"- [{idx}] {model_path}")
        print_summary_and_suggestion(
            args, next_pass_incorrect_models, max_subgraph_size
        )

    # --- Step 5: Save States ---
    decompose_config.save(work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--framework", type=str, required=True)
    parser.add_argument(
        "--test-config", type=str, required=True, help="Base64 encoded test config"
    )
    parser.add_argument("--method", type=str, required=True)
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
