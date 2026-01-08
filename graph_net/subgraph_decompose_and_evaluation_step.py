import os
import sys
import re
import json
import base64
import shutil
import argparse
import subprocess
import glob
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from graph_net.analysis_util import get_incorrect_models
from graph_net import path_utils

MAX_GRAPH_SIZE = 4096


def convert_b64_string_to_json(b64str):
    return json.loads(base64.b64decode(b64str).decode("utf-8"))


def convert_json_to_b64_string(config):
    return base64.b64encode(json.dumps(config).encode()).decode()


def get_pass_name(pass_id):
    return f"pass_{pass_id}"


def get_decompose_workspace_path(output_dir, pass_id):
    return os.path.join(output_dir, get_pass_name(pass_id))


def get_ranged_incorrect_models(tolerance_args: List[int], log_path: str) -> set:
    assert os.path.exists(log_path)

    t_start = tolerance_args[0]
    models_start = set(get_incorrect_models(t_start, log_path))

    if len(tolerance_args) == 1:
        return models_start

    t_end = tolerance_args[1]
    models_end = set(get_incorrect_models(t_end, log_path))

    print(
        f"[Init] number of incorrect models: {len(models_start)} (tolerance={t_start}) - {len(models_end)} (tolerance={t_end})",
        flush=True,
    )
    return models_start - models_end


def extract_model_name_and_subgraph_idx(subgraph_path):
    # Parse model name and subgraph index
    model_name_with_subgraph_idx = subgraph_path.rstrip("/").split(os.sep)[-1]
    model_name = "_".join(model_name_with_subgraph_idx.split("_")[:-1])
    subgraph_idx = int(model_name_with_subgraph_idx.split("_")[-1])
    return model_name, subgraph_idx


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
class ModelRecord:
    original_path: str
    uniform_split_positions: List[int] = field(default_factory=list)
    subgraph_paths: List[str] = field(default_factory=list)
    incorrect_subgraph_idxs: List[int] = None

    def get_split_positions(self, decompose_method):
        if decompose_method == "fixed-start":
            assert (
                len(self.uniform_split_positions) >= 2
            ), f"{self.uniform_split_positions=}"
            return [0, self.uniform_split_positions[1]]
        return self.uniform_split_positions

    def update_for_next_decompose(self, subgraph_idx, max_subgraph_size):
        self.uniform_split_positions = reconstruct_split_positions_for_subgraphs(
            self.uniform_split_positions, subgraph_idx, max_subgraph_size
        )


@dataclass
class RunningState:
    incorrect_models_from_log: List[str] = None
    model_name2record: Dict[str, ModelRecord] = field(default_factory=dict)

    @classmethod
    def init_from_dict(cls, incorrect_models_from_log, model_name2record):
        converted_model_name2record = {
            model_name: record
            if isinstance(record, ModelRecord)
            else ModelRecord(**record)
            for model_name, record in model_name2record.items()
        }
        return cls(incorrect_models_from_log, converted_model_name2record)

    def get_incorrect_models(self, decompose_method):
        if decompose_method != "fixed-start":
            return self.incorrect_models_from_log

        incorrect_models = []
        for model_name, model_record in self.model_name2record.items():
            assert model_record.subgraph_paths
            model_path_prefix = os.path.dirname(model_record.subgraph_paths[0])
            for subgraph_idx in model_record.incorrect_subgraph_idxs:
                subgraph_path = os.path.join(
                    model_path_prefix, f"{model_name}_{subgraph_idx}"
                )
                if subgraph_idx == 0:
                    assert subgraph_path in model_record.subgraph_paths
                incorrect_models.append(subgraph_path)
        return incorrect_models

    def collect_decomposed_subgraphs(self, decomposed_samples_dir):
        for root, dirs, files in os.walk(decomposed_samples_dir):
            if path_utils.is_single_model_dir(root):
                model_name, _ = extract_model_name_and_subgraph_idx(root)
                assert model_name in self.model_name2record
                model_record = self.model_name2record[model_name]
                model_record.subgraph_paths.append(root)

    def collect_incorrect_subgraph_idxs(
        self, decompose_method, incorrect_models_from_log
    ):
        self.incorrect_models_from_log = list(sorted(incorrect_models_from_log))

        model_name2subgraph_idxs = {}
        for subgraph_path in sorted(self.incorrect_models_from_log):
            model_name, subgraph_idx = extract_model_name_and_subgraph_idx(
                subgraph_path
            )
            assert (
                model_name in self.model_name2record
            ), f"{model_name=}, {subgraph_idx=}"

            if model_name not in model_name2subgraph_idxs:
                model_name2subgraph_idxs[model_name] = []
            model_name2subgraph_idxs[model_name].append(subgraph_idx)

        if decompose_method == "fixed-start":
            for model_name in self.model_name2record.keys():
                if model_name not in model_name2subgraph_idxs:
                    if (
                        len(self.model_name2record[model_name].uniform_split_positions)
                        > 2
                    ):
                        model_name2subgraph_idxs[model_name] = [1]
                    else:
                        model_name2subgraph_idxs[model_name] = []
                else:
                    assert model_name2subgraph_idxs[model_name] == [0]

        for model_name, model_record in sorted(self.model_name2record.items()):
            model_record.incorrect_subgraph_idxs = model_name2subgraph_idxs.get(
                model_name, None
            )


@dataclass
class DecomposeConfig:
    framework: str
    decompose_method: str
    tolerance: int | List[int]
    max_subgraph_size: int = -1
    running_states: Dict[str, RunningState] = field(default_factory=dict)

    def save(self, work_dir):
        """Save the current config to a JSON file."""
        config_path = self.get_config_path(work_dir)

        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"\n[INFO] Save state to: {config_path}", flush=True)

    @classmethod
    def load(cls, work_dir):
        """Initialize a object from a JSON file."""
        config_path = cls.get_config_path(work_dir)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing configuration file: {config_path}")

        with open(config_path, "r") as f:
            data = json.load(f)

        config = cls(**data)
        for pass_key, value in config.running_states.items():
            if isinstance(value, dict):
                config.running_states[pass_key] = RunningState.init_from_dict(**value)
        return config

    @classmethod
    def get_config_path(self, work_dir) -> str:
        return os.path.join(work_dir, "decompose_config.json")

    def _get_pass_key(self, pass_id):
        assert isinstance(pass_id, int) and pass_id >= -1, f"{pass_id=} is illegal."
        return get_pass_name(pass_id) if pass_id >= 0 else "initial"

    def get_running_state(self, pass_id):
        pass_key = self._get_pass_key(pass_id)
        assert pass_key in self.running_states, f"{pass_key=} is unexpected."
        return self.running_states[pass_key]

    def get_incorrect_models(self, pass_id):
        if pass_id >= 0:
            return self.get_running_state(pass_id).get_incorrect_models(
                self.decompose_method
            )
        else:
            return self.get_running_state(-1).incorrect_models_from_log

    def update_running_state(self, pass_id, running_state):
        pass_key = self._get_pass_key(pass_id)
        self.running_states[pass_key] = running_state

    def update_running_state_with_incorrect_models(
        self, pass_id, incorrect_models_from_log
    ):
        pass_key = self._get_pass_key(pass_id)
        self.running_states[pass_key].collect_incorrect_subgraph_idxs(
            self.decompose_method, incorrect_models_from_log
        )


def get_rectfied_model_path(model_path):
    graphnet_root = path_utils.get_graphnet_root()
    return os.path.join(graphnet_root, model_path.split("GraphNet/")[-1])


def count_samples(samples_dir):
    num_samples = 0
    for root, dirs, files in os.walk(samples_dir):
        if path_utils.is_single_model_dir(root):
            num_samples += 1
    return num_samples


def get_model_name_with_subgraph_tag(model_path):
    fields = model_path.rstrip("/").split(os.sep)
    pattern = r"^subgraph(_\d+)?$"
    return f"{fields[-2]}_{fields[-1]}" if re.match(pattern, fields[-1]) else fields[-1]


def run_decomposer_for_single_model(
    framework: str,
    model_name: str,
    model_path: str,
    split_positions: List[int],
    output_dir: str,
    log_path: str,
) -> bool:
    """Decomposes a single model."""

    graphnet_root = path_utils.get_graphnet_root()
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
    if framework == "paddle":
        post_process_configs = {
            "post_extract_process_path": f"{graphnet_root}/graph_net/{framework}/graph_meta_restorer.py",
            "post_extract_process_class_name": "GraphMetaRestorer",
            "post_extract_process_config": {
                "update_inplace": True,
                "input_meta_allow_partial_update": False,
            },
        }
        for key, value in post_process_configs.items():
            decorator_config["decorator_config"]["custom_extractor_config"][key] = value

    decorator_config_b64 = convert_json_to_b64_string(decorator_config)

    print(
        f"[Decomposition] model_path: {model_path}, split_positions: {split_positions}",
        flush=True,
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
    framework, decompose_method, model_name2record, output_dir, log_path
):
    failed_decomposition_models = []
    for model_name, model_record in model_name2record.items():
        original_path = model_record.original_path
        split_positions = model_record.get_split_positions(decompose_method)

        rectified_model_path = get_rectfied_model_path(original_path)
        assert os.path.exists(
            rectified_model_path
        ), f"{rectified_model_path} does not exist."

        success = run_decomposer_for_single_model(
            framework,
            model_name,
            rectified_model_path,
            split_positions,
            output_dir,
            log_path,
        )
        if not success:
            failed_decomposition_models.append(rectified_model_path)
    return failed_decomposition_models


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

    print(f"[Evaluation] Logging to: {log_path}", flush=True)
    print(f"[Evaluation] command: {' '.join(cmd)}", flush=True)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=f, text=True)
    assert (
        result.returncode == 0
    ), f"[ERROR] test failed for {work_dir}, please check the log."


def generate_unittest_for_single_model(
    framework, model_name, model_path, subgraph_range, tolerance, output_dir, log_path
):
    graphnet_root = path_utils.get_graphnet_root()
    decorator_config = {
        "decorator_path": f"{graphnet_root}/graph_net/paddle/extractor.py",
        "decorator_config": {
            "name": model_name,
            "custom_extractor_path": f"{graphnet_root}/graph_net/paddle/prologue_subgraph_unittest_generator.py",
            "custom_extractor_config": {
                "output_dir": output_dir,
                "subgraph_range": subgraph_range,
                "device": "auto",
                "tolerance": tolerance,
                "try_run": True,
            },
        },
    }

    decorator_config_b64 = convert_json_to_b64_string(decorator_config)

    print(
        f"[Unittest] model_path: {model_path}, subgraph_range: {subgraph_range}",
        flush=True,
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
    assert result.returncode == 0


def generate_unittest(decompose_config, pass_id, output_dir):
    running_state = decompose_config.get_running_state(pass_id)

    unittest_dir = os.path.join(output_dir, "unittests")
    log_path = os.path.join(output_dir, "log_unittest_generation.txt")
    print(f"[Unittest] log_path: {log_path}", flush=True)
    for model_name, model_record in running_state.model_name2record.items():
        if not model_record.incorrect_subgraph_idxs:
            continue

        original_path = model_record.original_path
        subgraph_idx = model_record.incorrect_subgraph_idxs[0]
        if decompose_config.decompose_method == "fixed-start":
            subgraph_range = [0, model_record.uniform_split_positions[subgraph_idx + 1]]
        else:
            subgraph_range = [
                model_record.uniform_split_positions[subgraph_idx],
                model_record.uniform_split_positions[subgraph_idx + 1],
            ]

        rectified_model_path = get_rectfied_model_path(original_path)
        assert os.path.exists(
            rectified_model_path
        ), f"{rectified_model_path} does not exist."

        generate_unittest_for_single_model(
            decompose_config.framework,
            model_name,
            rectified_model_path,
            subgraph_range,
            decompose_config.tolerance[0],
            unittest_dir,
            log_path,
        )


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
    print(f"[Init] Pass 0: Reading from log file: {args.log_file}", flush=True)

    if args.decompose_method == "fixed-start":
        max_subgraph_size = MAX_GRAPH_SIZE
    else:
        max_subgraph_size = min(args.max_subgraph_size, MAX_GRAPH_SIZE)

    initial_split_positions = reconstruct_split_positions_for_subgraphs(
        [0, MAX_GRAPH_SIZE], 0, max_subgraph_size
    )

    model_name2record = {}
    initial_incorrect_models = get_ranged_incorrect_models(
        args.tolerance, args.log_file
    )
    for model_path in initial_incorrect_models:
        model_name = get_model_name_with_subgraph_tag(model_path)
        model_name2record[model_name] = ModelRecord(
            original_path=model_path,
            uniform_split_positions=initial_split_positions,
        )

    decompose_config = DecomposeConfig(
        framework=args.framework,
        decompose_method=args.decompose_method,
        tolerance=args.tolerance,
        max_subgraph_size=max_subgraph_size,
    )
    decompose_config.update_running_state(
        pass_id=-1,
        running_state=RunningState(
            incorrect_models_from_log=list(sorted(initial_incorrect_models))
        ),
    )
    decompose_config.update_running_state(
        pass_id=0, running_state=RunningState(model_name2record=model_name2record)
    )
    return decompose_config


def generate_successor_tasks(args, output_dir, pass_id):
    """Generates tasks for Pass > 0 based on previous pass results."""

    prev_pass_dir = get_decompose_workspace_path(output_dir, pass_id - 1)
    print(
        f"[Init] Resuming from Pass_{pass_id - 1} (Dir: {prev_pass_dir})...", flush=True
    )

    prev_config = DecomposeConfig.load(prev_pass_dir)
    max_subgraph_size = prev_config.max_subgraph_size // 2

    decompose_config = DecomposeConfig(
        framework=args.framework,
        decompose_method=args.decompose_method,
        tolerance=args.tolerance,
        max_subgraph_size=max_subgraph_size,
        running_states=prev_config.running_states,
    )
    if max_subgraph_size <= 0:
        return decompose_config

    prev_running_state = prev_config.get_running_state(pass_id - 1)
    assert prev_running_state is not None

    model_name2record = {}
    for model_name, pre_model_record in prev_running_state.model_name2record.items():
        if not pre_model_record.incorrect_subgraph_idxs:
            continue

        split_positions = reconstruct_split_positions_for_subgraphs(
            pre_model_record.uniform_split_positions,
            pre_model_record.incorrect_subgraph_idxs,
            max_subgraph_size,
        )
        if args.decompose_method == "fixed-start" and len(split_positions) > 3:
            split_positions = split_positions[0:3]

        model_name2record[model_name] = ModelRecord(
            original_path=pre_model_record.original_path,
            uniform_split_positions=split_positions,
        )

    decompose_config.update_running_state(
        pass_id=pass_id,
        running_state=RunningState(model_name2record=model_name2record),
    )
    return decompose_config


def prepare_tasks_and_verify(args, pass_id, output_dir):
    if pass_id == 0:
        decompose_config = generate_initial_tasks(args)
    else:
        decompose_config = generate_successor_tasks(args, output_dir, pass_id)

    print(
        f"[Init] initial max_subgraph_size: {decompose_config.max_subgraph_size}",
        flush=True,
    )
    print_incorrect_models(decompose_config, pass_id - 1, log_prompt="[Init]")

    if not decompose_config.get_incorrect_models(pass_id - 1):
        print(
            f"\n[Conclusion] No incorrect models after {pass_id - 1} steps.", flush=True
        )
        sys.exit(0)

    if decompose_config.max_subgraph_size <= 0:
        print(
            f"\n[Conclusion] Decomposition has reached the minimal granularity (max_subgraph_size = 1) after {pass_id - 1} steps.",
            flush=True,
        )
        generate_unittest(decompose_config, pass_id - 1, output_dir)
        sys.exit(0)

    return decompose_config


def execute_decomposition_phase(decompose_config, pass_id, workspace):
    """Executes the decomposition phase."""

    max_subgraph_size = decompose_config.max_subgraph_size
    running_state = decompose_config.get_running_state(pass_id)

    failed_decomposition_models = []
    need_decompose = True if len(running_state.model_name2record) > 0 else False
    decomposed_samples_dir = os.path.join(
        workspace,
        "samples" if decompose_config.framework == "torch" else "paddle_samples",
    )

    while need_decompose:
        if not os.path.exists(decomposed_samples_dir):
            os.makedirs(decomposed_samples_dir, exist_ok=True)
            print(
                f"[Decomposition] decomposed_samples_dir: {decomposed_samples_dir}",
                flush=True,
            )

        log_path = os.path.join(
            workspace, f"log_decompose-max_subgraph_size_{max_subgraph_size}.txt"
        )
        print(
            f"[Decomposition] max_subgraph_size: {max_subgraph_size}, log_path: {log_path}",
            flush=True,
        )
        failed_decomposition_models = run_decomposer_for_multi_models(
            decompose_config.framework,
            decompose_config.decompose_method,
            running_state.model_name2record,
            decomposed_samples_dir,
            log_path,
        )
        num_decomposed_samples = count_samples(decomposed_samples_dir)
        print(
            f"[Decomposition] number of graphs: {len(running_state.model_name2record)} -> {num_decomposed_samples}",
            flush=True,
        )
        if (
            not failed_decomposition_models
            and num_decomposed_samples == len(running_state.model_name2record)
            and max_subgraph_size > 1
            and decompose_config.decompose_method != "fixed-start"
        ):
            need_decompose = True
            shutil.rmtree(decomposed_samples_dir)
            os.makedirs(decomposed_samples_dir, exist_ok=True)
            max_subgraph_size = max(1, max_subgraph_size // 2)
            for model_name, model_record in running_state.model_name2record.items():
                if (
                    not model_record.uniform_split_positions
                    or len(model_record.uniform_split_positions) < 2
                ):
                    continue
                model_record.update_for_next_decompose(0, max_subgraph_size)
        else:
            need_decompose = False
        print()

    if failed_decomposition_models:
        print(
            f"[WARN] {len(failed_decomposition_models)} models failed to decompose.",
            flush=True,
        )

    running_state.collect_decomposed_subgraphs(decomposed_samples_dir)
    decompose_config.max_subgraph_size = max_subgraph_size
    return decompose_config


def print_incorrect_models(decompose_config, pass_id, log_prompt):
    incorrect_models = decompose_config.get_incorrect_models(pass_id)
    if pass_id == -1:
        original_model_paths = set(incorrect_models)
    else:
        original_model_paths = set(
            model_name
            for subgraph_path in incorrect_models
            for model_name, _ in [extract_model_name_and_subgraph_idx(subgraph_path)]
        )

    print(
        f"{log_prompt} number of incorrect subgraphs: {len(incorrect_models)}; number of incorrect original models: {len(original_model_paths)}",
        flush=True,
    )
    for idx, model_path in enumerate(incorrect_models):
        print(f"- [{idx}] {model_path}", flush=True)


def print_summary_and_suggestion(decompose_config, pass_id):
    print("\n" + "=" * 80, flush=True)
    num_incorrect_models = len(decompose_config.get_incorrect_models(pass_id))
    if num_incorrect_models > 0 and decompose_config.max_subgraph_size > 1:
        print(
            f">>> [SUGGESTION] Issues remain (Count: {num_incorrect_models}).",
            flush=True,
        )
        print(
            ">>> Please start next decomposition step (Run this script again).",
            flush=True,
        )
    elif decompose_config.max_subgraph_size <= 1:
        print(
            ">>> [Conclusion] Decomposition has reached the minimal granularity (max_subgraph_size = 1) after {pass_id - 1} steps.",
            flush=True,
        )
    print("=" * 80, flush=True)


def main(args):
    task_controller = TaskController(args)
    base_output_dir = task_controller.root_output_dir
    current_pass_id = task_controller.current_pass_id

    print("=" * 80)
    print(f" GraphNet Auto-Debugger | ROUND: PASS_{current_pass_id}")
    print("=" * 80)

    # --- Step 1: Prepare Tasks and Workspace ---
    decompose_config = prepare_tasks_and_verify(args, current_pass_id, base_output_dir)
    work_dir = get_decompose_workspace_path(base_output_dir, current_pass_id)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # --- Step 2: Decomposition ---
    if task_controller.task_scheduler["run_decomposer"]:
        print("\n--- Phase 1: Decomposition ---", flush=True)
        decompose_config = execute_decomposition_phase(
            decompose_config, current_pass_id, work_dir
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
        tolerance = args.tolerance[0]
        print(f"\n--- Phase 3: Analysis (torlance={tolerance}) ---")
        next_pass_incorrect_models = sorted(get_incorrect_models(tolerance, log_path))
        decompose_config.update_running_state_with_incorrect_models(
            pass_id=current_pass_id,
            incorrect_models_from_log=list(next_pass_incorrect_models),
        )
        print_incorrect_models(
            decompose_config, current_pass_id, log_prompt="[Analysis]"
        )
        print_summary_and_suggestion(decompose_config, current_pass_id)

    # --- Step 5: Save States ---
    decompose_config.save(work_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--framework", type=str, choices=["paddle", "torch"], required=True
    )
    parser.add_argument(
        "--test-config", type=str, required=True, help="Base64 encoded test config"
    )
    parser.add_argument(
        "--decompose-method",
        type=str,
        choices=["uniform", "fixed-start"],
        required=True,
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        nargs="+",
        choices=range(-10, 5),
        required=True,
        help="Tolerance level range [-10, 5)",
    )
    parser.add_argument("--max-subgraph-size", type=int, default=4096)
    args = parser.parse_args()
    print(args)
    main(args)
