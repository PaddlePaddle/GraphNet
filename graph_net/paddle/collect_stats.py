import argparse
import os
import sys
import ast
import math
import importlib
import inspect
import subprocess
from datetime import datetime
from typing import Type
from dataclasses import dataclass, field
from collections import defaultdict

import paddle
from graph_net.paddle import utils


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def load_class_from_file(file_path: str, class_name: str) -> Type[paddle.nn.Layer]:
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    return model_class


def get_argument_name_and_types(model_class, func_name):
    argument_name2types = {}
    for name, func in inspect.getmembers(model_class, predicate=inspect.isfunction):
        if name == func_name:
            for arg_name, arg in inspect.signature(func).parameters.items():
                if arg_name != "self":
                    argument_name2types[arg_name] = (
                        None if arg.annotation is inspect._empty else arg.annotation
                    )
    return argument_name2types


def get_number_of_returns(file_path, class_name, func_name):
    source = None
    with open(f"{file_path}", "r") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == func_name:
                    for stmt in ast.walk(f):
                        if isinstance(stmt, ast.Return):
                            if stmt.value is None:
                                return 0
                            elif isinstance(stmt.value, ast.Tuple):
                                return len(stmt.value.elts)
                            else:
                                return 1
    return 0


def read_graph_source_and_tag(model_path):
    try:
        with open(os.path.join(model_path, "graph_net.json"), "r") as f:
            data = json.load(f)
            return data["source"], data["heuristic_tag"]
    except Exception:
        if "PaddleX" in model_path:
            return "PaddleX", "computer_vision"
        elif "PaddleNLP" in model_path:
            return "PaddleNLP", "nlp"
        elif "PaddleScience" in model_path:
            return "PaddleScience", "scientific_computing"
        else:
            return "unknown", "unknown"


def get_input_spec(model_path):
    inputs_params_list = utils.load_converted_list_from_text(f"{model_path}")
    input_spec = [None] * len(inputs_params_list)
    for i, v in enumerate(inputs_params_list):
        dtype = v["info"]["dtype"]
        shape = v["info"]["shape"]
        input_spec[i] = paddle.static.InputSpec(shape, dtype)
    return input_spec


@dataclass
class OpStat:
    op_name: str
    op_dtypes: dict[str, int] = field(default_factory=dict)
    count: int = 0

    def update(self, other):
        if isinstance(other, OpStat) and self.op_name == other.op_name:
            self.count += other.count
            for name, count in other.op_dtypes.items():
                self.op_dtypes[name] = self.op_dtypes.get(name, 0) + count


class ProgramAnalyzer:
    def __init__(self):
        self.op_stats = {}
        self.input_dict = {}
        self.num_ops = 0
        self.num_ops_misses_dtypes = 0
        self.is_complete = True

    def update_op_stats(self, op_name, op_dtype):
        if op_name is not None:
            dtype_str = str(op_dtype).replace("paddle.", "")
            if self.op_stats.get(op_name, None) is None:
                self.op_stats[op_name] = OpStat(op_name, {dtype_str: 1}, 1)
            else:
                self.op_stats[op_name].op_dtypes[dtype_str] = (
                    self.op_stats[op_name].op_dtypes.get(dtype_str, 0) + 1
                )
                self.op_stats[op_name].count += 1

    def __call__(self, program):
        assert isinstance(program, paddle.base.libpaddle.pir.Program)

        self.op_stats = {}
        self.num_ops_misses_dtypes = 0
        self.num_ops = 0
        for block in program.blocks:
            for op in block.ops:
                op_name = None
                op_dtype = None
                if op.name() == "pd_op.data":
                    op_attrs = op.attrs()
                    op_dtype = op_attrs["dtype"]
                    self.input_dict[op_attrs["name"]] = {
                        "dtype": str(op_dtype).replace("paddle.", ""),
                        "shape": op_attrs["shape"],
                    }
                elif not op.name().startswith("builtin."):
                    self.num_ops += 1
                    op_name = op.name().replace("pd_op.", "")
                    if len(op.results()) > 0:
                        op_dtype = op.results()[0].dtype

                if op_name is not None:
                    self.update_op_stats(op_name, op_dtype)
                elif op_dtype is None:
                    self.num_ops_misses_dtypes += 1

        if self.num_ops_misses_dtypes > 0:
            self.is_complete = False

    def summary(self):
        print(
            f"Totally {self.num_ops} operators, and {self.num_ops_misses_dtypes} operators failed to inference dtypes."
        )


def collect_op_stats(model, model_path):
    assert isinstance(model, paddle.nn.Layer), f"{type(model)=}"
    try:
        static_model = paddle.jit.to_static(
            model,
            input_spec=get_input_spec(model_path),
            full_graph=True,
            backend=None,
        )
        static_model.eval()
        program = static_model.forward.concrete_program.main_program

        program_analyzer = ProgramAnalyzer()
        program_analyzer(program)
        program_analyzer.summary()
        return program_analyzer
    except Exception:
        print("Failed with to_static")
        return None


def collect_model_stats(model_path, log_prompt):
    file_path = os.path.join(model_path, "model.py")
    model_class = load_class_from_file(file_path, "GraphModule")
    model = model_class()
    num_outputs = get_number_of_returns(file_path, "GraphModule", "forward")

    model_size = 0
    input_dtypes = {}
    param_dtypes = {}
    ops_count_dict = {}
    op_dtypes = {}

    program_analyzer = collect_op_stats(model, model_path)
    if program_analyzer is not None:
        for op_name, stat in sorted(program_analyzer.op_stats.items()):
            ops_count_dict[op_name] = stat.count
            for dtype_str, num in stat.op_dtypes.items():
                if dtype_str is not None and dtype_str != "None":
                    op_dtypes[dtype_str] = op_dtypes.get(dtype_str, 0) + num

        inputs_params = utils.load_converted_from_text(f"{model_path}")
        params = inputs_params["weight_info"]
        inputs = inputs_params["input_info"]

        for name, value in program_analyzer.input_dict.items():
            dtype_str = value["dtype"]
            if name in params.keys():
                param_numel = math.prod(value["shape"])
                model_size += param_numel
                param_dtypes[dtype_str] = param_dtypes.get(dtype_str, 0) + 1
            elif name in inputs.keys():
                input_dtypes[dtype_str] = input_dtypes.get(dtype_str, 0) + 1

    model_size_in_billion = model_size / 1e9
    num_params = sum(param_dtypes.values())
    num_inputs = sum(input_dtypes.values())
    num_ops = sum(ops_count_dict.values())
    source, heuristic_tag = read_graph_source_and_tag(model_path)
    method = "to_static"
    is_complete = (
        program_analyzer.is_complete if program_analyzer is not None else False
    )

    def dict_to_string(d):
        kv_list = [f"{k}:{v}" for k, v in d.items()]
        return " ".join(kv_list)

    def print_with_log_prompt(key, value):
        print(
            f"{log_prompt} [ModelStats.{key}] model_path:{model_path} {value}",
            flush=True,
        )

    print_with_log_prompt("num_inputs", num_inputs)
    print_with_log_prompt("num_params", num_params)
    print_with_log_prompt("num_outputs", num_outputs)
    print_with_log_prompt("num_ops", num_ops)
    print_with_log_prompt("model_size", f"{model_size_in_billion}B")
    print_with_log_prompt("input_dtypes", dict_to_string(input_dtypes))
    print_with_log_prompt("param_dtypes", dict_to_string(param_dtypes))
    print_with_log_prompt("op_dtypes", dict_to_string(op_dtypes))
    print_with_log_prompt("ops", dict_to_string(ops_count_dict))
    print_with_log_prompt("source", source)
    print_with_log_prompt("heuristic_tag", heuristic_tag)
    print_with_log_prompt("method", method)
    print_with_log_prompt("is_complete", is_complete)


def main(args):
    if args.model_path is not None:
        assert os.path.isdir(args.model_path)
        assert is_single_model_dir(args.model_path)
        timestamp_sec = datetime.now().timestamp()
        dt = datetime.fromtimestamp(timestamp_sec)
        formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{formatted_dt}] Collect information for {args.model_path}")
        collect_model_stats(args.model_path, args.log_prompt)
    else:
        graph_net_samples_path = (
            (graph_net.paddle.samples_util.get_default_samples_directory())
            if args.graph_net_samples_path is None
            else args.graph_net_samples_path
        )

        previous_failed_model_pathes = []
        if args.previous_collect_result_path is not None:
            with open(args.previous_collect_result_path, "r") as f:
                for line in f.readlines():
                    if "[ModelStats]" in line:
                        fields = line.strip().split()
                        model_path = fields[2].split(":")[-1]
                        is_complete = fields[-1].split(":")[-1]
                        if is_complete == "False":
                            previous_failed_model_pathes.append(model_path)

        i = 0
        for root, dirs, files in os.walk(graph_net_samples_path):
            if is_single_model_dir(root) and (
                args.previous_collect_result_path is None
                or root in previous_failed_model_pathes
            ):
                print(f"[{i}] Collect information for {root}")
                cmd = [
                    "python",
                    "-m",
                    "graph_net.torch.collect_stats",
                    f"--device={args.device}",
                    f"--model-path={root}",
                    f"--log-prompt={args.log_prompt}",
                ]
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600,
                )
                print(result.stdout)
                if result.returncode != 0:
                    print(result.stderr)
                i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect stats for computation graph samples. return 0 if success"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for testing the compiler (e.g., 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=None,
        help="Computation graph sample directory. e.g '../../paddle_samples/PaddleX/ResNet18'",
    )
    parser.add_argument(
        "--graph-net-samples-path",
        type=str,
        required=False,
        default=None,
        help="GraphNet samples directory. e.g '../../paddle_samples'",
    )
    parser.add_argument(
        "--previous-collect-result-path",
        type=str,
        required=False,
        default=None,
        help="Previous collect result path, use to recollect the failed cases",
    )
    parser.add_argument(
        "--log-prompt",
        type=str,
        required=False,
        default="graph-net-collect-stats-log",
        help="Log prompt for stats log filtering.",
    )
    args = parser.parse_args()
    main(args=args)
