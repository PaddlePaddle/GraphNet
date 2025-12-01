import ast
import json
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Dict


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


@dataclass
class ModelStats:
    model_path: str
    num_inputs: int = None
    num_params: int = None
    num_outputs: int = None
    num_ops: int = None
    model_size_in_billion: float = None
    input_dtypes: Dict[str, int] = field(default_factory=dict)
    param_dtypes: Dict[str, int] = field(default_factory=dict)
    input_shapes: Dict[str, list] = field(default_factory=dict)
    op_dtypes: Dict[str, int] = field(default_factory=dict)
    ops: Dict[str, int] = field(default_factory=dict)
    source: str = None
    heuristic_tag: str = None


def print_model_stats(stats, log_prompt):
    assert isinstance(stats, ModelStats), f"{type(stats)=}"

    def print_with_log_prompt(key, value):
        print(
            f"{log_prompt} [ModelStats.{key}] model_path:{stats.model_path} {value}",
            flush=True,
        )

    print_with_log_prompt("num_inputs", stats.num_inputs)
    print_with_log_prompt("num_params", stats.num_params)
    print_with_log_prompt("num_outputs", stats.num_outputs)
    print_with_log_prompt("num_ops", stats.num_ops)
    print_with_log_prompt("model_size", f"{stats.model_size_in_billion}B")
    print_with_log_prompt("input_dtypes", json.dumps(stats.input_dtypes))
    print_with_log_prompt("param_dtypes", json.dumps(stats.param_dtypes))
    print_with_log_prompt("input_shapes", json.dumps(stats.input_shapes))
    print_with_log_prompt("op_dtypes", json.dumps(stats.op_dtypes))
    print_with_log_prompt("ops", json.dumps(stats.ops))
    print_with_log_prompt("source", stats.source)
    print_with_log_prompt("heuristic_tag", stats.heuristic_tag)


def load_class_from_file(file_path, class_name):
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
