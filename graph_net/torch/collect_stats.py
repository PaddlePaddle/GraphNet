import argparse
import os
import sys
import ast
import math
import importlib
import inspect
import subprocess
from typing import Type
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from functorch import make_fx
from graph_net.torch import utils


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def load_class_from_file(file_path: str, class_name: str) -> Type[torch.nn.Module]:
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


def get_input_dict(model_path, device):
    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    for tensor_meta in params.values():
        if hasattr(tensor_meta, "device"):
            tensor_meta.device = device
    return {
        k: utils.replay_tensor(v).to(torch.device(device)) for k, v in params.items()
    }


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


def resolve_native_multi_head_attention(*args, **kwargs):
    query, key, value = args[0], args[1], args[2]
    seq_len, batch_size, embed_dim = query.shape
    attn_output = torch.empty(
        (seq_len, batch_size, embed_dim), dtype=query.dtype, device="meta"
    )

    # TODO(Xreki): get value from args
    need_weights = False
    if need_weights:
        seq_len_k = key.shape[0]
        num_heads = args[4]
        attn_output_weights = torch.empty(
            (batch_size, num_heads, seq_len, seq_len_k),
            dtype=query.dtype,
            device="meta",
        )
        return attn_output, attn_output_weights
    else:
        return attn_output


def resolve_tensor_to(tensor, *args, **kwargs):
    if len(args) > 0 and isinstance(args[0], torch.dtype):
        dtype = args[0]
    else:
        dtype = tensor.dtype
    return torch.empty(tensor.shape, dtype=dtype, device="meta")


def resolve_tensor_item(tensor):
    return torch.empty((), dtype=tensor.dtype, device="meta")


def resolve_get_attr(gm: torch.fx.GraphModule, node: torch.fx.Node):
    attr_itr = node.target.split(".")
    val = gm
    for a in attr_itr:
        val = getattr(val, a)
    out = val.to(device="meta") if isinstance(val, torch.Tensor) else val
    return out


def convert_real_to_meta(x):
    if isinstance(x, torch.Tensor) and not x.is_meta:
        return torch.empty_like(x, device="meta")
    elif isinstance(x, (list, tuple)):
        return type(x)(convert_real_to_meta(v) for v in x)
    elif isinstance(x, dict):
        return {k: convert_real_to_meta(v) for k, v in x.items()}
    else:
        return x


def convert_meta_to_real(x, device):
    if isinstance(x, torch.Tensor) and x.is_meta:
        return torch.empty_like(x, device=device)
    elif isinstance(x, (list, tuple)):
        return type(x)(convert_meta_to_real(v, device) for v in x)
    elif isinstance(x, dict):
        return {k: convert_meta_to_real(v, device) for k, v in x.items()}
    else:
        return x


def resolve_with_real_tensor(op_func, device, meta_args, meta_kwargs):
    try:
        real_args = convert_meta_to_real(meta_args, device)
        real_kwargs = convert_meta_to_real(meta_kwargs, device)

        real_out = op_func(*real_args, **real_kwargs)
        return convert_real_to_meta(real_out)
    except Exception:
        return None


torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_sparse_compute = True
torch._dynamo.config.raise_on_ctx_manager_usage = False
torch._dynamo.config.allow_rnn = True


class GraphMetaExecutor:
    def __init__(self, device):
        self.device = device
        self.op_stats = {}
        self.is_complete = True
        self.num_ops = 0
        self.num_ops_misses_dtypes = 0
        self.subgraph_counter = 0

    def get_output_dtype(self, out):
        if isinstance(out, torch.Tensor):
            return out.dtype
        if (
            isinstance(out, (list, tuple))
            and len(out) > 0
            and isinstance(out[0], torch.Tensor)
        ):
            return out[0].dtype
        else:
            return None

    def get_op_name_and_func(self, gm, node, node_outputs):
        op_name = None
        op_func = None
        try:
            if node.op == "call_module":
                # classname of module
                submod = gm.get_submodule(node.target)
                op_name = submod.__class__.__name__
                op_func = submod
            elif node.op == "call_function":
                op_name = node.target.__name__
                op_func = node.target
            elif node.op == "call_method":
                op_name = node.target
                self_obj = (
                    node_outputs[node.args[0].name]
                    if isinstance(node.args[0], torch.fx.Node)
                    else node.args[0]
                )
                op_func = getattr(self_obj, node.target)
            elif node.op in ["get_attr", "placeholder", "output"]:
                op_name = node.op
        except Exception:
            pass
        return op_name, op_func

    def update_op_stats(self, op_stats, op_name, op_dtype):
        if op_name is not None:
            dtype_str = str(op_dtype).replace("torch.", "")
            if op_stats.get(op_name, None) is None:
                op_stats[op_name] = OpStat(op_name, {dtype_str: 1}, 1)
            else:
                op_stats[op_name].op_dtypes[dtype_str] = (
                    op_stats[op_name].op_dtypes.get(dtype_str, 0) + 1
                )
                op_stats[op_name].count += 1

    def __call__(self, gm: torch.fx.GraphModule, sample_inputs):
        # Use meta tensors as input to avoid actually running the model
        meta_sample_inputs = convert_real_to_meta(sample_inputs)

        op_stats = {}
        num_ops_misses_dtypes = 0

        input_idx = 0
        node_outputs = {}
        for node in gm.graph.nodes:
            out = None
            op_dtype = None
            op_name, op_func = self.get_op_name_and_func(gm, node, node_outputs)
            if node.op == "placeholder":
                out = meta_sample_inputs[input_idx]
                input_idx += 1
            elif node.op in ["call_function", "call_module", "call_method"]:
                try:
                    node_args = torch.fx.map_arg(
                        node.args,
                        lambda n: node_outputs[n.name]
                        if isinstance(n, torch.fx.Node)
                        else n,
                    )
                    node_kwargs = torch.fx.map_arg(
                        node.kwargs,
                        lambda n: node_outputs[n.name]
                        if isinstance(n, torch.fx.Node)
                        else n,
                    )
                    if node.op == "call_method":
                        node_args = node_args[1:]

                    if op_name == "_native_multi_head_attention":
                        out = resolve_native_multi_head_attention(
                            *node_args, **node_kwargs
                        )
                    elif op_name == "to":
                        out = resolve_tensor_to(
                            node_outputs[node.args[0].name], *node_args, **node_kwargs
                        )
                    elif op_name == "item":
                        out = resolve_tensor_item(node_outputs[node.args[0].name])
                    else:
                        assert op_func is not None, f"op_func of {node} is None."
                        out = op_func(*node_args, **node_kwargs)
                except Exception:
                    out = resolve_with_real_tensor(
                        op_func, self.device, node_args, node_kwargs
                    )
                    if out is None:
                        if num_ops_misses_dtypes == 0:
                            print(
                                f"dtype inference failed: node.op={node.op}, op_name={op_name}"
                            )
                        num_ops_misses_dtypes += 1
            elif node.op == "get_attr":
                out = resolve_get_attr(gm, node)
            elif node.op == "output":
                pass
            else:
                assert False, f"node.op: {node.op}"

            if out is not None:
                node_outputs[node.name] = out
                op_dtype = self.get_output_dtype(out)

            if node.op not in ["placeholder", "output"]:
                self.update_op_stats(op_stats, op_name, op_dtype)

        if num_ops_misses_dtypes > 0:
            self.is_complete = False
            self.num_ops_misses_dtypes += num_ops_misses_dtypes
        num_ops = 0
        for name, stat in op_stats.items():
            num_ops += stat.count
            if name in self.op_stats.keys():
                self.op_stats[name].update(stat)
            else:
                self.op_stats[name] = stat
        self.num_ops += num_ops
        self.subgraph_counter += 1
        return gm.forward

    def summary(self):
        print(
            f"Totally {self.subgraph_counter} subgraphs, {self.num_ops} operators, and {self.num_ops_misses_dtypes} operators failed to inference dtypes."
        )


def collect_op_stats_with_compile(model, sample_inputs, device):
    assert isinstance(model, torch.nn.Module), f"{type(model)=}"
    try:
        meta_executor = GraphMetaExecutor(device)
        compiled_model = torch.compile(model, backend=meta_executor)
        compiled_model(*sample_inputs)
        meta_executor.summary()
        return meta_executor.is_complete, meta_executor.op_stats
    except Exception:
        print("Failed with torch.compile")
        return False, None


def collect_op_stats_with_symbolic_trace(model, sample_inputs, device):
    assert isinstance(model, torch.nn.Module), f"{type(model)=}"
    try:
        # FX symbolic trace
        traced = torch.fx.symbolic_trace(model)
        # print(traced.graph)
    except Exception:
        print("Failed with symbolic_trace")
        return False, None

    meta_executor = GraphMetaExecutor(device)
    meta_executor(traced, sample_inputs)
    meta_executor.summary()
    return meta_executor.is_complete, meta_executor.op_stats


def collect_op_stats_with_make_fx(model, sample_inputs):
    # Use meta tensors as input to avoid actually running the model
    meta_input_list = convert_real_to_meta(sample_inputs)

    try:
        # Generate FX Graph, and automatically fill in meta information
        fx_model = make_fx(model)(*meta_input_list)
    except Exception:
        print("Failed to execute make_fx")
        return False, None

    is_complete = True
    op_stats = {}
    for node in fx_model.graph.nodes:
        op_name = None
        if node.op == "call_module":
            # classname of module
            submod = fx_model.get_submodule(node.target)
            op_name = submod.__class__.__name__
        elif node.op == "call_function":
            op_name = node.target.__name__
        elif node.op == "call_method":
            op_name = node.target
        elif node.op in ["placeholder", "output", "get_attr"]:
            op_name = node.op
        else:
            assert False, f"node.op: {node.op}"

        dtype = None
        if node.op not in ["placeholder", "output"]:
            if "tensor_meta" in node.meta:
                tensor_meta = node.meta["tensor_meta"]
                dtype = tensor_meta.dtype
                # print(f"node.op={node.op}, node.target={node.target}, dtype={tensor_meta.dtype}")
            else:
                print(
                    f"node.op={node.op}, node.target={node.target} has no tensor_meta!"
                )
                is_complete = False

        op_name = (
            op_name.replace(".default", "")
            .replace(".Tensor", "")
            .replace(".Scalar", "")
        )
        dtype_str = str(dtype).replace("torch.", "")
        if op_stats.get(op_name, None) is None:
            op_stats[op_name] = OpStat(op_name, {dtype_str: 1}, 1)
        else:
            op_stats[op_name].op_dtypes[dtype_str] = (
                op_stats[op_name].op_dtypes.get(dtype_str, 0) + 1
            )
            op_stats[op_name].count = op_stats[op_name].count + 1
    return is_complete, op_stats


def collect_op_stats(model, sample_inputs, device):
    is_complete_symbolic, op_stats_symbolic = collect_op_stats_with_symbolic_trace(
        model, sample_inputs, device
    )
    if not is_complete_symbolic:
        is_complete_compile, op_stats_compile = collect_op_stats_with_compile(
            model, sample_inputs, device
        )
        if is_complete_compile or op_stats_symbolic is None:
            return "torch.compile", is_complete_compile, op_stats_compile
    return "symbolic_trace", is_complete_symbolic, op_stats_symbolic


def collect_model_stats(model_path, device, log_prompt):
    file_path = os.path.join(model_path, "model.py")
    model_class = load_class_from_file(file_path, "GraphModule")
    model = model_class()
    argument_name2types = get_argument_name_and_types(model_class, "forward")
    num_outputs = get_number_of_returns(file_path, "GraphModule", "forward")

    input_dict = get_input_dict(model_path, device)
    ordered_input_list = [
        input_dict[arg_name] for arg_name in argument_name2types.keys()
    ]

    num_ops = 0
    ops_count_dict = {}
    op_dtypes = {}
    method, is_complete, op_stats = collect_op_stats(model, ordered_input_list, device)
    if op_stats is not None:
        for op_name, stat in sorted(op_stats.items()):
            if op_name not in ["placeholder", "output"]:
                num_ops += stat.count
                ops_count_dict[op_name] = stat.count
            for dtype_str, num in stat.op_dtypes.items():
                if dtype_str is not None and dtype_str != "None":
                    op_dtypes[dtype_str] = op_dtypes.get(dtype_str, 0) + num

    num_params = 0
    model_size = 0
    input_dtypes = {}
    param_dtypes = {}
    for name, arg_type in argument_name2types.items():
        if arg_type == torch.nn.parameter.Parameter:
            param_numel = math.prod(input_dict[name].shape)
            # print(f"Parameter {name}: {count}")
            num_params += 1
            model_size += param_numel
            dtype_str = str(input_dict[name].dtype).replace("torch.", "")
            param_dtypes[dtype_str] = param_dtypes.get(dtype_str, 0) + 1
        else:
            dtype_str = str(input_dict[name].dtype).replace("torch.", "")
            input_dtypes[dtype_str] = input_dtypes.get(dtype_str, 0) + 1
    model_size_in_billion = model_size / 1e9
    num_inputs = len(argument_name2types) - num_params

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
    print_with_log_prompt("method", method)
    print_with_log_prompt("is_complete", is_complete)


def main(args):
    if args.model_path is not None:
        assert os.path.isdir(args.model_path)
        assert is_single_model_dir(args.model_path)
        print(f"Collect information for {args.model_path}")
        collect_model_stats(args.model_path, args.device, args.log_prompt)
    else:
        graph_net_samples_path = (
            (graph_net.torch.samples_util.get_default_samples_directory())
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
            if is_single_model_dir(root) and root in previous_failed_model_pathes:
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
        description="Validate a computation graph sample. return 0 if success"
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
        help="Computation graph sample directory. e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--graph-net-samples-path",
        type=str,
        required=False,
        default=None,
        help="GraphNet samples directory. e.g '../../samples'",
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
