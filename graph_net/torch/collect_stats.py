import argparse
import os
import sys
import math
import importlib
import inspect
from typing import Type
from dataclasses import dataclass, field
from collections import defaultdict

import torch
from torch.fx.passes.shape_prop import ShapeProp
from graph_net.torch import utils


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def load_class_from_file(file_path: str, class_name: str) -> Type[torch.nn.Module]:
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    return model_class


def get_argument_types(model_class, func_name):
    arg_types = {}
    for name, func in inspect.getmembers(model_class, predicate=inspect.isfunction):
        if name == func_name:
            for arg_name, arg in inspect.signature(func).parameters.items():
                if arg_name != "self":
                    arg_types[arg_name] = (
                        None if arg.annotation is inspect._empty else arg.annotation
                    )
    return arg_types


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
    dtype: set[str] = field(default_factory=set)
    count: int = 0


def resolve_native_multi_head_attention(*args, **kwargs):
    query, key, value = args[0], args[1], args[2]
    seq_len, batch_size, embed_dim = query.shape
    attn_output = torch.empty(
        (seq_len, batch_size, embed_dim), dtype=query.dtype, device="meta"
    )

    # seq_len_k = key.shape[0]
    # num_heads = args[4]
    # attn_output_weights = torch.empty((batch_size, num_heads, seq_len, seq_len_k),
    #                                  dtype=query.dtype, device='meta')
    return attn_output  # , attn_output_weights


def resolve_get_attr(gm: torch.fx.GraphModule, node: torch.fx.Node):
    attr_itr = node.target.split(".")
    val = gm
    for a in attr_itr:
        val = getattr(val, a)
    out = val.to(device="meta") if isinstance(val, torch.Tensor) else val
    return out


def collect_op_stats(model, input_dict):
    try:
        # FX symbolic trace
        traced = torch.fx.symbolic_trace(model)
        # print(traced.graph)
    except Exception:
        print("Failed to FX symbolic trace")
        return False, None

    # Use meta tensors as input to avoid actually running the model
    meta_input_dict = {}
    for name, x in input_dict.items():
        meta_input_dict[name] = (
            torch.empty_like(x, device="meta") if isinstance(x, torch.Tensor) else x
        )

    is_complete = True
    op_stats = {}
    node_outputs = {}
    for node in traced.graph.nodes:
        op_name = None
        dtype = None
        if node.op == "placeholder":
            node_outputs[node.name] = meta_input_dict[node.target]
            op_name = node.op
            dtype = node_outputs[node.name].dtype
        elif node.op in ["call_function", "call_module", "call_method"]:
            node_args = torch.fx.map_arg(
                node.args,
                lambda n: node_outputs[n.name] if isinstance(n, torch.fx.Node) else n,
            )
            node_kwargs = torch.fx.map_arg(
                node.kwargs,
                lambda n: node_outputs[n.name] if isinstance(n, torch.fx.Node) else n,
            )

            try:
                if node.op == "call_module":
                    # classname of module
                    submod = traced.get_submodule(node.target)
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
                    node_args = node_args[1:]

                if op_name == "_native_multi_head_attention":
                    out = resolve_native_multi_head_attention(*node_args, **node_kwargs)
                else:
                    out = op_func(*node_args, **node_kwargs)
                node_outputs[node.name] = out
                dtype = out.dtype if isinstance(out, torch.Tensor) else None
            except Exception:
                print(f"dtype inference failed: node.op={node.op}, op_name={op_name}")
                node_outputs[node.name] = None
                is_complete = False
        elif node.op == "get_attr":
            op_name = node.op
            out = resolve_get_attr(traced, node)
            node_outputs[node.name] = out
            dtype = out.dtype if isinstance(out, torch.Tensor) else None
        elif node.op == "output":
            op_name = node.op
            node_args = torch.fx.map_arg(
                node.args,
                lambda n: node_outputs[n.name] if isinstance(n, torch.fx.Node) else n,
            )
            node_outputs[node.name] = node_args[0] if len(node_args) == 1 else node_args
            dtype = (
                node_args[0].dtype if isinstance(node_args[0], torch.Tensor) else None
            )
        else:
            assert False, f"node.op: {node.op}"

        if op_name is not None:
            dtype_str = str(dtype).replace("torch.", "") if dtype is not None else None
            if op_stats.get(op_name, None) is None:
                op_stats[op_name] = OpStat(op_name, {dtype_str}, 1)
            else:
                op_stats[op_name].dtype.add(dtype_str)
                op_stats[op_name].count = op_stats[op_name].count + 1
    return is_complete, op_stats


def collect_model_stats(model_path, device, log_prompt):
    if not hasattr(collect_model_stats, "_counter"):
        collect_model_stats._counter = 0
    else:
        collect_model_stats._counter += 1
    print(f"[{collect_model_stats._counter}] Collect information for {model_path}")

    model_class = load_class_from_file(
        os.path.join(model_path, "model.py"), "GraphModule"
    )
    model = model_class()
    input_dict = get_input_dict(model_path, device)

    num_ops = 0
    num_inputs = 0
    num_outputs = 0
    dtypes = set()
    is_complete, op_stats = collect_op_stats(model, input_dict)
    if op_stats is not None:
        for op_name, stat in op_stats.items():
            if op_name == "placeholder":
                num_inputs += stat.count
            elif op_name == "output":
                num_outputs += stat.count
            else:
                num_ops += stat.count
            for v in stat.dtype:
                if v is not None:
                    dtypes.add(v)

    arg_types = get_argument_types(model_class, "forward")
    num_inputs = len(arg_types) if op_stats is None else num_inputs
    num_params = 0
    param_dtypes = set()
    for name, arg_type in arg_types.items():
        if arg_type == torch.nn.parameter.Parameter:
            count = math.prod(input_dict[name].shape)
            # print(f"Parameter {name}: {count}")
            num_params += count
            param_dtypes.add(str(input_dict[name].dtype).replace("torch.", ""))
    num_params_in_billion = num_params / 1e9

    dtypes_str = "[" + ",".join(dtypes) + "]"
    param_dtypes_str = "[" + ",".join(param_dtypes) + "]"
    print(
        f"{log_prompt} [ModelStats] model_path:{model_path} num_inputs:{num_inputs} num_outputs:{num_outputs} num_ops:{num_ops} num_params:{num_params_in_billion}B param_dtypes:{param_dtypes_str} op_dtypes:{dtypes_str} is_complete:{is_complete}",
        file=sys.stderr,
        flush=True,
    )


def main(args):
    if args.model_path is not None:
        assert os.path.isdir(args.model_path)
        assert is_single_model_dir(args.model_path)
        collect_model_stats(args.model_path, args.device, args.log_prompt)
    else:
        graph_net_samples_path = (
            (graph_net.torch.samples_util.get_default_samples_directory())
            if args.graph_net_samples_path is None
            else args.graph_net_samples_path
        )
        for root, dirs, files in os.walk(graph_net_samples_path):
            if is_single_model_dir(root):
                collect_model_stats(root, args.device, args.log_prompt)


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
        "--log-prompt",
        type=str,
        required=False,
        default="graph-net-collect-stats-log",
        help="Log prompt for stats log filtering.",
    )
    args = parser.parse_args()
    main(args=args)
