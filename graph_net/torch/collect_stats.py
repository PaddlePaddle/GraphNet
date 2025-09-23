import argparse
import os
import sys
import math
import json
import subprocess
from datetime import datetime

import torch
from graph_net import collect_stats_util
from graph_net.torch import utils


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def read_graph_source_and_tag(model_path):
    try:
        with open(os.path.join(model_path, "graph_net.json"), "r") as f:
            data = json.load(f)
            return data["source"], data["heuristic_tag"]
    except Exception:
        return "unknown", "unknown"


def get_input_dict(model_path, device):
    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    for tensor_meta in params.values():
        if hasattr(tensor_meta, "device"):
            tensor_meta.device = device
    return {
        k: utils.replay_tensor(v).to(torch.device(device)) for k, v in params.items()
    }


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
                op_stats[op_name] = collect_stats_util.OpStat(
                    op_name, {dtype_str: 1}, 1
                )
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
        return meta_executor
    except Exception:
        print("Failed with torch.compile")
        return None


def collect_op_stats_with_symbolic_trace(model, sample_inputs, device):
    assert isinstance(model, torch.nn.Module), f"{type(model)=}"
    try:
        # FX symbolic trace
        traced = torch.fx.symbolic_trace(model)
    except Exception:
        print("Failed with symbolic_trace")
        return None

    meta_executor = GraphMetaExecutor(device)
    meta_executor(traced, sample_inputs)
    meta_executor.summary()
    return meta_executor


def collect_op_stats(model, sample_inputs, device):
    meta_executor_symbolic = collect_op_stats_with_symbolic_trace(
        model, sample_inputs, device
    )
    if meta_executor_symbolic is None or not meta_executor_symbolic.is_complete:
        meta_executor_compile = collect_op_stats_with_compile(
            model, sample_inputs, device
        )
        if meta_executor_symbolic is None or (
            meta_executor_compile is not None and meta_executor_compile.is_complete
        ):
            return "torch.compile", meta_executor_compile
    return "symbolic_trace", meta_executor_symbolic


def collect_model_stats(model_path, device, log_prompt):
    file_path = os.path.join(model_path, "model.py")
    model_class = collect_stats_util.load_class_from_file(file_path, "GraphModule")
    model = model_class()
    argument_name2types = collect_stats_util.get_argument_name_and_types(
        model_class, "forward"
    )

    input_dict = get_input_dict(model_path, device)
    ordered_input_list = [
        input_dict[arg_name] for arg_name in argument_name2types.keys()
    ]

    ops_count_dict = {}
    op_dtypes = {}
    method, meta_executor = collect_op_stats(model, ordered_input_list, device)
    if meta_executor is not None:
        for op_name, stat in sorted(meta_executor.op_stats.items()):
            if op_name not in ["placeholder", "output"]:
                ops_count_dict[op_name] = stat.count
            for dtype_str, num in stat.op_dtypes.items():
                if dtype_str is not None and dtype_str != "None":
                    op_dtypes[dtype_str] = op_dtypes.get(dtype_str, 0) + num

    model_size = 0
    input_shapes = set()
    input_dtypes = {}
    param_dtypes = {}
    for name, arg_type in argument_name2types.items():
        if (
            name.startswith("L_self_modules_")
            or arg_type == torch.nn.parameter.Parameter
        ):
            # Some parameters like L_self_modules_bn1_buffers_running_mean_ are torch.Tensor.
            param_numel = math.prod(input_dict[name].shape)
            model_size += param_numel
            dtype_str = str(input_dict[name].dtype).replace("torch.", "")
            param_dtypes[dtype_str] = param_dtypes.get(dtype_str, 0) + 1
        elif arg_type == torch.Tensor:
            dtype_str = str(input_dict[name].dtype).replace("torch.", "")
            input_dtypes[dtype_str] = input_dtypes.get(dtype_str, 0) + 1
            input_shapes.add(str(list(input_dict[name].shape)))

    num_outputs = collect_stats_util.get_number_of_returns(
        file_path, "GraphModule", "forward"
    )
    num_ops = meta_executor.num_ops if meta_executor is not None else 0
    source, heuristic_tag = read_graph_source_and_tag(model_path)

    is_complete = meta_executor.is_complete if meta_executor is not None else False
    print(
        f"model_stats collection information: model_path={model_path} method={method} is_ops_complete={is_complete}"
    )

    stats = collect_stats_util.ModelStats(
        model_path=model_path,
        num_inputs=sum(input_dtypes.values()),
        num_params=sum(param_dtypes.values()),
        num_outputs=num_outputs,
        num_ops=num_ops,
        model_size_in_billion=model_size / 1e9,
        input_dtypes=input_dtypes,
        param_dtypes=param_dtypes,
        input_shapes=list(input_shapes),
        op_dtypes=op_dtypes,
        ops=ops_count_dict,
        source=source,
        heuristic_tag=heuristic_tag,
    )
    collect_stats_util.print_model_stats(stats, log_prompt)


def main(args):
    if args.model_path is not None:
        assert os.path.isdir(args.model_path)
        assert is_single_model_dir(args.model_path)
        timestamp_sec = datetime.now().timestamp()
        dt = datetime.fromtimestamp(timestamp_sec)
        formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{formatted_dt}] Collect information for {args.model_path}")
        collect_model_stats(args.model_path, args.device, args.log_prompt)
    else:
        graph_net_samples_path = (
            (graph_net.torch.samples_util.get_default_samples_directory())
            if args.graph_net_samples_path is None
            else args.graph_net_samples_path
        )

        i = 0
        for root, dirs, files in os.walk(graph_net_samples_path):
            if is_single_model_dir(root):
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
        "--log-prompt",
        type=str,
        required=False,
        default="graph-net-collect-stats-log",
        help="Log prompt for stats log filtering.",
    )
    args = parser.parse_args()
    main(args=args)
