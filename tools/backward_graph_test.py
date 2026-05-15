#!/usr/bin/env python3
"""Batch backward graph generation and test_compiler validation tool.

Usage:
    # Only generate backward graphs
    python tools/backward_graph_test.py \
        --sample-root /path/to/samples \
        --limit 100 \
        --output-dir /tmp/bw_results

    # Generate + test_compiler + kernel collection
    python tools/backward_graph_test.py \
        --sample-root /path/to/samples \
        --limit 20 \
        --output-dir /tmp/bw_results \
        --test-compiler \
        --collect-kernels \
        --device cuda
"""

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_net.torch.fx_graph_module_util import (
    get_torch_module_and_inputs,
)
from graph_net.torch.fx_graph_module_util import _get_tensor_metas as get_tensor_metas


def find_samples(root_dir, limit=-1):
    """Recursively find sample directories containing model.py."""
    samples = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "model.py" in filenames:
            samples.append(dirpath)
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def load_model_from_path(model_path, device="cpu"):
    """Load GraphModule and inputs from a sample directory."""
    module, inputs = get_torch_module_and_inputs(
        model_path, use_dummy_inputs=False, device=device
    )
    return module, inputs


def set_requires_grad_by_meta(model_path, module, inputs):
    """Set requires_grad based on weight_meta original_name."""
    try:
        tensor_metas = get_tensor_metas(model_path)
    except Exception:
        return inputs

    name2tensor_meta = {tm.name: tm for tm in tensor_metas}
    param_names = list(inspect.signature(module.forward).parameters.keys())
    for input_idx, name in enumerate(param_names):
        if input_idx >= len(inputs):
            break
        tensor = inputs[input_idx]
        if not isinstance(tensor, torch.Tensor):
            continue
        if not tensor.is_floating_point():
            continue
        tm = name2tensor_meta.get(name)
        if tm is None:
            continue
        check_name = tm.original_name or tm.name
        nograd_keywords = [
            "running_mean",
            "running_var",
            "num_batches_tracked",
            "mask",
            "indices",
            "position_ids",
            "anchor",
        ]
        if any(kw in check_name for kw in nograd_keywords):
            tensor.requires_grad = False
        else:
            tensor.requires_grad = True
    return inputs


def is_pure_shape_graph(module):
    """Check if the graph only contains shape manipulation ops."""
    if not hasattr(module, "graph"):
        return False
    shape_only_ops = {
        torch.ops.aten.view,
        torch.ops.aten.reshape,
        torch.ops.aten.squeeze,
        torch.ops.aten.unsqueeze,
        torch.ops.aten.permute,
        torch.ops.aten.transpose,
        torch.ops.aten.expand,
        torch.ops.aten.flatten,
        torch.ops.aten.t,
        "view",
        "reshape",
        "squeeze",
        "unsqueeze",
        "permute",
        "transpose",
        "expand",
        "flatten",
        "t",
    }
    for node in module.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        if node.op == "call_function" and node.target in shape_only_ops:
            continue
        if node.op == "call_method" and node.target in shape_only_ops:
            continue
        return False
    return True


def _tensor_meta_py_str(name, shape, dtype, device, mean=0.0, std=1.0):
    dtype_str = str(dtype).replace("torch.", "torch.")
    return (
        f"class Program_input_tensor_meta_{name}:\n"
        f'    name = "{name}"\n'
        f"    shape = {list(shape)}\n"
        f'    dtype = "{dtype_str}"\n'
        f'    device = "{device}"\n'
        f"    mean = {mean:.3f}\n"
        f"    std = {std:.3f}\n"
        f"    data = None\n"
    )


def _save_backward_model(gm, backward_inputs, output_path):
    """Save backward GraphModule with model.py, input_meta.py, weight_meta.py."""
    os.makedirs(output_path, exist_ok=True)

    # model.py
    model_py_path = os.path.join(output_path, "model.py")
    code = gm.code
    if "class GraphModule" not in code:
        code = "import torch\n\n" "class GraphModule(torch.nn.Module):\n" + "\n".join(
            "    " + line if line.strip() else "" for line in code.split("\n")
        )
    with open(model_py_path, "w", encoding="utf-8") as f:
        f.write(code)

    # GraphNet test_compiler reads weight_meta.py as model inputs.
    # Write backward graph inputs into weight_meta.py.
    param_names = list(inspect.signature(gm.forward).parameters.keys())
    weight_meta_lines = []
    for idx, name in enumerate(param_names):
        if idx < len(backward_inputs):
            t = backward_inputs[idx]
            if isinstance(t, torch.Tensor):
                weight_meta_lines.append(
                    _tensor_meta_py_str(
                        name,
                        t.shape,
                        t.dtype,
                        str(t.device),
                        mean=0.0,
                        std=1.0,
                    )
                )
    weight_meta_path = os.path.join(output_path, "weight_meta.py")
    with open(weight_meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(weight_meta_lines))

    # input_meta.py: empty (test_compiler does not use it)
    input_meta_path = os.path.join(output_path, "input_meta.py")
    with open(input_meta_path, "w", encoding="utf-8") as f:
        f.write("")


def capture_backward_graph(module, inputs, device="cpu"):
    """Capture forward and backward FX Graph via aot_autograd.

    Returns:
        (backward_gm, backward_inputs) or (None, None) if no valid grad pairs.
    """
    gm_holder = {}
    backward_inputs = []

    def forward_compiler(fx_gm, fwd_inputs):
        gm_holder["forward_gm"] = fx_gm
        return fx_gm

    def backward_compiler(fx_gm, bwd_inputs):
        gm_holder["backward_gm"] = fx_gm
        placeholders = [n for n in fx_gm.graph.nodes if n.op == "placeholder"]
        origin_forward = fx_gm.forward
        fx_gm._original_forward = origin_forward

        def wrapped_forward(*args):
            for node, arg in zip(placeholders, args):
                backward_inputs.append(arg.detach().clone())
            return origin_forward(*args)

        fx_gm.forward = wrapped_forward
        return make_boxed_func(fx_gm)

    compiled = aot_module_simplified(
        module,
        inputs,
        fw_compiler=forward_compiler,
        bw_compiler=backward_compiler,
    )
    outs = compiled(*inputs)
    outs = [outs] if isinstance(outs, torch.Tensor) else outs
    valid_pairs = [
        (out, torch.ones_like(out))
        for out in outs
        if isinstance(out, torch.Tensor) and out.requires_grad
    ]

    if not valid_pairs:
        return None, None

    tensors, grads = zip(*valid_pairs)
    torch.autograd.backward(tensors, grads)

    backward_gm = gm_holder.get("backward_gm")
    if backward_gm is not None:
        # Restore original forward for correct signature when saving
        if hasattr(backward_gm, "_original_forward"):
            backward_gm.forward = backward_gm._original_forward
        backward_gm = _remove_none_from_output(backward_gm)
    return backward_gm, backward_inputs


def _remove_none_from_output(gm):
    output_node = next(
        (n for n in gm.graph.nodes if n.op == "output"),
        None,
    )
    if output_node is None:
        return gm
    outs = (
        output_node.args[0]
        if output_node and isinstance(output_node.args, (tuple, list))
        else output_node.args
    )
    if isinstance(outs, (tuple, list)):
        new_outs = tuple(out for out in outs if out is not None)
        if new_outs != outs:
            output_node.args = (new_outs,)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def run_test_compiler(backward_model_path, device="cuda", compiler="nope", trials=10):
    """Run graph_net_bench torch test_compiler on a backward model."""
    env = os.environ.copy()
    env["GRAPH_NET_FLUCTUATION_DETECT_THRESHOLD"] = "0.5"
    cmd = [
        sys.executable,
        "-m",
        "graph_net_bench.torch.test_compiler",
        f"--model-path={backward_model_path}",
        f"--compiler={compiler}",
        f"--device={device}",
        "--warmup=3",
        f"--trials={trials}",
        "--log-prompt=graph-net-backward-test-compiler-log",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
        stdout = result.stdout
        stderr = result.stderr
        success = result.returncode == 0 and "[Result] status: success" in stderr
        return success, stdout, stderr, result.returncode
    except subprocess.TimeoutExpired:
        return False, "", "Timeout", -1
    except Exception as e:
        return False, "", str(e), -1


def collect_triton_kernels(backward_model_path, device="cuda"):
    """Collect Triton kernels by running torch.compile with inductor backend."""
    kernels = []
    try:
        import logging

        triton_logger = logging.getLogger("torch._inductor.codecache")
        triton_handler = None
        kernel_codes = []

        class KernelCaptureHandler(logging.Handler):
            def emit(self, record):
                msg = record.getMessage()
                if "triton" in msg.lower() and ".py" in msg.lower():
                    kernel_codes.append(msg)

        triton_handler = KernelCaptureHandler()
        triton_logger.addHandler(triton_handler)
        triton_logger.setLevel(logging.DEBUG)

        module, inputs = get_torch_module_and_inputs(
            backward_model_path, use_dummy_inputs=False, device=device
        )
        compiled = torch.compile(module, backend="inductor")
        _ = compiled(*inputs)
        torch.cuda.synchronize() if "cuda" in device else None

        triton_logger.removeHandler(triton_handler)
        kernels = kernel_codes
    except Exception as e:
        kernels = [f"Error collecting kernels: {e}"]
    return kernels


def process_single_sample(
    sample_path,
    output_dir,
    device="cpu",
    test_compiler=False,
    collect_kernels=False,
    replace_inplace=False,
    skip_pure_shape=True,
):
    """Process a single sample: generate backward graph, optionally test and collect kernels.

    Returns:
        dict with status and paths.
    """
    rel_path = os.path.relpath(
        sample_path, os.path.dirname(os.path.dirname(sample_path))
    )
    result = {
        "sample": sample_path,
        "rel_path": rel_path,
        "status": "unknown",
        "reason": "",
        "backward_path": None,
        "test_compiler_success": None,
        "kernels": [],
    }

    try:
        module, inputs = load_model_from_path(sample_path, device=device)
        module.eval()

        if skip_pure_shape and is_pure_shape_graph(module):
            result["status"] = "skipped"
            result["reason"] = "pure_shape_graph"
            return result

        inputs = [
            inp.detach().clone() if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]
        inputs = set_requires_grad_by_meta(sample_path, module, inputs)

        backward_gm, backward_inputs = capture_backward_graph(
            module, inputs, device=device
        )

        if backward_gm is None:
            result["status"] = "failed"
            result["reason"] = "no_valid_grad_pairs"
            return result

        backward_dir = os.path.join(output_dir, "backward_graphs", rel_path)
        os.makedirs(backward_dir, exist_ok=True)
        _save_backward_model(backward_gm, backward_inputs, backward_dir)

        # Copy graph_net.json if it exists
        src_json = os.path.join(sample_path, "graph_net.json")
        if os.path.exists(src_json):
            shutil.copy2(src_json, os.path.join(backward_dir, "graph_net.json"))

        result["backward_path"] = backward_dir
        result["status"] = "success"

        if test_compiler:
            success, stdout, stderr, rc = run_test_compiler(
                backward_dir, device=device, compiler="nope", trials=10
            )
            result["test_compiler_success"] = success
            result["test_compiler_rc"] = rc
            result["test_compiler_stderr"] = (
                stderr[-2000:] if len(stderr) > 2000 else stderr
            )
            if not success:
                result["status"] = "test_compiler_failed"

        if collect_kernels and result["status"] == "success":
            kernels = collect_triton_kernels(backward_dir, device=device)
            result["kernels"] = kernels

    except Exception as e:
        result["status"] = "exception"
        result["reason"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch backward graph generation and validation."
    )
    parser.add_argument(
        "--sample-root",
        type=str,
        required=True,
        help="Root directory containing subdirectories with model.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for backward graphs and results",
    )
    parser.add_argument(
        "--test-compiler",
        action="store_true",
        help="Run test_compiler on generated backward graphs",
    )
    parser.add_argument(
        "--collect-kernels",
        action="store_true",
        help="Collect Triton kernels from backward graphs",
    )
    parser.add_argument(
        "--replace-inplace",
        action="store_true",
        help="Auto replace inplace=True with inplace=False in model code",
    )
    parser.add_argument(
        "--skip-pure-shape",
        action="store_true",
        default=True,
        help="Skip pure shape operation subgraphs (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model execution",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed samples in output dir",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    samples = find_samples(args.sample_root, limit=args.limit)
    print(f"Found {len(samples)} samples under {args.sample_root}")

    results = []
    stats = {"total": 0, "success": 0, "skipped": 0, "failed": 0, "exception": 0}

    for idx, sample_path in enumerate(samples):
        print(f"[{idx + 1}/{len(samples)}] Processing {sample_path} ...")
        stats["total"] += 1

        if args.resume:
            rel_path = os.path.relpath(
                sample_path, os.path.dirname(os.path.dirname(sample_path))
            )
            backward_dir = os.path.join(args.output_dir, "backward_graphs", rel_path)
            if os.path.exists(os.path.join(backward_dir, "model.py")):
                print("  [Resume] Skip already processed.")
                continue

        result = process_single_sample(
            sample_path,
            args.output_dir,
            device=args.device,
            test_compiler=args.test_compiler,
            collect_kernels=args.collect_kernels,
            replace_inplace=args.replace_inplace,
            skip_pure_shape=args.skip_pure_shape,
        )
        results.append(result)

        status = result["status"]
        if status in stats:
            stats[status] += 1
        else:
            stats["exception"] += 1

        print(f"  Status: {status}")
        if result.get("reason"):
            print(f"  Reason: {result['reason']}")
        if result.get("test_compiler_success") is not None:
            print(
                f"  test_compiler: {'success' if result['test_compiler_success'] else 'failed'}"
            )

    # Save results
    summary = {
        "args": vars(args),
        "stats": stats,
        "results": results,
    }
    result_path = os.path.join(args.output_dir, "backward_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print(f"Total: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Exception: {stats['exception']}")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
