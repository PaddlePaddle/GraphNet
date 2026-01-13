#!/usr/bin/env python3
"""
Remote reference device test client.
Executes test_reference_device.py on remote server and retrieves outputs.
"""

import argparse
import torch
import os
import sys
from pathlib import Path

from graph_net.graph_net_bench.sample_remote_executor import SampleRemoteExecutor
from graph_net import path_utils
from graph_net import test_compiler_util


def get_reference_log_path(reference_dir, model_path):
    model_name = model_path.split("torch_samples/")[-1].replace(os.sep, "_")
    return os.path.join(reference_dir, f"{model_name}.log")


def get_reference_output_path(reference_dir, model_path):
    model_name = model_path.split("torch_samples/")[-1].replace(os.sep, "_")
    return os.path.join(reference_dir, f"{model_name}.pth")


def test_single_model_remote(args):
    ref_log = get_reference_log_path(args.reference_dir, args.model_path)
    ref_dump = get_reference_output_path(args.reference_dir, args.model_path)
    
    print(f"Reference log path: {ref_log}", file=sys.stderr, flush=True)
    print(f"Reference outputs path: {ref_dump}", file=sys.stderr, flush=True)
    
    # 1. 初始化远程执行器
    executor = SampleRemoteExecutor(
        machine=args.machine,
        port=args.port,
        rpc_cmd=args.rpc_cmd,
    )
    
    try:
        # 2. 调用远程执行
        print(f"Sending request to {args.machine}:{args.port}...", file=sys.stderr)
        files_dict = executor(args.model_path, args.random_seed)
        
        # 3. 处理返回的文件
        log_filename = Path(ref_log).name
        pth_filename = Path(ref_dump).name
        
        if log_filename in files_dict:
            log_content = files_dict[log_filename]
            with open(ref_log, "wb") as f:
                f.write(log_content)
            print(f"Saved log to {ref_log}", file=sys.stderr)
        else:
            print(f"Warning: log file {log_filename} not found in remote output", file=sys.stderr)
        
        if pth_filename in files_dict:
            pth_content = files_dict[pth_filename]
            with open(ref_dump, "wb") as f:
                f.write(pth_content)
            print(f"Saved outputs to {ref_dump}", file=sys.stderr)
            
            # 验证加载
            try:
                outputs = torch.load(ref_dump)
                print(f"Loaded {len(outputs) if isinstance(outputs, tuple) else 1} output tensor(s)", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to load outputs: {e}", file=sys.stderr)
        else:
            print(f"Warning: output file {pth_filename} not found in remote output", file=sys.stderr)
        
        # 4. 将日志内容打印到 stderr（与 test_reference_device.py 一致）
        if log_filename in files_dict:
            print(files_dict[log_filename].decode('utf-8'), file=sys.stderr, flush=True)
        
        print("Remote execution completed successfully!", file=sys.stderr)
        
    except Exception as e:
        print(f"Remote execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        executor.close()


def test_multi_models_remote(args):
    test_samples = test_compiler_util.get_allow_samples(args.allow_list)
    
    sample_idx = 0
    failed_samples = []
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    
    for model_path in path_utils.get_recursively_model_path(args.model_path):
        if test_samples is None or os.path.abspath(model_path) in test_samples:
            print(f"[{sample_idx}] {module_name}, model_path: {model_path}", file=sys.stderr)
            
            cmd = " ".join([
                sys.executable,
                f"-m graph_net.torch.{module_name}",
                f"--model-path {model_path}",
                f"--machine {args.machine}",
                f"--port {args.port}",
                f"--compiler {args.compiler}",
                f"--device {args.device}",
                f"--op-lib {args.op_lib}",
                f"--warmup {args.warmup}",
                f"--trials {args.trials}",
                f"--seed {args.seed}",
                f"--reference-dir {args.reference_dir}",
            ])
            
            cmd_ret = os.system(cmd)
            if cmd_ret != 0:
                failed_samples.append(model_path)
            sample_idx += 1
    
    print(f"Totally {sample_idx} verified samples, failed {len(failed_samples)} samples.", file=sys.stderr)
    for model_path in failed_samples:
        print(f"- {model_path}", file=sys.stderr)


def main(args):
    assert os.path.isdir(args.model_path)
    
    ref_dump_dir = Path(args.reference_dir)
    ref_dump_dir.mkdir(parents=True, exist_ok=True)
    
    if path_utils.is_single_model_dir(args.model_path):
        test_single_model_remote(args)
    else:
        test_multi_models_remote(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compiler performance via remote execution.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--compiler", type=str, default="inductor")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--op-lib", type=str, default="default")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reference-dir", type=str, required=True)
    parser.add_argument("--allow-list", type=str, default=None)
    
    parser.add_argument("--machine", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--rpc-cmd", type=str, 
                       default="python3 -m graph_net.torch.test_reference_device")
    
    args = parser.parse_args()
    main(args=args)