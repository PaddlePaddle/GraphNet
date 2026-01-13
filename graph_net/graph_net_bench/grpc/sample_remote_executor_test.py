#!/usr/bin/env python3
"""gRPC Client CLI for SampleRemoteExecutor.

Usage:
    python -m graph_net.graph_net_bench.client --help
"""

import argparse
import sys

from graph_net.graph_net_bench.sample_remote_executor import SampleRemoteExecutor

def main(args):
    parser = argparse.ArgumentParser(
        description="gRPC Client for remote model execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--machine",
        type=str,
        default="localhost",
        help="Remote server address (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50052,
        help="gRPC server port (default: 50052)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory containing model.py and weight_meta.py",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible inference (default: 42)",
    )
    parser.add_argument(
        "--rpc-cmd",
        type=str,
        default="python3 -m graph_net.torch.test_reference_device",
        help="Command to execute on remote server",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output tensors (default: current directory)",
    )


    executor = SampleRemoteExecutor(
        machine=args.machine,
        port=args.port,
        rpc_cmd=args.rpc_cmd,
    )

    try:
        print(f"Sending request to {args.machine}:{args.port}...", file=sys.stderr)
        tensors = executor(args.model_path, args.random_seed)

        print(f"Received {len(tensors)} output tensors:", file=sys.stderr)
        for i, tensor in enumerate(tensors):
            print(f"  output_{i}: shape={tensor.shape}, dtype={tensor.dtype}", file=sys.stderr)

        if args.output_dir:
            import os
            from pathlib import Path
            import torch

            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for i, tensor in enumerate(tensors):
                output_file = output_path / f"output_{i}.pt"
                torch.save(tensor, output_file)
                print(f"Saved output_{i} to {output_file}", file=sys.stderr)

        print("Execution completed successfully!", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        executor.close()


if __name__ == "__main__":
    main()