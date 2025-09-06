import argparse
import os
import tempfile
import sys
import contextlib
import graph_net


@contextlib.contextmanager
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        old = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = tmp_dir_name
        yield tmp_dir_name
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = old


def main(args):
    model_path = args.model_path
    with temp_workspace() as tmp_dir_name:
        print("Check extractability ...")
        cmd = f"{sys.executable} -m graph_net.torch.single_device_runner --model-path {model_path}"
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f"{cmd_ret=}, {cmd=}"
        extract_name = "temp"
        cmd = f"{sys.executable} -m graph_net.torch.single_device_runner --model-path {model_path} --enable-extract True --extract-name {extract_name} --dump-graph-hash-key"
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f"{cmd_ret=}, {cmd=}"
        cmd = f"{sys.executable} -m graph_net.torch.single_device_runner --model-path {tmp_dir_name}/{extract_name}"
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f"{cmd_ret=}, {cmd=}"
        if not args.no_check_redundancy:
            print("Check redundancy ...")
            graph_net_samples_path = (
                (graph_net.torch.samples_util.get_default_samples_directory())
                if args.graph_net_samples_path is None
                else args.graph_net_samples_path
            )
            cmd = f"{sys.executable} -m graph_net.torch.check_redundant_incrementally --model-path {args.model_path} --graph-net-samples-path {graph_net_samples_path}"
            cmd_ret = os.system(cmd)
            rm_cmd = f"{sys.executable} -m graph_net.torch.remove_redundant_incrementally --model-path {args.model_path} --graph-net-samples-path {graph_net_samples_path}"
            assert (
                cmd_ret == 0
            ), f"\nPlease use the following command to remove redundant model directories:\n\n{rm_cmd}\n"

        print(f"Validation success, {model_path=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a computation graph sample. return 0 if success"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Computation graph sample directory. e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--graph-net-samples-path",
        type=str,
        required=False,
        default=None,
        help="GraphNet samples directory. used for redundancy check. e.g '../../samples'",
    )
    parser.add_argument(
        "--no-check-redundancy",
        action="store_true",
        help="Diable redundancy check (default: False).",
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "./workspace"),
        help="temporary directory for validation (default: env var GRAPH_NET_EXTRACT_WORKSPACE). ",
    )
    args = parser.parse_args()
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.workspace

    main(args=args)
