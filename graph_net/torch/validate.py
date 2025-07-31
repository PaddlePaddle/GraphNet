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
        extract_name = "temp"
        print("Check extractability ...")
        cmd = f"{sys.executable} -m graph_net.torch.single_device_runner --model-path {model_path} --enable-extract True --extract-name {extract_name} --dump-graph-hash-key"
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f"{cmd_ret=}, {cmd=}"
        cmd = f"{sys.executable} -m graph_net.torch.single_device_runner --model-path {tmp_dir_name}/{extract_name}"
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f"{cmd_ret=}, {cmd=}"
        if not args.no_check_redundancy:
            print("Check redundancy ...")
            graph_net_samples_path = (
                graph_net.torch.samples_util.get_default_samples_directory()
            )
            cmd = f"{sys.executable} -m graph_net.torch.check_redundant_incrementally --model-path {args.model_path} --graph-net-samples-path {graph_net_samples_path}"
            cmd_ret = os.system(cmd)
            rm_cmd = f"{sys.executable} -m graph_net.torch.remove_redundant_incrementally --model-path {args.model_path} --graph-net-samples-path {graph_net_samples_path}"
            assert (
                cmd_ret == 0
            ), f"\nPlease use the following command to remove redundant model directories:\n\n{rm_cmd}\n"

        print(f"Validation success, {model_path=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to folder e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--no-check-redundancy",
        action="store_true",
        help="whether check model graph redundancy",
    )
    args = parser.parse_args()
    main(args=args)
