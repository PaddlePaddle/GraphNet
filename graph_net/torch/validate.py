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


def validate(args, model_path):
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
            cmd = f"{sys.executable} -m graph_net.torch.check_redundant_incrementally --model-path {model_path} --graph-net-samples-path {graph_net_samples_path}"
            cmd_ret = os.system(cmd)
            rm_cmd = f"{sys.executable} -m graph_net.torch.remove_redundant_incrementally --model-path {model_path} --graph-net-samples-path {graph_net_samples_path}"
            assert (
                cmd_ret == 0
            ), f"\nPlease use the following command to remove redundant model directories:\n\n{rm_cmd}\n"

        print(f"Validation success, {model_path=}")


def get_recursively_model_path(root_dir):
    for sub_dir in get_immediate_subdirectory_paths(root_dir):
        if is_single_model_dir(sub_dir):
            yield sub_dir
        else:
            yield from get_recursively_model_path(sub_dir)


def get_immediate_subdirectory_paths(parent_dir):
    return [
        sub_dir
        for name in os.listdir(parent_dir)
        for sub_dir in [os.path.join(parent_dir, name)]
        if os.path.isdir(sub_dir)
    ]


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def main(args):
    model_path = args.model_path
    if is_single_model_dir(args.model_path):
        validate(args, model_path)
    else:
        for model_path in get_recursively_model_path(args.model_path):
            validate(args, model_path)


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
