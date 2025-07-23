import argparse
import os 
import tempfile
import sys
import contextlib

@contextlib.contextmanager
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        old = os.environ.get('GRAPH_NET_EXTRACT_WORKSPACE')
        os.environ['GRAPH_NET_EXTRACT_WORKSPACE'] = tmp_dir_name
        yield tmp_dir_name
        os.environ['GRAPH_NET_EXTRACT_WORKSPACE'] = old


def main(args):
    model_path = args.model_path
    with temp_workspace() as tmp_dir_name:
        extract_name = "temp"
        cmd = f'{sys.executable} -m graph_net.torch.single_device_runner --model-path {model_path} --enable-extract True --extract-name {extract_name}'
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f'{cmd_ret=}, {cmd=}'
        cmd = f'{sys.executable} -m graph_net.torch.single_device_runner --model-path {tmp_dir_name}/{extract_name}'
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f'{cmd_ret=}, {cmd=}'
        print(f'Validation success, {model_path=}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to folder e.g '../../samples/torch/resnet18'")
    args = parser.parse_args()
    main(args=args)


