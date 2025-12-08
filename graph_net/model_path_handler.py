import argparse
from graph_net.imp_util import load_module
import logging
import sys
import json
import base64
import subprocess

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _convert_to_dict(config_str):
    if config_str is None:
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


def _load_class_from_file(file_path, class_name):
    module = load_module(file_path)
    return getattr(module, class_name)


def _get_handler(args):
    if args.handler_config is None:
        return lambda model_path: model_path
    handler_config = _convert_to_dict(args.handler_config)
    handler_class = _load_class_from_file(
        handler_config["handler_path"], class_name=handler_config["handler_class_name"]
    )
    return handler_class(handler_config.get("handler_config", {}))


def main(args):
    handler = _get_handler(args)
    if args.model_path is not None:
        handle_model_path(handler, args.model_path)
    elif args.use_subprocess:
        handle_model_path_list_in_subprocess(args)
    else:
        handle_model_path_list_in_current_process(handler, args)


def handle_model_path_list_in_current_process(handler, args):
    for model_path in _get_model_path_list(args):
        try:
            handle_model_path(handler, model_path)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            return


def handle_model_path_list_in_subprocess(args):
    for model_path in _get_model_path_list(args):
        cmd = f"{sys.executable} -m graph_net.model_path_handler --model-path {model_path} --handler-config {args.handler_config}"
        try:
            subprocess.Popen(cmd, shell=True).wait()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            return


def handle_model_path(handler, model_path):
    print(f"{model_path=}", flush=True)
    handler(model_path)


def _get_model_path_list(args):
    assert args.model_path is None
    assert args.model_path_list is not None
    with open(args.model_path_list) as f:
        yield from (
            clean_line
            for line in f
            for clean_line in [line.strip()]
            if len(clean_line) > 0
            if not clean_line.startswith("#")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model path handler entry")
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=None,
        help="Path to folder e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--model-path-list",
        type=str,
        required=False,
        default=None,
        help="Path of file containing model paths.",
    )
    parser.add_argument(
        "--handler-config",
        type=str,
        required=False,
        default=None,
        help="handler configuration string",
    )
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        default=False,
        help="use subprocess",
    )
    args = parser.parse_args()
    main(args=args)
