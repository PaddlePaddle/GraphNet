import argparse
import traceback
from graph_net.imp_util import load_module
import logging
import sys
import json
import subprocess

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _load_class_from_file(sample_pass_file_path, sample_pass_class_name):
    module = load_module(sample_pass_file_path)
    return getattr(module, sample_pass_class_name)


def _get_handler(args):
    handler_class = _load_class_from_file(
        args.sample_pass_file_path, args.sample_pass_class_name
    )

    sample_pass_config = {}
    if args.sample_pass_config:
        try:
            sample_pass_config = json.loads(args.sample_pass_config)
        except json.JSONDecodeError:
            print(
                f"Error: Failed to parse --sample_pass_config as JSON. Received: {args.sample_pass_config}"
            )
            sys.exit(1)

    return handler_class(sample_pass_config)


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
        except Exception:
            print("------------[apply_sample_pass failed]------------", flush=True)
            traceback.print_exc()


def handle_model_path_list_in_subprocess(args):
    for model_path in _get_model_path_list(args):
        cmd = (
            f"{sys.executable} -m graph_net.apply_sample_pass "
            f"--sample-pass-file-path {args.sample_pass_file_path} "
            f"--sample-pass-class-name {args.sample_pass_class_name} "
            f"--sample-pass-config '{args.sample_pass_config}' "
            f"--model-path {model_path}"
        )
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
    parser = argparse.ArgumentParser(description="apply sample pass entry")
    parser.add_argument(
        "--sample-pass-file-path",
        type=str,
        required=True,
        help="Path to the python file containing the handler class.",
    )
    parser.add_argument(
        "--sample-pass-class-name",
        type=str,
        required=True,
        help="Name of the handler class to instantiate.",
    )
    parser.add_argument(
        "--sample-pass-config",
        type=str,
        required=False,
        default="{}",
        help="JSON string for handler sample_pass_configuration.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=None,
        help="Path to single model folder.",
    )
    parser.add_argument(
        "--model-path-list",
        type=str,
        required=False,
        default=None,
        help="Path of file containing model paths.",
    )
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        default=False,
        help="Execute each model handling in a separate subprocess.",
    )

    args = parser.parse_args()
    main(args=args)
