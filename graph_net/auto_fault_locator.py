import os
import sys
import subprocess
import argparse
from graph_net.subgraph_decompose_and_evaluation_step import (
    DecomposeConfig,
    convert_json_to_b64_string,
    determine_max_pass_id,
    get_decompose_workspace_path,
)


class AutoFaultLocator:
    def __init__(self, args):
        self.log_file = os.path.abspath(args.log_file)
        self.output_dir = os.path.abspath(args.output_dir)
        self.framework = args.framework
        self.decompose_method = args.decompose_method
        self.max_subgraph_size = str(args.max_subgraph_size)
        self.tolerance = [str(t) for t in args.tolerance]
        self.reference_device = args.reference_device
        self.target_device = args.target_device
        self.machine = args.machine
        self.port = args.port

    def get_one_step_cmd(self, config_str):
        config_b64 = convert_json_to_b64_string(config_str)
        return [
            sys.executable,
            "-m",
            "graph_net.subgraph_decompose_and_evaluation_step",
            "--log-file",
            self.log_file,
            "--output-dir",
            self.output_dir,
            "--framework",
            self.framework,
            "--test-config",
            config_b64,
            "--decompose-method",
            self.decompose_method,
            "--tolerance",
            *self.tolerance,
            "--max-subgraph-size",
            self.max_subgraph_size,
        ]

    def run_remote_test_reference(self):
        print(
            "\n>>> [Step 1] Run Remote Reference Device (Decomposition And Evaluation)\n"
        )

        test_remote_reference_device_config_str = {
            "test_module_name": "test_remote_reference_device",
            "test_remote_reference_device_arguments": {
                "model-path": None,
                "reference-dir": None,
                "compiler": "nope",
                "device": self.reference_device,
                "op-lib": "default",
                "warmup": 5,
                "trials": 20,
                "seed": 123,
                "machine": self.machine,
                "port": self.port,
            },
        }

        cmd = self.get_one_step_cmd(test_remote_reference_device_config_str)
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, text=True)
        assert (
            result.returncode == 0
        ), f"Run Remote Reference Device failed with return code {result.returncode}"

    def run_local_test_target(self):
        print("\n>>> [Step 2] Run Local Target Device (Evaluation And Analysis)\n")

        test_target_device_config_str = {
            "test_module_name": "test_target_device",
            "test_target_device_arguments": {
                "model-path": None,
                "reference-dir": None,
                "device": self.target_device,
            },
        }

        cmd = self.get_one_step_cmd(test_target_device_config_str)
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, text=True)
        assert (
            result.returncode == 0
        ), f"Run Local Target Device failed with return code {result.returncode}"

    def analyze_and_decide_next(self):
        current_pass_id = determine_max_pass_id(self.output_dir)
        current_pass_dir = get_decompose_workspace_path(
            self.output_dir, current_pass_id
        )

        try:
            decompose_config = DecomposeConfig.load(current_pass_dir)
        except Exception as e:
            print(f"[AutoFaultLocator] Error loading config: {e}")
            return False

        if not decompose_config.get_incorrect_models(current_pass_id):
            return False
        if decompose_config.max_subgraph_size <= 1:
            return False
        return True


def main(args):
    locator = AutoFaultLocator(args)
    while True:
        locator.run_remote_test_reference()
        locator.run_local_test_target()
        should_continue = locator.analyze_and_decide_next()
        if not should_continue:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--framework", type=str, choices=["paddle", "torch"], required=True
    )
    parser.add_argument(
        "--decompose-method",
        type=str,
        choices=["uniform", "fixed-start"],
        required=True,
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        nargs="+",
        choices=range(-10, 5),
        help="Tolerance level range [-10, 5)",
    )
    parser.add_argument("--max-subgraph-size", type=int, default=4096)
    parser.add_argument(
        "--reference-device",
        type=str,
        default="cuda",
        required=True,
    )
    parser.add_argument(
        "--target-device",
        type=str,
        default="xpu",
        required=True,
    )
    parser.add_argument("--machine", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50052)

    args = parser.parse_args()
    main(args)
