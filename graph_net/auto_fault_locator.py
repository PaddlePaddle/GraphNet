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

    def execute_one_step_cmd(self, test_config):
        test_config_b64_str = convert_json_to_b64_string(test_config)
        cmd = [
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
            test_config_b64_str,
            "--decompose-method",
            self.decompose_method,
            "--tolerance",
            *self.tolerance,
            "--max-subgraph-size",
            self.max_subgraph_size,
        ]

        print(f"[AutoFaultLocator] Executing: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, check=True, text=True)
        return result

    def run_test_reference_device(self, is_remote):
        print(
            "\n>>> [AutoFaultLocator 2/1] Run Test Reference Device (Decomposition And Evaluation)\n",
            flush=True,
        )

        test_module_name = (
            "test_remote_reference_device" if is_remote else "test_reference_device"
        )
        test_reference_device_config = {
            "test_module_name": test_module_name,
            f"{test_module_name}_arguments": {
                "model-path": None,
                "reference-dir": None,
                "compiler": "nope",
                "device": self.reference_device,
                "warmup": 5,
                "trials": 20,
                "seed": 123,
            },
        }
        if args.framework == "torch":
            test_reference_device_config[f"{test_module_name}_arguments"].update(
                {"op-lib": "default"}
            )
        if is_remote:
            test_reference_device_config[f"{test_module_name}_arguments"].update(
                {
                    "machine": self.machine,
                    "port": self.port,
                }
            )

        result = self.execute_one_step_cmd(test_reference_device_config)
        assert (
            result.returncode == 0
        ), f"Run Remote Reference Device failed with return code {result.returncode}"

    def run_test_target_device(self, is_remote):
        print(
            "\n>>> [AutoFaultLocator 2/2] Run Test Target Device (Evaluation And Analysis)\n",
            flush=True,
        )

        test_module_name = (
            "test_remote_target_device" if is_remote else "test_target_device"
        )
        test_target_device_config = {
            "test_module_name": test_module_name,
            f"{test_module_name}_arguments": {
                "model-path": None,
                "reference-dir": None,
                "compiler": "nope",
                "device": self.target_device,
                "warmup": 5,
                "trials": 20,
                "seed": 123,
            },
        }
        if is_remote:
            test_target_device_config[f"{test_module_name}_arguments"].update(
                {
                    "machine": self.machine,
                    "port": self.port,
                }
            )

        result = self.execute_one_step_cmd(test_target_device_config)
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
        locator.run_test_reference_device(is_remote=False)
        locator.run_test_target_device(is_remote=True)
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
