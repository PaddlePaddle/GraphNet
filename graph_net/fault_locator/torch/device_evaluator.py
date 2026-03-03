import sys
import subprocess
import time
from pathlib import Path
from graph_net.declare_config_mixin import DeclareConfigMixin


class DeviceEvaluator(DeclareConfigMixin):
    """
    Evaluator responsible for comparing model performance and accuracy between
    a reference device (e.g., CPU) and a target device (e.g., CUDA).
    Uses 'default' as the operator library for all target executions.
    """

    def __init__(self, config=None):
        self.init_config(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        ref_device: str = "cpu",
        target_device: str = "cuda",
        compiler: str = "nope",
    ):
        """
        Configuration schema for cross-device benchmarking.
        """
        pass

    def __call__(self, rel_model_path: str) -> str:
        """
        Orchestrates the evaluation pipeline:
        1. Generates ground truth data on the reference device.
        2. Validates performance/accuracy on the target device.
        """
        output_path = Path(self.config["output_dir"])
        full_model_path = Path(self.config["model_path_prefix"]) / rel_model_path

        # Define specific workspace for target device logs
        workspace = output_path / self.config["target_device"] / rel_model_path
        workspace.mkdir(parents=True, exist_ok=True)

        # Directory for sharing ground truth data between runs
        reference_dir = output_path / "reference_data"
        reference_dir.mkdir(parents=True, exist_ok=True)

        log_file = workspace / "validation.log"

        # Step 1: Execute reference test to establish baseline
        print(f"Generating reference data on: {self.config['ref_device']}")
        self._run_reference_test(full_model_path, reference_dir)

        # Step 2: Execute target test and return captured logs
        print(f"Running target evaluation on: {self.config['target_device']}")
        return self._run_target_test(full_model_path, reference_dir, log_file)

    def _run_reference_test(self, full_model_path: Path, reference_dir: Path):
        """
        Invokes the reference module to generate expected outputs (Ground Truth).
        """
        cmd = [
            sys.executable,
            "-m",
            "graph_net.torch.test_reference_device",
            "--model-path",
            str(full_model_path),
            "--reference-dir",
            str(reference_dir),
            "--compiler",
            self.config["compiler"],
            "--device",
            self.config["ref_device"],
        ]
        # Reference runs are silent; errors will raise a CalledProcessError
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _run_target_test(
        self, full_model_path: Path, reference_dir: Path, log_file: Path
    ) -> str:
        """
        Executes the model on the target device using 'default' op_lib
        and captures the full output log.
        """
        cmd = [
            sys.executable,
            "-m",
            "graph_net.torch.test_target_device",
            "--model-path",
            str(full_model_path),
            "--reference-dir",
            str(reference_dir),
            "--device",
            self.config["target_device"],
            "--op-lib",
            "default",
        ]

        print(" ".join(cmd))
        # Redirect all output to the log file for persistence and analysis
        with log_file.open("w") as f:
            start_time = time.perf_counter()
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            end_time = time.perf_counter()
            print(f"Target execution completed in {end_time - start_time:.4f} seconds")

        return log_file.read_text()
