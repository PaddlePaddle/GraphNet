import sys
import subprocess
import time
from pathlib import Path
from graph_net.declare_config_mixin import DeclareConfigMixin


class OpLibEvaluator(DeclareConfigMixin):
    """
    Functor responsible for evaluating model samples by comparing a target operator
    library's (e.g., FlagGems) performance and accuracy against a reference implementation.
    The evaluator manages reference data generation and captures execution logs.
    """

    def __init__(self, config=None):
        self.init_config(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        op_lib: str,
        device: str = "cuda",
        compiler: str = "nope",
    ):
        """
        Configuration schema for operator library benchmarking.
        The reference_data directory is automatically managed within the output_dir.
        """
        pass

    def __call__(self, rel_model_path: str) -> str:
        """
        Orchestrates the pipeline for reference data generation and target library testing.
        Returns:
            The complete log content from the target device test execution.
        """
        output_path = Path(self.config["output_dir"])
        # Create an isolated workspace for the current model sample
        workspace = output_path / rel_model_path
        workspace.mkdir(parents=True, exist_ok=True)

        # Determine the shared directory for reference ground truth
        reference_dir = output_path / "reference_data"
        reference_dir.mkdir(parents=True, exist_ok=True)

        # Construct the absolute model path
        full_model_path = Path(self.config["model_path_prefix"]) / rel_model_path
        log_file = workspace / "op_lib_validation.log"

        # 1. Execute reference test to establish baseline metrics
        self._run_reference_test(full_model_path, reference_dir)

        # 2. Execute target library test and capture performance/accuracy logs
        return self._run_target_test(full_model_path, reference_dir, log_file)

    def _run_reference_test(self, full_model_path: Path, reference_dir: Path):
        """
        Invokes the reference device test module to generate ground truth data.
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
            self.config["device"],
        ]
        # Reference tests are executed synchronously; output is captured but not returned
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _run_target_test(
        self, full_model_path: Path, reference_dir: Path, log_file: Path
    ) -> str:
        """
        Invokes the target device test module for the specified op_lib and merges
        stdout/stderr into the local log file.
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
            self.config["device"],
            "--op-lib",
            self.config["op_lib"],
        ]

        print(" ".join(cmd))
        # Redirect all output to the log file for persistence and analysis
        with log_file.open("w") as f:
            starttime = time.time()
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            endtime = time.time()
            print("run_target_test running time {:.5f} s".format(endtime - starttime))

        return log_file.read_text()
