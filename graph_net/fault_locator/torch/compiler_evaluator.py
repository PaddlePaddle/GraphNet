import sys
import subprocess
from pathlib import Path
from graph_net.declare_config_mixin import DeclareConfigMixin


class CompilerEvaluator(DeclareConfigMixin):
    """
    Functor responsible for evaluating a model sample using the Torch benchmarking suite.
    It executes external compilation tests and parses the resulting Error Scores (ES).
    """

    def __init__(self, config=None):
        self.init_config(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        compiler: str = "inductor",
        device: str = "cuda",
    ):
        """
        Configuration schema for the Torch benchmark environment and hardware settings.
        """
        pass

    def __call__(self, rel_model_path: str) -> str:
        """
        Returns:
            A dictionary mapping layer indices (int) to their respective ES scores (float).
        """
        tmp_path = Path(self.config["output_dir"])
        tmp_path.mkdir(parents=True, exist_ok=True)

        # 1. Setup intermediate file paths within the transient workspace
        allow_list = self._prepare_workspace(tmp_path, rel_model_path)
        log_file = tmp_path / "validation.log"

        # 2. Execute the sequential stages of the evaluation process
        self._execute_benchmark(allow_list, log_file)
        return Path(log_file).read_text()

    def _prepare_workspace(self, tmp_dir: Path, rel_model_path: str) -> Path:
        """
        Generates the temporary allow-list file required by the torch compiler tester.
        """
        allow_list_path = tmp_dir / "allow_list.txt"
        allow_list_path.write_text(rel_model_path)
        return allow_list_path

    def _execute_benchmark(self, allow_list_path: Path, log_file: Path):
        """
        Invokes the torch.test_compiler module and redirects output to a log file.
        Uses sys.executable to ensure the same Python environment is used.
        """
        cmd = [
            sys.executable,
            "-m",
            "graph_net_bench.torch.test_compiler",
            "--model-path-prefix",
            f"{self.config['model_path_prefix']}/",
            "--allow-list",
            str(allow_list_path),
            "--compiler",
            self.config["compiler"],
            "--device",
            self.config["device"],
        ]
        print(" ".join(cmd))
        with log_file.open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"{open(log_file).read()}")
