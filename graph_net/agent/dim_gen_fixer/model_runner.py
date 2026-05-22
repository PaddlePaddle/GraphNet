import os
import sys
import subprocess


class ModelRunner:
    """Execute graph_net.torch.run_model and capture output."""

    def __init__(self, timeout: int, gpu_id: int | None = None):
        self.timeout = timeout
        self.gpu_id = gpu_id

    def run(self, sample_path: str) -> dict:
        """
        Execute run_model.

        Returns:
            {
                "success": bool,
                "returncode": int,
                "output": str,      # Full output (stdout + stderr)
            }
        """
        cmd = [
            sys.executable,
            "-m",
            "graph_net.torch.run_model",
            "--model-path",
            sample_path,
        ]
        env = os.environ.copy()
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self._get_project_root(),
                env=env,
            )
            output = proc.stdout + proc.stderr
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "output": output,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "output": f"TIMEOUT: run_model exceeded {self.timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "output": f"ERROR: {e}",
            }

    @staticmethod
    def _get_project_root() -> str:
        """Get the project root directory."""
        # Script is in graph_net/agent/dim_gen_fixer/, root is its grand-grand-parent directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
