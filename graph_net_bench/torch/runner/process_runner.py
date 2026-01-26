"""Process runner for subprocess-based model evaluation."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import torch

from .base_runner import BaseRunner, RunResult


def _get_env_with_pythonpath() -> Dict[str, str]:
    """Get environment with PYTHONPATH set to repo root."""
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    return env


class ProcessRunner(BaseRunner):
    """Execute model evaluation in a separate subprocess on the local machine."""

    def run(self, model_path: str, output_dir: str) -> RunResult:
        os.makedirs(output_dir, exist_ok=True)

        result = RunResult(
            output_path=self._get_output_path(output_dir, model_path),
            log_path=self._get_log_path(output_dir, model_path),
        )

        cmd = self._build_command(model_path, output_dir)
        print(f"[ProcessRunner] Executing: {cmd}", file=sys.stderr, flush=True)

        self._execute_subprocess(cmd, result, output_dir, model_path)
        print(result.log_content, file=sys.stderr, flush=True)

        return result

    def _execute_subprocess(
        self, cmd: str, result: RunResult, output_dir: str, model_path: str
    ) -> None:
        """Execute subprocess and handle results."""
        try:
            proc = self._run_process(cmd)
            result.log_content = proc.stderr or ""
            self._handle_process_result(proc, result, output_dir, model_path)
        except subprocess.TimeoutExpired as e:
            result.success = False
            result.error_message = f"Process timed out: {e}"
        except Exception as e:
            result.success = False
            result.error_message = f"Process execution failed: {e}"

    def _run_process(self, cmd: str) -> subprocess.CompletedProcess:
        """Run subprocess with configured timeout."""
        return subprocess.run(
            cmd,
            shell=True,
            env=_get_env_with_pythonpath(),
            capture_output=True,
            text=True,
            timeout=self.config.strategy.subprocess_timeout,
        )

    def _handle_process_result(
        self,
        proc: subprocess.CompletedProcess,
        result: RunResult,
        output_dir: str,
        model_path: str,
    ) -> None:
        """Handle subprocess completion result."""
        if proc.returncode != 0:
            result.success = False
            result.error_message = (
                f"Process exited with code {proc.returncode}\n"
                f"stdout: {proc.stdout}\n"
                f"stderr: {proc.stderr}"
            )
            return
        result.success = True
        self._parse_result(result, output_dir, model_path)

    def _build_command(self, model_path: str, output_dir: str) -> str:
        """Build subprocess command string."""
        from graph_net_bench import test_compiler_util

        config_str = test_compiler_util.convert_to_base64(self.config.to_dict())
        cmd_parts = [
            sys.executable,
            "-m",
            "graph_net_bench.torch.eval_backend_perf",
            "--model-path",
            model_path,
            "--output-path",
            output_dir,
            "--config",
            config_str,
        ]
        return " ".join(cmd_parts)

    def _parse_result(
        self, result: RunResult, output_dir: str, model_path: str
    ) -> None:
        """Parse outputs and logs from subprocess result."""
        self._load_outputs(result)
        self._parse_log(result)

    def _load_outputs(self, result: RunResult) -> None:
        """Load model outputs from file."""
        if not result.output_path or not result.output_path.exists():
            return
        try:
            result.outputs = torch.load(str(result.output_path))
        except Exception as e:
            result.error_message += f"\nFailed to load outputs: {e}"

    def _parse_log(self, result: RunResult) -> None:
        """Parse log file for content and timing stats."""
        if not result.log_path or not result.log_path.exists():
            return
        from graph_net_bench import test_compiler_util

        try:
            log_path_str = str(result.log_path)
            result.log_content = test_compiler_util.extract_log_content(log_path_str)
            result.time_stats = test_compiler_util.parse_performance_stats(log_path_str)
        except Exception as e:
            result.error_message += f"\nFailed to parse log: {e}"
