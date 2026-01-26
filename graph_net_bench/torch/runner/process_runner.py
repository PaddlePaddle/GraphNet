import os
import sys
import subprocess
from pathlib import Path

import torch

from .base_runner import BaseRunner, RunResult


class ProcessRunner(BaseRunner):
    """Execute model evaluation in a separate subprocess on the local machine."""

    def run(self, model_path: str, output_dir: str) -> RunResult:
        os.makedirs(output_dir, exist_ok=True)

        log_path = self._get_log_path(output_dir, model_path)
        output_path = self._get_output_path(output_dir, model_path)

        result = RunResult(
            output_path=output_path,
            log_path=log_path,
        )

        cmd = self._build_command(model_path, output_dir)
        print(f"[ProcessRunner] Executing: {cmd}", file=sys.stderr, flush=True)

        try:
            env = os.environ.copy()
            repo_root = Path(__file__).resolve().parents[3]
            env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

            proc = subprocess.run(
                cmd,
                shell=True,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.strategy.subprocess_timeout,
            )

            result.log_content = proc.stderr or ""

            if proc.returncode != 0:
                result.success = False
                result.error_message = (
                    f"Process exited with code {proc.returncode}\n"
                    f"stdout: {proc.stdout}\n"
                    f"stderr: {proc.stderr}"
                )
            else:
                result.success = True
                self._parse_result(result, output_dir, model_path)

        except subprocess.TimeoutExpired as e:
            result.success = False
            result.error_message = f"Process timed out: {e}"
        except Exception as e:
            result.success = False
            result.error_message = f"Process execution failed: {e}"

        print(result.log_content, file=sys.stderr, flush=True)
        return result

    def _build_command(self, model_path: str, output_dir: str) -> str:
        cmd_parts = [
            sys.executable,
            "-m",
            "graph_net_bench.torch.eval_backend_perf",
            "--model-path",
            model_path,
            "--output-path",
            output_dir,
        ]

        config_dict = self.config.to_dict()
        from graph_net_bench import test_compiler_util

        config_str = test_compiler_util.convert_to_base64(config_dict)
        cmd_parts.extend(["--config", config_str])

        return " ".join(cmd_parts)

    def _parse_result(self, result: RunResult, output_dir: str, model_path: str):
        from graph_net_bench import test_compiler_util

        if result.output_path and result.output_path.exists():
            try:
                result.outputs = torch.load(str(result.output_path))
            except Exception as e:
                result.error_message += f"\nFailed to load outputs: {e}"

        if result.log_path and result.log_path.exists():
            try:
                result.log_content = test_compiler_util.extract_log_content(
                    str(result.log_path)
                )
                result.time_stats = test_compiler_util.parse_performance_stats(
                    str(result.log_path)
                )
            except Exception as e:
                result.error_message += f"\nFailed to parse log: {e}"
