"""Remote runner for gRPC-based model evaluation."""

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import torch

from .base_runner import BaseRunner, RunResult


def _find_file_by_extension(
    files_dict: Dict[str, bytes], expected_name: Optional[str], extension: str
) -> Optional[str]:
    """Find file in dict by expected name or by extension if only one exists."""
    if expected_name and expected_name in files_dict:
        return expected_name
    available = sorted(k for k in files_dict.keys() if k.endswith(extension))
    if len(available) == 1:
        return available[0]
    return None


def _save_bytes_to_file(path: Path, content: bytes) -> None:
    """Save bytes content to file."""
    with open(path, "wb") as f:
        f.write(content)


class RemoteRunner(BaseRunner):
    """Execute model evaluation on a remote machine via gRPC."""

    def run(self, model_path: str, output_dir: str) -> RunResult:
        os.makedirs(output_dir, exist_ok=True)

        result = RunResult(
            output_path=self._get_output_path(output_dir, model_path),
            log_path=self._get_log_path(output_dir, model_path),
        )

        self._execute_remote(model_path, result)
        return result

    def _execute_remote(self, model_path: str, result: RunResult) -> None:
        """Execute model on remote machine."""
        from graph_net_rpc.sample_remote_executor import SampleRemoteExecutor

        executor = SampleRemoteExecutor(
            machine=self.config.strategy.remote_machine,
            port=self.config.strategy.remote_port,
        )

        try:
            self._log_execution_start()
            rpc_cmd = self._build_rpc_command()
            print(f"[RemoteRunner] rpc_cmd: {rpc_cmd}", file=sys.stderr, flush=True)

            files_dict = executor.execute(model_path, rpc_cmd)
            self._process_remote_output(result, files_dict)
            result.success = True
        except Exception as e:
            result.success = False
            result.error_message = (
                f"Remote execution failed: {e}\n{traceback.format_exc()}"
            )
            print(result.error_message, file=sys.stderr, flush=True)
        finally:
            executor.close()

    def _log_execution_start(self) -> None:
        """Log remote execution start."""
        machine = self.config.strategy.remote_machine
        port = self.config.strategy.remote_port
        print(
            f"[RemoteRunner] Sending to {machine}:{port}", file=sys.stderr, flush=True
        )

    def _build_rpc_command(self) -> str:
        """Build remote execution command string."""
        exec_cfg = self.config.execution
        cmd_parts = [
            "python3 -m graph_net.torch.test_reference_device",
            '--model-path "$INPUT_WORKSPACE"',
            '--reference-dir "$OUTPUT_WORKSPACE"',
            f"--compiler {exec_cfg.compiler}",
            f"--device {exec_cfg.device}",
            f"--op-lib {exec_cfg.op_lib}",
            f"--warmup {exec_cfg.warmup}",
            f"--trials {exec_cfg.trials}",
            f"--seed {exec_cfg.seed}",
        ]
        if exec_cfg.log_prompt:
            cmd_parts.append(f"--log-prompt {exec_cfg.log_prompt}")
        if exec_cfg.backend_config:
            cmd_parts.append(f"--config {exec_cfg.backend_config}")
        return " ".join(cmd_parts)

    def _process_remote_output(
        self, result: RunResult, files_dict: Dict[str, bytes]
    ) -> None:
        """Process files received from remote execution."""
        self._process_log_file(result, files_dict)
        self._process_output_file(result, files_dict)

    def _process_log_file(
        self, result: RunResult, files_dict: Dict[str, bytes]
    ) -> None:
        """Process log file from remote output."""
        expected_name = result.log_path.name if result.log_path else None
        log_filename = _find_file_by_extension(files_dict, expected_name, ".log")

        if not log_filename:
            available = [k for k in files_dict.keys() if k.endswith(".log")]
            print(
                f"Warning: log not found. expected={expected_name}, available={available}",
                file=sys.stderr,
            )
            return

        log_bytes = files_dict[log_filename]
        self._save_and_parse_log(result, log_bytes)

    def _save_and_parse_log(self, result: RunResult, log_bytes: bytes) -> None:
        """Save log file and parse timing stats."""

        if result.log_path:
            _save_bytes_to_file(result.log_path, log_bytes)

        result.log_content = self._decode_log_content(log_bytes)
        print(result.log_content, file=sys.stderr, flush=True)

        self._parse_time_stats(result)

    def _decode_log_content(self, log_bytes: bytes) -> str:
        """Decode log bytes to string."""
        try:
            return log_bytes.decode("utf-8")
        except Exception:
            return f"[Binary log, {len(log_bytes)} bytes]"

    def _parse_time_stats(self, result: RunResult) -> None:
        """Parse performance stats from log file."""
        if not result.log_path:
            return
        from graph_net_bench import test_compiler_util

        try:
            result.time_stats = test_compiler_util.parse_performance_stats(
                str(result.log_path)
            )
        except Exception as e:
            print(f"Warning: Failed to parse time stats: {e}", file=sys.stderr)

    def _process_output_file(
        self, result: RunResult, files_dict: Dict[str, bytes]
    ) -> None:
        """Process output .pth file from remote output."""
        expected_name = result.output_path.name if result.output_path else None
        pth_filename = _find_file_by_extension(files_dict, expected_name, ".pth")

        if not pth_filename:
            available = [k for k in files_dict.keys() if k.endswith(".pth")]
            print(
                f"Warning: output not found. expected={expected_name}, available={available}",
                file=sys.stderr,
            )
            return

        pth_bytes = files_dict[pth_filename]
        self._save_and_load_outputs(result, pth_bytes)

    def _save_and_load_outputs(self, result: RunResult, pth_bytes: bytes) -> None:
        """Save output file and load tensors."""
        if result.output_path:
            _save_bytes_to_file(result.output_path, pth_bytes)
        try:
            result.outputs = torch.load(str(result.output_path))
        except Exception as e:
            print(f"Warning: Failed to load outputs: {e}", file=sys.stderr)
