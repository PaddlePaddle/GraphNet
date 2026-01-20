import os
import sys
from typing import Dict

import torch

from .base_runner import BaseRunner, RunResult


class RemoteRunner(BaseRunner):
    """Execute model evaluation on a remote machine via gRPC."""

    def run(self, model_path: str, output_dir: str) -> RunResult:
        from graph_net_rpc.sample_remote_executor import SampleRemoteExecutor

        os.makedirs(output_dir, exist_ok=True)

        log_path = self._get_log_path(output_dir, model_path)
        output_path = self._get_output_path(output_dir, model_path)

        result = RunResult(
            output_path=output_path,
            log_path=log_path,
        )

        rpc_cmd = self._build_rpc_command()
        executor = SampleRemoteExecutor(
            machine=self.config.strategy.remote_machine,
            port=self.config.strategy.remote_port,
        )

        try:
            print(
                f"[RemoteRunner] Sending to {self.config.machine}:{self.config.port}",
                file=sys.stderr,
                flush=True,
            )
            print(f"[RemoteRunner] rpc_cmd: {rpc_cmd}", file=sys.stderr, flush=True)

            files_dict = executor.execute(model_path, rpc_cmd)
            self._process_remote_output(result, files_dict, output_dir, model_path)
            result.success = True

        except Exception as e:
            import traceback

            result.success = False
            result.error_message = (
                f"Remote execution failed: {e}\n{traceback.format_exc()}"
            )
            print(result.error_message, file=sys.stderr, flush=True)

        finally:
            executor.close()

        return result

    def _build_rpc_command(self) -> str:
        cmd = "python3 -m graph_net.torch.test_reference_device"
        cmd += ' --model-path "$INPUT_WORKSPACE"'
        cmd += ' --reference-dir "$OUTPUT_WORKSPACE"'
        cmd += f" --compiler {self.config.execution.compiler}"
        cmd += f" --device {self.config.execution.device}"
        cmd += f" --op-lib {self.config.execution.op_lib}"
        cmd += f" --warmup {self.config.execution.warmup}"
        cmd += f" --trials {self.config.execution.trials}"
        cmd += f" --seed {self.config.execution.seed}"

        if self.config.execution.log_prompt:
            cmd += f" --log-prompt {self.config.execution.log_prompt}"
        if self.config.execution.backend_config:
            cmd += f" --config {self.config.execution.backend_config}"

        return cmd

    def _process_remote_output(
        self,
        result: RunResult,
        files_dict: Dict[str, bytes],
        output_dir: str,
        model_path: str,
    ):
        from graph_net_bench import test_compiler_util

        log_filename = result.log_path.name if result.log_path else None
        pth_filename = result.output_path.name if result.output_path else None

        available_logs = sorted([k for k in files_dict.keys() if k.endswith(".log")])
        available_pths = sorted([k for k in files_dict.keys() if k.endswith(".pth")])

        if log_filename not in files_dict and len(available_logs) == 1:
            log_filename = available_logs[0]
        if pth_filename not in files_dict and len(available_pths) == 1:
            pth_filename = available_pths[0]

        if log_filename and log_filename in files_dict:
            log_bytes = files_dict[log_filename]
            if result.log_path:
                with open(result.log_path, "wb") as f:
                    f.write(log_bytes)
            try:
                result.log_content = log_bytes.decode("utf-8")
                print(result.log_content, file=sys.stderr, flush=True)
            except Exception:
                result.log_content = f"[Binary log, {len(log_bytes)} bytes]"
                # Write binary content as text for parsing
                with open(result.log_path, "wb") as f:
                    f.write(log_bytes)

            try:
                result.time_stats = test_compiler_util.parse_performance_stats(
                    str(result.log_path)
                )
            except Exception as e:
                print(f"Warning: Failed to parse time stats: {e}", file=sys.stderr)
        else:
            print(
                f"Warning: log not found. expected={log_filename}, available={available_logs}",
                file=sys.stderr,
            )

        if pth_filename and pth_filename in files_dict:
            pth_bytes = files_dict[pth_filename]
            if result.output_path:
                with open(result.output_path, "wb") as f:
                    f.write(pth_bytes)
            try:
                result.outputs = torch.load(str(result.output_path))
            except Exception as e:
                print(f"Warning: Failed to load outputs: {e}", file=sys.stderr)
        else:
            print(
                f"Warning: output not found. expected={pth_filename}, available={available_pths}",
                file=sys.stderr,
            )
