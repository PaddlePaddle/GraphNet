"""Local runner for in-process model evaluation."""

import json
import os
import sys
import traceback
import types
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

import torch

from .base_runner import BaseRunner, RunResult, RunnerConfig


def _write_log_file(log_path: Path, content: str) -> None:
    """Write log content to file."""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)


def _create_eval_args(
    model_path: str, output_dir: str, config: RunnerConfig
) -> types.SimpleNamespace:
    """Create evaluation arguments from config."""
    return types.SimpleNamespace(
        model_path=model_path,
        output_path=output_dir,
        seed=config.execution.seed,
        compiler=config.execution.compiler,
        device=config.execution.device,
        op_lib=config.execution.op_lib,
        warmup=config.execution.warmup,
        trials=config.execution.trials,
        log_prompt=config.execution.log_prompt,
        backend_config=config.execution.backend_config,
    )


class LocalRunner(BaseRunner):
    """Execute model evaluation in the current process."""

    def run(self, model_path: str, output_dir: str) -> RunResult:
        os.makedirs(output_dir, exist_ok=True)

        log_path = self._get_log_path(output_dir, model_path)
        output_path = self._get_output_path(output_dir, model_path)
        eval_args = _create_eval_args(model_path, output_dir, self.config)

        log_buffer = StringIO()
        result = RunResult(output_path=output_path, log_path=log_path)

        self._execute_with_logging(eval_args, result, log_buffer)
        self._finalize_result(result, log_buffer, log_path)

        return result

    def _execute_with_logging(
        self,
        eval_args: types.SimpleNamespace,
        result: RunResult,
        log_buffer: StringIO,
    ) -> None:
        """Execute evaluation with output redirection."""
        from graph_net_bench.torch import eval_backend_perf

        try:
            eval_backend_perf.register_op_lib(self.config.execution.op_lib)
            eval_backend_perf.set_seed(self.config.execution.seed)
            with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                self._run_evaluation(eval_args, result)
        except Exception as e:
            result.success = False
            result.error_message = f"{str(e)}\n{traceback.format_exc()}"
            log_buffer.write(f"\n[ERROR] {result.error_message}\n")

    def _finalize_result(
        self, result: RunResult, log_buffer: StringIO, log_path: Path
    ) -> None:
        """Finalize result: save log and print to stderr."""
        result.log_content = log_buffer.getvalue()
        _write_log_file(log_path, result.log_content)
        print(result.log_content, file=sys.stderr, flush=True)

    def _run_evaluation(self, args: types.SimpleNamespace, result: RunResult) -> None:
        """Run model evaluation and populate result."""
        from graph_net_bench.torch import eval_backend_perf

        compiler, model, input_dict = self._prepare_model(args)
        self._log_config(args)

        compiled_model = compiler(model)

        def model_call():
            return compiled_model(**input_dict)

        outputs, time_stats = eval_backend_perf.measure_performance(
            model_call, args, compiler
        )

        self._populate_result(result, outputs, time_stats)
        self._log_completion(args, time_stats)

    def _prepare_model(self, args: types.SimpleNamespace) -> tuple:
        """Prepare compiler, model, and inputs."""
        from graph_net_bench.torch import eval_backend_perf

        compiler = eval_backend_perf.get_compiler_backend(args)
        input_dict = eval_backend_perf.get_input_dict(args)
        model = eval_backend_perf.get_model(args)
        model.eval()
        return compiler, model, input_dict

    def _log_config(self, args: types.SimpleNamespace) -> None:
        """Log configuration information."""
        from graph_net_bench.torch import eval_backend_perf
        from graph_net_bench import test_compiler_util

        test_compiler_util.print_config(
            args,
            eval_backend_perf.get_hardware_name(args.device),
            eval_backend_perf.get_compiler_version(args.compiler),
        )

    def _populate_result(
        self, result: RunResult, outputs: Any, time_stats: dict
    ) -> None:
        """Populate result with outputs and stats."""
        result.success = True
        result.outputs = outputs
        result.time_stats = time_stats
        if result.output_path:
            torch.save(outputs, str(result.output_path))

    def _log_completion(self, args: types.SimpleNamespace, time_stats: dict) -> None:
        """Log completion status and performance stats."""
        from graph_net_bench import test_compiler_util

        test_compiler_util.print_running_status(args, True)
        test_compiler_util.print_with_log_prompt(
            "[Performance][eager]:", json.dumps(time_stats), args.log_prompt
        )
