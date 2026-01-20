import os
import sys
import json
import types
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

import torch

from .base_runner import BaseRunner, RunResult


class LocalRunner(BaseRunner):
    """Execute model evaluation in the current process."""

    def run(self, model_path: str, output_dir: str) -> RunResult:
        from graph_net_bench.torch import eval_backend_perf

        os.makedirs(output_dir, exist_ok=True)

        log_path = self._get_log_path(output_dir, model_path)
        output_path = self._get_output_path(output_dir, model_path)

        eval_args = types.SimpleNamespace(
            model_path=model_path,
            output_path=output_dir,
            seed=self.config.execution.seed,
            compiler=self.config.execution.compiler,
            device=self.config.execution.device,
            op_lib=self.config.execution.op_lib,
            warmup=self.config.execution.warmup,
            trials=self.config.execution.trials,
            log_prompt=self.config.execution.log_prompt,
            backend_config=self.config.execution.backend_config,
        )

        log_buffer = StringIO()
        result = RunResult(
            output_path=output_path,
            log_path=log_path,
        )

        try:
            eval_backend_perf.register_op_lib(self.config.execution.op_lib)
            eval_backend_perf.set_seed(self.config.execution.seed)

            with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
                self._run_evaluation(eval_args, result)

        except Exception as e:
            result.success = False
            result.error_message = f"{str(e)}\n{traceback.format_exc()}"
            log_buffer.write(f"\n[ERROR] {result.error_message}\n")

        result.log_content = log_buffer.getvalue()

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(result.log_content)

        print(result.log_content, file=sys.stderr, flush=True)

        return result

    def _run_evaluation(self, args: types.SimpleNamespace, result: RunResult):
        from graph_net_bench.torch import eval_backend_perf
        from graph_net_bench import test_compiler_util

        compiler = eval_backend_perf.get_compiler_backend(args)
        input_dict = eval_backend_perf.get_input_dict(args)
        model = eval_backend_perf.get_model(args)
        model.eval()

        test_compiler_util.print_config(
            args,
            eval_backend_perf.get_hardward_name(args.device),
            eval_backend_perf.get_compiler_version(args.compiler),
        )

        compiled_model = compiler(model)

        def model_call():
            return compiled_model(**input_dict)

        outputs, time_stats = eval_backend_perf.measure_performance(
            model_call, args, compiler
        )

        result.success = True
        result.outputs = outputs
        result.time_stats = time_stats

        if result.output_path:
            torch.save(outputs, str(result.output_path))

        test_compiler_util.print_running_status(args, True)
        test_compiler_util.print_with_log_prompt(
            "[Performance][eager]:", json.dumps(time_stats), args.log_prompt
        )
