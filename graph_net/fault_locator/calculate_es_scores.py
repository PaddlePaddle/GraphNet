import sys
import subprocess
import json
import os
from graph_net.fault_locator.utils import get_tmp_file_path
from typing import Dict
from graph_net.declare_config_mixin import DeclareConfigMixin
from graph_net_bench.calculate_es_scores import calculate_es_scores_for_log_contents


class ESScoresCalculator(DeclareConfigMixin):
    """
    Functor responsible for converting raw log strings into structured Error Scores (ES).
    It wraps the core scoring logic and provides a configuration interface.
    """

    def __init__(self, config=None):
        self.init_config(config)

    def declare_config(
        self,
        interpretation_type: str = "default",
        negative_speedup_penalty: float = 0.0,
        fpdb: float = 0.1,
        enable_aggregation_mode: bool = True,
    ):
        """
        Configuration schema for ES score calculation parameters.
        """
        pass

    def __call__(self, log_content: str) -> Dict[int, float]:
        """
        Calculates ES scores from a raw log string.

        Args:
            log_content: The full stdout/stderr content from a benchmark run.

        Returns:
            A dictionary mapping layer indices (int) to their error scores (float).
        """
        # The core logic expects a list of strings (lines or log blocks)
        # Here we treat the single call's content as a one-element list
        results = calculate_es_scores_for_log_contents(
            log_contents=[log_content],
            interpretation_type=self.config["interpretation_type"],
            negative_speedup_penalty=self.config["negative_speedup_penalty"],
            fpdb=self.config["fpdb"],
            enable_aggregation_mode=self.config["enable_aggregation_mode"],
        )

        # Validate that the log produced exactly one set of scores
        if not results:
            return {}

        assert (
            len(results) == 1
        ), f"Expected 1 model result from log content, but got {len(results)}"

        # Return the score dictionary for the single model processed
        return results[0]


def calculate_es_scores(log_file_path: str) -> dict[int, float]:
    with get_tmp_file_path() as output_file_path:
        _calculate_es_scores(log_file_path, output_file_path)
        with open(output_file_path) as f:
            es_scores = json.load(f)
    return {int(k): float(v) for k, v in es_scores.items()}


def _calculate_es_scores(log_file_path: str, output_file_path: str) -> dict[int, float]:
    cmd = [
        sys.executable,
        "-m",
        "graph_net_bench.calculate_es_scores",
        "--benchmark-path",
        log_file_path,
        "--output-json-file-path",
        output_file_path,
    ]
    try:
        # We use subprocess.run, but we need to catch KeyboardInterrupt
        subprocess.run(cmd, text=True, check=True)
    except KeyboardInterrupt:
        # Log a professional message and exit with code 130 (Standard for SIGINT)
        print("\n[INTERRUPT] User requested termination. Exiting fault locator...")
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        error_msg = f"ES calculation failed with return code {e.returncode}.\nStderr: {e.stderr}"
        raise RuntimeError(error_msg) from e

    # Post-process the results
    if not os.path.exists(output_file_path):
        raise FileNotFoundError(f"ES results missing at: {output_file_path}")
