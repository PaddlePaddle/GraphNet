import sys
import subprocess
import json
import os
from graph_net.fault_locator.utils import get_tmp_file_path


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
