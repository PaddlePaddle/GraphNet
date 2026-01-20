import os
import json
import base64
from pathlib import Path
from typing import Callable
from graph_net_bench.calculate_es_scores import calculate_es_scores_for_each_model_path
from graph_net.fault_locator.fault_detector import has_fault_at
from graph_net.declare_config_mixin import DeclareConfigMixin


class FaultModelPathFilter(DeclareConfigMixin):
    def __init__(self, config=None):
        self.init_config(config)

    def declare_config(
        self,
        log_file_path: str,
        output_txt_file_path: str,
        model_path_prefix: str,
        graph_net_log_prompt: str = "graph-net-test-compiler-log",
        tolerance: int = 0,
        interpretation_type: str = "default",
        negative_speedup_penalty: float = 0.0,
        fpdb: float = 0.1,
        enable_aggregation_mode: bool = True,
    ):
        """
        Configuration for filtering models with numerical faults.
        """
        pass

    def _get_model_path_extractor(self) -> Callable[[str], str | None]:
        """
        Creates an extractor that looks for:
        '{graph_net_log_prompt} [Processing] {model_path}'
        """
        prompt = self.config.get("graph_net_log_prompt", "graph-net-test-compiler-log")
        header = f"{prompt} [Processing]"

        def extractor(line: str) -> str | None:
            if line.startswith(header):
                parts = line.split()
                # Example: "graph-net-test-compiler-log [Processing] /path/to/model"
                # Index 0: prompt, Index 1: [Processing], Index 2: model_path
                if len(parts) >= 3:
                    return parts[2].strip()
            return None

        return extractor

    def __call__(self):
        # 1. Load log content
        log_path = self.config["log_file_path"]
        log_content = Path(log_path).read_text(encoding="utf-8")

        # 2. Extract scores grouped by model_path
        # Uses the logic: cumsum -> groupby -> calculate_es_scores_for_log_contents
        path2es_scores = calculate_es_scores_for_each_model_path(
            log_content=log_content,
            get_model_path_for_each_log_line=self._get_model_path_extractor(),
            interpretation_type=self.config.get("interpretation_type", "default"),
            negative_speedup_penalty=self.config.get("negative_speedup_penalty", 0.0),
            fpdb=self.config.get("fpdb", 0.1),
            enable_aggregation_mode=self.config.get("enable_aggregation_mode", True),
        )

        # 3. Filter for faults and convert to relative paths
        faulty_relative_paths = []
        prefix = self.config["model_path_prefix"]
        tolerance = self.config.get("tolerance", 0)

        for full_path, es_scores in path2es_scores.items():
            # Imported/Existing function check
            if has_fault_at(es_scores, tolerance):
                # Ensure we output paths relative to the model_path_prefix
                rel_path = os.path.relpath(full_path, prefix)
                faulty_relative_paths.append(rel_path)

        # 4. Save results (one model_path per line)
        output_path = self.config["output_txt_file_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for path in faulty_relative_paths:
                f.write(f"{path}\n")

        print(f"[Success] Identified {len(faulty_relative_paths)} faulty models.")
        print(f"[File] Saved to: {output_path}")


if __name__ == "__main__":
    import sys

    # Expecting a single argument: base64 encoded JSON string
    if len(sys.argv) > 1:
        try:
            encoded_config = sys.argv[1]
            decoded_json = base64.b64decode(encoded_config).decode("utf-8")
            config_dict = json.loads(decoded_json)

            filter_app = FaultModelPathFilter(config_dict)
            filter_app()
        except Exception as e:
            print(f"Filter execution failed: {e}")
            sys.exit(1)
