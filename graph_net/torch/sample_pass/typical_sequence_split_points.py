import os
import argparse
import tempfile
from pathlib import Path

from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.torch.typical_sequence_split_points import main as original_main


class TypicalSequenceSplitPointsGenerator(SamplePass):
    def __init__(self, config=None):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        op_names_path_prefix: str,
        output_dir: str,
        resume: bool = False,
        window_size: int = 10,
        fold_policy: str = "default",
        fold_times: int = 0,
        min_seq_ops: int = 2,
        max_seq_ops: int = 64,
        subgraph_ranges_file_name: str = "subgraph_ranges.json",
        subgraph_ranges_key_name: str = "subgraph_ranges",
        subgraph_ranges_json: str = "subgraph_ranges.json",
        output_json: str = "split_results.json",
        device: str = "cpu",
    ):
        pass

    def __call__(self, rel_model_path: str):
        txt_path = (
            Path(self.config["op_names_path_prefix"]) / rel_model_path / "op_names.txt"
        )
        if not txt_path.exists():
            print(f"[Warning] Operator names file not found: {txt_path}")
            return

        print(f"[TypicalSequenceSplitPointsGenerator] Queued {rel_model_path}")

    def END(self, rel_model_path: str):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for model_path in rel_model_path:
                f.write(f"{model_path}\n")
            model_list_file = f.name

        args = argparse.Namespace(
            model_list=model_list_file,
            op_names_path_prefix=self.config["op_names_path_prefix"],
            output_dir=self.config["output_dir"],
            window_size=self.config.get("window_size", 10),
            fold_policy=self.config.get("fold_policy", "default"),
            fold_times=self.config.get("fold_times", 0),
            subgraph_ranges_file_name=self.config.get(
                "subgraph_ranges_file_name", "subgraph_ranges.json"
            ),
            subgraph_ranges_key_name=self.config.get(
                "subgraph_ranges_key_name", "subgraph_ranges"
            ),
            device=self.config.get("device", "cpu"),
            enable_resume=self.config.get("resume", False),
            min_seq_ops=self.config.get("min_seq_ops", 2),
            max_seq_ops=self.config.get("max_seq_ops", 64),
            subgraph_ranges_json=self.config.get(
                "subgraph_ranges_json", "subgraph_ranges.json"
            ),
            output_json=self.config.get("output_json", "split_results.json"),
        )

        print(f"[TypicalSequenceSplitPointsGenerator] Processing {rel_model_path}")
        original_main(args)
        if os.path.exists(model_list_file):
            os.unlink(model_list_file)
