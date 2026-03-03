import json
from pathlib import Path
from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin


class OpExtractPointsGenerator(SamplePass, ResumableSamplePassMixin):
    def __init__(self, config=None):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        op_names_path_prefix: str,
        resume: bool = False,
        limits_handled_models: int = None,
        subgraph_ranges_file_name: str = "subgraph_ranges.json",
        subgraph_ranges_json: str = None,
        output_json: str = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        return self.naive_sample_handled(
            rel_model_path, search_file_name=self.config["subgraph_ranges_file_name"]
        )

    def resume(self, rel_model_path: str):
        txt_path = (
            Path(self.config["op_names_path_prefix"]) / rel_model_path / "op_names.txt"
        )
        if not txt_path.exists():
            print(f"File not found: {txt_path}")
            return
        with open(txt_path, "r") as f:
            seq = [line.strip() for line in f if line.strip()]
        if not seq:
            print(f"Empty sequence in: {txt_path}")
            return

        num_ops = len(seq)
        model_name = Path(rel_model_path).name
        subgraph_ranges = [[i, i + 1] for i in range(num_ops)]
        self._save_individual_subgraph_ranges(
            rel_model_path, model_name, num_ops, subgraph_ranges
        )
        self._update_combined_results(
            rel_model_path, model_name, num_ops, subgraph_ranges
        )
        print(
            f"Created single operator ranges for {model_name}: {len(subgraph_ranges)} subgraphs"
        )

    def _save_individual_subgraph_ranges(
        self, rel_model_path, model_name, num_ops, subgraph_ranges
    ):
        model_output_dir = Path(self.config["output_dir"]) / rel_model_path
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_output_dir / self.config["subgraph_ranges_file_name"]
        with open(output_file, "w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "subgraph_ranges": subgraph_ranges,
                    "total_length": num_ops,
                },
                f,
                indent=4,
            )

    def _update_combined_results(
        self, rel_model_path, model_name, num_ops, subgraph_ranges
    ):
        split_positions_json = self._load_combined_json(self.config.get("output_json"))
        subgraph_ranges_json_data = self._load_combined_json(
            self.config.get("subgraph_ranges_json")
        )
        split_positions = sorted(
            set(pos for start, end in subgraph_ranges for pos in (start, end))
        )
        split_positions_json[str(rel_model_path)] = {
            "model_name": model_name,
            "split_positions": split_positions,
            "total_length": num_ops,
        }
        subgraph_ranges_json_data[str(rel_model_path)] = {
            "model_name": model_name,
            "subgraph_ranges": subgraph_ranges,
            "total_length": num_ops,
        }
        if self.config.get("output_json"):
            with open(self.config["output_json"], "w") as f:
                json.dump(split_positions_json, f, indent=4)

        if self.config.get("subgraph_ranges_json"):
            with open(self.config["subgraph_ranges_json"], "w") as f:
                json.dump(subgraph_ranges_json_data, f, indent=4)

    def _load_combined_json(self, json_path):
        if not json_path:
            return {}

        path = Path(json_path)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}
