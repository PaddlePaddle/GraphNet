from graph_net.sample_pass.sample_pass import SamplePass
from pathlib import Path
import json


class GroupRangesFromSubgraphSources(SamplePass):
    def __init__(self, config=None):
        super().__init__(config)
        self.original_sample_rel_model_path2ranges: dict[str, list[(int, int)]] = {}

    def declare_config(
        self,
        subgraph_model_path_prefix: str,
        output_dir: str,
        subgraph_sources_json_file_name: str = "subgraph_sources.json",
        output_json_file_name: str = "grouped_ranges_from_subgraph_sources.json",
        output_json_key: str = "grouped_ranges_from_subgraph_sources",
    ):
        pass

    def __call__(self, rel_model_path: str):
        model_path = (
            Path(self.config["subgraph_model_path_prefix"])
            / rel_model_path
            / self.config["subgraph_sources_json_file_name"]
        )
        subgraph_sources = json.load(open(model_path))
        for rel_model_path, subgraph_ranges in subgraph_sources.items():
            self._collect_original_sample_rel_model_path2ranges(
                rel_model_path, subgraph_ranges
            )

    def _collect_original_sample_rel_model_path2ranges(
        self, rel_model_path, subgraph_ranges
    ):
        old_ranges = self.original_sample_rel_model_path2ranges.get(rel_model_path, [])
        self.original_sample_rel_model_path2ranges[rel_model_path] = [
            *old_ranges,
            *subgraph_ranges,
        ]

    def END(self, rel_model_paths: list[str]):
        for (
            rel_model_path,
            subgraph_ranges,
        ) in self.original_sample_rel_model_path2ranges.items():
            self._save_original_sample_subgraph_ranges(rel_model_path, subgraph_ranges)

    def _save_original_sample_subgraph_ranges(self, rel_model_path, subgraph_ranges):
        model_dir = Path(self.config["output_dir"]) / rel_model_path
        model_dir.mkdir(parents=True, exist_ok=True)
        json_str = self._get_original_sample_subgraph_ranges_json_str(subgraph_ranges)
        (model_dir / self.config["output_json_file_name"]).write_text(json_str)

    def _get_original_sample_subgraph_ranges_json_str(
        self, subgraph_ranges: list[(int, int)]
    ):
        subgraph_ranges = sorted(set(tuple(x) for x in subgraph_ranges))
        json_obj = {self.config["output_json_key"]: subgraph_ranges}
        return json.dumps(json_obj, indent=4)
