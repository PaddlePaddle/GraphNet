from graph_net.sample_pass.sample_pass import SamplePass
from pathlib import Path
import json


class GroupFusibleSubgraphRanges(SamplePass):
    def __init__(self, config=None):
        super().__init__(config)
        self.original_graph_rel_model_path2ranges: dict[str, list[(int, int)]] = {}
        self.original_graph_rel_model_path2subgraph_rel_model_paths: dict[
            str, list[str]
        ] = {}

    def declare_config(
        self,
        subgraph_model_path_prefix: str,
        output_dir: str,
        input_json_file_name: str = "fusible_subgraph_ranges.json",
        output_json_file_name: str = "grouped_fusible_subgraph_ranges.json",
        output_json_key: str = "subgraph_ranges",
        output_json_subgraph_rel_model_path_key: str = "fusible_subgraph_relative_model_paths",
    ):
        pass

    def __call__(self, subgraph_rel_model_path: str):
        model_path = (
            Path(self.config["subgraph_model_path_prefix"])
            / subgraph_rel_model_path
            / self.config["input_json_file_name"]
        )
        subgraph_sources = json.load(open(model_path))
        subgraph_ranges = subgraph_sources.get(self.config["output_json_key"], [])
        original_graph_rel_model_path = self._extract_original_model_path(
            subgraph_rel_model_path
        )
        self._collect_original_graph_rel_model_path2ranges(
            original_graph_rel_model_path, subgraph_ranges
        )
        self._collect_original_graph_rel_model_path2subgraph_rel_model_path(
            original_graph_rel_model_path,
            [subgraph_rel_model_path] * len(subgraph_ranges),
        )

    def _extract_original_model_path(self, rel_model_path: str) -> str:
        path_parts = Path(rel_model_path).parts
        if "_decomposed" in path_parts:
            decomposed_idx = path_parts.index("_decomposed")
            return str(Path(*path_parts[:decomposed_idx]))
        return rel_model_path

    def _collect_original_graph_rel_model_path2subgraph_rel_model_path(
        self,
        original_graph_rel_model_path: str,
        subgraph_rel_model_paths: list[str],
    ):
        old = self.original_graph_rel_model_path2subgraph_rel_model_paths.get(
            original_graph_rel_model_path, []
        )
        self.original_graph_rel_model_path2subgraph_rel_model_paths[
            original_graph_rel_model_path
        ] = [
            *old,
            *subgraph_rel_model_paths,
        ]

    def _collect_original_graph_rel_model_path2ranges(
        self, original_graph_rel_model_path, subgraph_ranges
    ):
        old_ranges = self.original_graph_rel_model_path2ranges.get(
            original_graph_rel_model_path, []
        )
        self.original_graph_rel_model_path2ranges[original_graph_rel_model_path] = [
            *old_ranges,
            *subgraph_ranges,
        ]

    def END(self, rel_model_paths: list[str]):
        for (
            original_graph_rel_model_path,
            subgraph_ranges,
        ) in self.original_graph_rel_model_path2ranges.items():
            subgraph_rel_model_paths = (
                self.original_graph_rel_model_path2subgraph_rel_model_paths[
                    original_graph_rel_model_path
                ]
            )
            self._save_json(
                original_graph_rel_model_path, subgraph_ranges, subgraph_rel_model_paths
            )

    def _save_json(
        self, original_graph_rel_model_path, subgraph_ranges, subgraph_rel_model_paths
    ):
        model_dir = Path(self.config["output_dir"]) / original_graph_rel_model_path
        model_dir.mkdir(parents=True, exist_ok=True)

        # Sort ranges by start index, and sort paths accordingly
        sorted_data = sorted(
            zip(subgraph_ranges, subgraph_rel_model_paths), key=lambda x: x[0][0]
        )
        sorted_ranges, sorted_paths = zip(*sorted_data) if sorted_data else ([], [])

        ranges_json = self._get_ranges_json(list(sorted_ranges))
        paths_json = self._get_paths_json(list(sorted_paths))
        json_obj = {**ranges_json, **paths_json}
        json_str = json.dumps(json_obj, indent=4)
        (model_dir / self.config["output_json_file_name"]).write_text(json_str)

    def _get_paths_json(self, subgraph_rel_model_paths: list[str]):
        json_obj = {
            self.config[
                "output_json_subgraph_rel_model_path_key"
            ]: subgraph_rel_model_paths
        }
        return json_obj

    def _get_ranges_json(self, subgraph_ranges: list[(int, int)]):
        json_obj = {self.config["output_json_key"]: subgraph_ranges}
        return json_obj
