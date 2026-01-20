from graph_net.sample_pass.sample_pass import SamplePass
from pathlib import Path
import json
import re


class GroupRangesFromSubgraphSources(SamplePass):
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
        subgraph_sources_json_file_name: str = "subgraph_sources.json",
        output_json_file_name: str = "grouped_ranges_from_subgraph_sources.json",
        output_json_key: str = "grouped_ranges_from_subgraph_sources",
        output_json_subgraph_rel_model_path_key: str = "subgraph_relative_model_paths",
    ):
        pass

    def __call__(self, subgraph_rel_model_path: str):
        model_path = (
            Path(self.config["subgraph_model_path_prefix"])
            / subgraph_rel_model_path
            / self.config["subgraph_sources_json_file_name"]
        )
        subgraph_sources = json.load(open(model_path))
        for original_graph_rel_model_path, subgraph_ranges in subgraph_sources.items():
            # Extract actual start-end range from path, establish precise one-to-one mapping
            path_range = self._extract_range_from_path(subgraph_rel_model_path)
            if path_range:
                self._collect_original_graph_rel_model_path2ranges(
                    original_graph_rel_model_path, path_range
                )
                self._collect_original_graph_rel_model_path2subgraph_rel_model_path(
                    original_graph_rel_model_path, [subgraph_rel_model_path]
                )

    def _extract_range_from_path(self, path: str) -> list[int]:
        """Extract start and end range from path, e.g., vgg16_bn_start0_end3_0 → [0,3]"""
        match = re.search(r"start(\d+)_end(\d+)", path)
        if match:
            return [int(match.group(1)), int(match.group(2))]
        return None

    def _collect_original_graph_rel_model_path2subgraph_rel_model_path(
        self,
        original_graph_rel_model_path: str,
        subgraph_rel_model_paths: list[str],
    ):
        """Collect subgraph paths with automatic deduplication"""
        old = self.original_graph_rel_model_path2subgraph_rel_model_paths.get(
            original_graph_rel_model_path, []
        )
        # Deduplicate and merge, maintaining path uniqueness
        combined = old + [p for p in subgraph_rel_model_paths if p not in old]
        self.original_graph_rel_model_path2subgraph_rel_model_paths[
            original_graph_rel_model_path
        ] = combined

    def _collect_original_graph_rel_model_path2ranges(
        self, original_graph_path: str, path_range: list[int]
    ):
        """Collect subgraph ranges with automatic deduplication"""
        old_ranges = self.original_graph_rel_model_path2ranges.get(
            original_graph_path, []
        )
        if path_range not in old_ranges:
            old_ranges.append(path_range)
        self.original_graph_rel_model_path2ranges[original_graph_path] = old_ranges

    def END(self, rel_model_paths: list[str]):
        """Final processing: sort by range and save results"""
        for (
            original_graph_rel_model_path
        ) in self.original_graph_rel_model_path2ranges.keys():
            actual_ranges = self.original_graph_rel_model_path2ranges.get(
                original_graph_rel_model_path, []
            )
            subgraph_rel_model_paths = (
                self.original_graph_rel_model_path2subgraph_rel_model_paths.get(
                    original_graph_rel_model_path, []
                )
            )

            # Establish mapping relationship between ranges and paths
            range_to_path = {}
            for path in subgraph_rel_model_paths:
                path_range = self._extract_range_from_path(path)
                if path_range:
                    range_to_path[tuple(path_range)] = path

            # Sort by start position of ranges to ensure structured output
            sorted_ranges = sorted(actual_ranges, key=lambda x: x[0])
            sorted_paths = [
                range_to_path[tuple(r)]
                for r in sorted_ranges
                if tuple(r) in range_to_path
            ]

            self._save_json(original_graph_rel_model_path, sorted_ranges, sorted_paths)

    def _save_json(
        self, original_graph_rel_model_path, subgraph_ranges, subgraph_rel_model_paths
    ):
        """Save final aggregated results to JSON files"""
        model_dir = Path(self.config["output_dir"]) / original_graph_rel_model_path
        model_dir.mkdir(parents=True, exist_ok=True)
        ranges_json = self._get_ranges_json(subgraph_ranges)
        paths_json = self._get_paths_json(subgraph_rel_model_paths)
        json_obj = {**ranges_json, **paths_json}
        json_str = json.dumps(json_obj, indent=4)
        (model_dir / self.config["output_json_file_name"]).write_text(json_str)

    def _get_paths_json(self, subgraph_rel_model_paths: list[str]):
        """Generate JSON object for paths section"""
        json_obj = {
            self.config[
                "output_json_subgraph_rel_model_path_key"
            ]: subgraph_rel_model_paths
        }
        return json_obj

    def _get_ranges_json(self, subgraph_ranges: list[(int, int)]):
        """Generate JSON object for ranges section"""
        json_obj = {self.config["output_json_key"]: subgraph_ranges}
        return json_obj
