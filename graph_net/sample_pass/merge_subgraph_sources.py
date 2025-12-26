from graph_net.sample_pass.sample_pass import SamplePass
import graph_net.subgraph_range_util as subgraph_range_util
from pathlib import Path
import json
from typing import Union


class MergeSubgraphSources(SamplePass):
    def __init__(self, config=None):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str = "",
        source_model_path_prefixes: list = None,
        subgraph_sources_json_file_name: str = "subgraph_sources.json",
    ):
        pass

    def __call__(self, rel_model_path: str):
        model_path_prefix = self.config.get("model_path_prefix", "")
        target_model_path = (
            Path(model_path_prefix) / rel_model_path
            if model_path_prefix
            else Path(rel_model_path)
        )

        source_model_path_prefixes = self.config.get("source_model_path_prefixes") or []
        source_model_paths = [
            Path(prefix) / rel_model_path for prefix in source_model_path_prefixes
        ]

        self.merge_sources_for_deduplication(target_model_path, source_model_paths)
        print(f"Merged {len(source_model_paths)} sources into {target_model_path}")

    def merge_sources_for_deduplication(
        self,
        target_model_path: Union[str, Path],
        source_model_paths: list[Union[str, Path]],
    ) -> dict[str, list[tuple[int, int]]]:
        merged_sources = self._load_sources(target_model_path)
        for source_path in source_model_paths:
            source_sources = self._load_sources(source_path)
            if source_sources:
                merged_sources = subgraph_range_util.merge_subgraph_ranges(
                    merged_sources, source_sources
                )
        self._save_sources(target_model_path, merged_sources)
        return merged_sources

    def _get_sources_file_path(self, model_path: Union[str, Path]) -> Path:
        return Path(model_path) / self.config["subgraph_sources_json_file_name"]

    def _load_sources(
        self, model_path: Union[str, Path]
    ) -> dict[str, list[tuple[int, int]]]:
        file_path = self._get_sources_file_path(model_path)
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_sources(
        self,
        model_path: Union[str, Path],
        sources: dict[str, list[tuple[int, int]]],
    ) -> None:
        file_path = self._get_sources_file_path(model_path)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=4)
