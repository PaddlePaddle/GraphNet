from graph_net.sample_pass.sample_pass import SamplePass
import graph_net.subgraph_range_util as range_util
from pathlib import Path
import json
from typing import Union


class MergeSubgraphSources(SamplePass):
    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)

    def declare_config(
        self,
        subgraph_sources_json_file_name: str = "subgraph_sources.json",
    ):
        pass

    def __call__(self, rel_model_path: str):
        pass

    def merge_sources_for_deduplication(
        self,
        target_model_path: Union[str, Path],
        source_model_paths: list[Union[str, Path]],
    ) -> dict[str, list[tuple[int, int]]]:
        merged_sources = self._load_sources(target_model_path)
        for source_path in source_model_paths:
            source_sources = self._load_sources(source_path)
            if source_sources:
                merged_sources = range_util.merge_subgraph_ranges(
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
