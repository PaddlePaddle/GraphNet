from graph_net.sample_pass.sample_pass import SamplePass
from typing import Generator
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
import os
from pathlib import Path
import torch
import json

from graph_net.torch.decompose_util import gen_submodule_input_nodes
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module


class SubgraphInputProducerIndexesGenerator(SamplePass, ResumableSamplePassMixin):
    def __init__(self, config):
        super().__init__(config)

    def declare_config(
        self,
        output_dir: str,
        model_path_prefix: str,
        subgraph_ranges_json_root: str,
        subgraph_ranges_json_file_name: str,
        subgraph_ranges_json_key: str,
        subgraph_ranges_json_rel_model_path_key: str = "subgraph_rel_model_paths",
        output_json_file_name: str = "subgraph_input_producer_indexes.json",
        output_json_key: str = "input_producer_indexes",
        output_json_subgraph_rel_model_path_key: str = "subgraph_rel_model_paths",
        group_head_and_tail: bool = False,
        chain_style: bool = False,
        device: str = "auto",
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        file_name = self.config["output_json_file_name"]
        return self.naive_sample_handled(rel_model_path, search_file_name=file_name)

    def resume(self, rel_model_path: str):
        subgraph_input_producer_indexes = self._get_subgraph_input_producer_indexes(
            rel_model_path
        )
        dst_model_path = Path(self.config["output_dir"]) / rel_model_path
        dst_model_path.mkdir(parents=True, exist_ok=True)
        json_str = json.dumps(subgraph_input_producer_indexes, indent=4)
        (dst_model_path / self.config["output_json_file_name"]).write_text(json_str)

    def _get_subgraph_input_producer_indexes(self, rel_model_path):
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        torch.cuda.empty_cache()
        device = self._choose_device(self.config["device"])
        module, inputs = get_torch_module_and_inputs(
            model_path, use_dummy_inputs=False, device=device
        )
        gm = parse_sole_graph_module(module, inputs)
        torch.cuda.empty_cache()
        subgraph_info_json = self._get_subgraph_info_json(rel_model_path)

        def get_subgraph_input_producer_indexes_json_obj():
            subgraph_ranges = self._get_subgraph_ranges(subgraph_info_json)
            triples: Generator[(int, int, torch.fx.Node)] = gen_submodule_input_nodes(
                gm,
                subgraph_ranges=subgraph_ranges,
                group_head_and_tail=self.config.get("group_head_and_tail", False),
                chain_style=self.config.get("chain_style", False),
            )
            node2node_idx = dict((node, i) for i, node in enumerate(gm.graph.nodes))
            input_producer_indexes = [
                {
                    "range_start": start,
                    "range_end": end,
                    "input_producer_indexes": [node2node_idx[node] for node in nodes],
                }
                for start, end, nodes in triples
            ]
            return {self.config["output_json_key"]: input_producer_indexes}

        def get_subgraph_rel_model_paths_json_obj():
            return {
                self.config[
                    "output_json_subgraph_rel_model_path_key"
                ]: self._get_subgraph_paths(subgraph_info_json)
            }

        return {
            **get_subgraph_input_producer_indexes_json_obj(),
            **get_subgraph_rel_model_paths_json_obj(),
        }

    def _get_subgraph_info_json(self, rel_model_path: str) -> dict[str, list]:
        model_path = Path(self.config["subgraph_ranges_json_root"]) / rel_model_path
        file_path = model_path / self.config["subgraph_ranges_json_file_name"]
        json_str = file_path.read_text()
        return json.loads(json_str)

    def _get_subgraph_ranges(self, subgraph_ranges_and_paths_json) -> list[(int, int)]:
        key = self.config["subgraph_ranges_json_key"]
        return subgraph_ranges_and_paths_json[key]

    def _get_subgraph_paths(self, subgraph_ranges_and_paths_json) -> list[str]:
        key = self.config["subgraph_ranges_json_rel_model_path_key"]
        return subgraph_ranges_and_paths_json[key]

    def _choose_device(self, device) -> str:
        if device in ["cpu", "cuda"]:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"
