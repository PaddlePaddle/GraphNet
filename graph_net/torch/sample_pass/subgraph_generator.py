from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
import graph_net.subgraph_range_util as range_util
import os
import shutil
from pathlib import Path
import torch
import json

from graph_net.torch.decompose_util import convert_to_submodules_graph
from graph_net.torch.extractor import GraphExtractor as BuiltinGraphExtractor
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class SubgraphGenerator(SamplePass, ResumableSamplePassMixin):
    def __init__(self, config):
        super().__init__(config)

    def declare_config(
        self,
        output_dir: str,
        model_path_prefix: str,
        subgraph_ranges_json_root: str,
        subgraph_ranges_json_file_name: str = "subgraph_ranges.json",
        subgraph_ranges_json_key: str = "subgraph_ranges",
        group_head_and_tail: bool = False,
        chain_style: bool = False,
        use_all_inputs: bool = False,
        device: str = "auto",
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        subgraph_ranges = self._get_subgraph_ranges(rel_model_path)
        split_positions = self._get_split_positions(subgraph_ranges)
        if self.config["chain_style"]:
            return self._has_enough_subgraphs(
                rel_model_path,
                num_subgraphs=len(split_positions) + 1,
            )
        else:
            return self._has_enough_subgraphs(
                rel_model_path,
                num_subgraphs=len(subgraph_ranges),
            )

    def _has_enough_subgraphs(self, rel_model_path, num_subgraphs):
        decomposed_model_path = Path(self.config["output_dir"]) / rel_model_path
        num_decomposed = len(list(decomposed_model_path.rglob("model.py")))
        if num_decomposed > 0 and num_subgraphs != num_decomposed:
            shutil.rmtree(decomposed_model_path / "_decomposed")
            return False
        return num_subgraphs == num_decomposed

    def resume(self, rel_model_path: str):
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        torch.cuda.empty_cache()
        device = self._choose_device(self.config["device"])
        module, inputs = get_torch_module_and_inputs(
            model_path, use_dummy_inputs=False, device=device
        )
        gm = parse_sole_graph_module(module, inputs)
        torch.cuda.empty_cache()
        subgraph_ranges = self._get_subgraph_ranges(rel_model_path)
        split_positions = self._get_split_positions(subgraph_ranges)
        rewrited_gm: torch.fx.GraphModule = convert_to_submodules_graph(
            gm,
            submodule_hook=self.get_naive_decomposer_extractor(
                rel_model_path, subgraph_ranges
            ),
            split_positions=split_positions,
            subgraph_ranges=subgraph_ranges,
            group_head_and_tail=self.config.get("group_head_and_tail", False),
            chain_style=self.config.get("chain_style", False),
            use_all_inputs=self.config.get("use_all_inputs", False),
        )
        rewrited_gm(*inputs)

    def _get_subgraph_ranges(self, rel_model_path: str) -> list[(int, int)]:
        model_path = Path(self.config["subgraph_ranges_json_root"]) / rel_model_path
        file_path = model_path / self.config["subgraph_ranges_json_file_name"]
        json_str = file_path.read_text()
        json_obj = json.loads(json_str)
        key = self.config["subgraph_ranges_json_key"]
        return json_obj[key]

    def _get_split_positions(self, subgraph_ranges: list[(int, int)]):
        split_positions = [position for pair in subgraph_ranges for position in pair]
        return sorted(set(split_positions))

    def get_naive_decomposer_extractor(self, rel_model_path, subgraph_ranges):
        def fn(submodule, seq_no):
            return NaiveDecomposerExtractorModule(
                config=self.config,
                parent_graph_rel_model_path=rel_model_path,
                submodule=submodule,
                seq_no=seq_no,
                subgraph_start=subgraph_ranges[seq_no][0],
                subgraph_end=subgraph_ranges[seq_no][1],
                parent_graph_model_path_root=self.config["model_path_prefix"],
            )

        return fn

    def _choose_device(self, device) -> str:
        if device in ["cpu", "cuda"]:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"


class NaiveDecomposerExtractorModule(torch.nn.Module):
    def __init__(
        self,
        config: dict,
        parent_graph_rel_model_path: str,
        submodule: torch.nn.Module,
        seq_no: int,
        subgraph_start: int,
        subgraph_end: int,
        parent_graph_model_path_root: str,
    ):
        super().__init__()
        self.extracted = False
        self.config = config
        self.parent_graph_rel_model_path = parent_graph_rel_model_path
        self.submodule = submodule
        self.seq_no = seq_no
        self.subgraph_start = subgraph_start
        self.subgraph_end = subgraph_end
        self.parent_graph_model_path_root = parent_graph_model_path_root
        parent_graph_model_name = os.path.basename(parent_graph_rel_model_path)
        if self.seq_no is None:
            self.model_name = parent_graph_model_name
        else:
            submodule_name = f"{parent_graph_model_name}_start{subgraph_start}_end{subgraph_end}_{self.seq_no}"
            self.model_name = submodule_name
        self.builtin_extractor = BuiltinGraphExtractor(
            name=submodule_name,
            dynamic=False,
            mut_graph_codes=[],
            placeholder_auto_rename=False,
            workspace_path=os.path.join(
                self.config["output_dir"], parent_graph_rel_model_path, "_decomposed"
            ),
        )
        self._save_subgraph_sources()

    def _get_model_path(self) -> Path:
        return (
            Path(self.config["output_dir"])
            / self.parent_graph_rel_model_path
            / "_decomposed"
            / self.model_name
        )

    def forward(self, *args):
        if not self.extracted:
            self.builtin_extractor(self.submodule, args)
            self.extracted = True
        return self.submodule(*args)

    def _save_subgraph_sources(self):
        sources_json_obj = self._get_sources_json_obj()
        model_path = self._get_model_path()
        model_path.mkdir(parents=True, exist_ok=True)
        print(f"{str(model_path)=}")
        with open(model_path / self._get_subgraph_range_json_file_name(), "w") as f:
            json.dump(sources_json_obj, f, indent=4)

    def _get_sources_json_obj(self):
        cur_sources_json_obj = self._get_cur_sources_json_obj()
        parent_sources_json_obj = self._get_parent_sources_json_obj()
        return self._compose_sources_json_obj(
            cur_sources_json_obj, parent_sources_json_obj
        )

    def _get_cur_sources_json_obj(self):
        return {
            self.parent_graph_rel_model_path: [(self.subgraph_start, self.subgraph_end)]
        }

    def _get_parent_sources_json_obj(self):
        file_path = self._get_parent_sources_json_file_path()

        def get_grand_parent_subgraph_ranges():
            return json.load(open(file_path)) if file_path.exists() else {}

        return {self.parent_graph_rel_model_path: get_grand_parent_subgraph_ranges()}

    def _get_parent_sources_json_file_path(self):
        return (
            Path(self._get_parent_model_path_root())
            / self.parent_graph_rel_model_path
            / self._get_subgraph_range_json_file_name()
        )

    def _get_parent_model_path_root(self):
        return self.parent_graph_model_path_root

    def _get_subgraph_range_json_file_name(self):
        return "subgraph_sources.json"

    def _compose_sources_json_obj(
        self,
        cur_sources_json_obj: dict[str, list[(int, int)]],
        parent_sources_json_obj: dict[str, dict[str, list[(int, int)]]],
    ):
        return range_util.compose_subgraph_ranges(
            cur_sources_json_obj, parent_sources_json_obj
        )
