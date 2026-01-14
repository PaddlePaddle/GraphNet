import os
import shutil
from pathlib import Path
import torch
import json
import sys

from graph_net.torch.decompose_util import convert_to_submodules_graph
from graph_net.torch.extractor import GraphExtractor as BuiltinGraphExtractor
import graph_net.imp_util as imp_util
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module

import logging

logger = logging.getLogger(__name__)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class GraphExtractor:
    """
    Used by graph_net.torch.run_model
    """

    def __init__(
        self,
        config: dict,
        name,
        dynamic,
        mut_graph_codes=None,
        placeholder_auto_rename=False,
    ):
        self.subgraph_counter = 0
        self.name = name
        self.dynamic = dynamic
        self.mut_graph_codes = mut_graph_codes
        self.placeholder_auto_rename = placeholder_auto_rename
        self.config = self.make_config(**config)

    def make_config(
        self,
        split_positions=(),
        group_head_and_tail=False,
        chain_style=False,
        use_all_inputs=False,
        output_dir="./tmp/naive_decomposer_dir",
        filter_path=None,
        filter_config=None,
        **kwargs,
    ):
        for pos in split_positions:
            assert isinstance(
                pos, int
            ), f"split_positions should be list of int, {split_positions=}"
        return {
            "split_positions": split_positions,
            "group_head_and_tail": group_head_and_tail,
            "chain_style": chain_style,
            "use_all_inputs": use_all_inputs,
            "output_dir": output_dir,
            "filter_path": filter_path,
            "filter_config": filter_config if filter_config is not None else {},
        }

    def __call__(self, gm: torch.fx.GraphModule, sample_inputs):
        config = {
            k: v
            for k, v in self.config.items()
            if k
            in {
                "split_positions",
                "group_head_and_tail",
                "chain_style",
                "use_all_inputs",
            }
        }
        rewrited_gm = convert_to_submodules_graph(
            gm,
            submodule_hook=self.get_naive_decomposer_extractor,
            **config,
        )
        return rewrited_gm

    def get_naive_decomposer_extractor(self, submodule, seq_no):
        return NaiveDecomposerExtractorModule(
            config=self.config,
            parent_graph_rel_model_path=self.name,
            submodule=submodule,
            seq_no=seq_no,
        )


class NaiveDecomposerExtractor:
    """
    Used by graph_net.model_path_handler
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)

    def _make_config(
        self,
        output_dir,
        split_positions=(),
        group_head_and_tail=False,
        chain_style=False,
        use_all_inputs=False,
        filter_path=None,
        filter_config=None,
        model_path_prefix="",
        **kwargs,
    ):
        for pos in split_positions:
            assert isinstance(
                pos, int
            ), f"split_positions should be list of int, {split_positions=}"
        return {
            "split_positions": split_positions,
            "group_head_and_tail": group_head_and_tail,
            "chain_style": chain_style,
            "use_all_inputs": use_all_inputs,
            "output_dir": output_dir,
            "filter_path": filter_path,
            "filter_config": filter_config if filter_config is not None else {},
            "model_path_prefix": model_path_prefix,
        }

    def __call__(self, rel_model_path):
        # logger.warning("naive decomposer called")
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        config = {
            k: v
            for k, v in self.config.items()
            if k
            in {
                "split_positions",
                "group_head_and_tail",
                "chain_style",
                "use_all_inputs",
            }
        }
        module, inputs = get_torch_module_and_inputs(model_path, use_dummy_inputs=False)
        gm = parse_immutable_model_path_into_sole_graph_module(model_path)
        try:
            # logger.warning("convert_to_submodules_graph-call-begin")
            rewrited_gm: torch.fx.GraphModule = convert_to_submodules_graph(
                gm,
                submodule_hook=self.get_naive_decomposer_extractor(model_path),
                **config,
            )
            rewrited_gm(*inputs)
        finally:
            pass
            # logger.warning("convert_to_submodules_graph-call-end")

    def get_naive_decomposer_extractor(self, model_path):
        def fn(submodule, seq_no):
            return NaiveDecomposerExtractorModule(
                config=self.config,
                parent_graph_rel_model_path=os.path.basename(model_path),
                submodule=submodule,
                seq_no=seq_no,
            )

        return fn


class RangeDecomposerExtractor:
    """
    Used by graph_net.model_path_handler
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)

    def _make_config(
        self,
        resume: bool = False,
        split_results_path=None,
        subgraph_ranges_path=None,
        group_head_and_tail=False,
        chain_style=False,
        use_all_inputs=False,
        output_dir="./tmp/naive_decomposer_dir",
        filter_path=None,
        filter_config=None,
        model_path_prefix="",
        **kwargs,
    ):
        if os.path.isfile(split_results_path) and split_results_path.endswith(".json"):
            pass
        elif os.path.isfile(subgraph_ranges_path) and subgraph_ranges_path.endswith(
            ".json"
        ):
            pass
        else:
            raise ValueError(
                f"split_results_path should be a valid JSON file path, but got {split_results_path=}"
            )
        return {
            "resume": resume,
            "split_results_path": split_results_path,
            "subgraph_ranges_path": subgraph_ranges_path,
            "group_head_and_tail": group_head_and_tail,
            "chain_style": chain_style,
            "use_all_inputs": use_all_inputs,
            "output_dir": output_dir,
            "filter_path": filter_path,
            "filter_config": filter_config if filter_config is not None else {},
            "model_path_prefix": model_path_prefix,
        }

    def _is_model_handled(self, rel_model_path, split_positions, subgraph_ranges):
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

    def __call__(self, rel_model_path):
        assert os.path.realpath(self.config["model_path_prefix"]) != os.path.realpath(
            self.config["output_dir"]
        )
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        split_positions_json = load_json(self.config["split_results_path"])
        subgraph_ranges_json = load_json(self.config["subgraph_ranges_path"])
        if (
            split_positions_json[rel_model_path]["split_positions"] is None
            or len(split_positions_json[rel_model_path]["split_positions"]) == 0
            or subgraph_ranges_json[rel_model_path]["subgraph_ranges"] is None
            or len(subgraph_ranges_json[rel_model_path]["subgraph_ranges"]) == 0
        ):
            sys.stderr.write(f"Error: {rel_model_path} has no split positions.\n")
            return
        split_positions = split_positions_json[rel_model_path]["split_positions"]
        subgraph_ranges = subgraph_ranges_json[rel_model_path]["subgraph_ranges"]
        if self.config["resume"] and self._is_model_handled(
            rel_model_path, split_positions, subgraph_ranges
        ):
            return

        torch.cuda.empty_cache()
        module, inputs = get_torch_module_and_inputs(model_path, use_dummy_inputs=False)
        gm = parse_sole_graph_module(module, inputs)

        torch.cuda.empty_cache()
        rewrited_gm: torch.fx.GraphModule = convert_to_submodules_graph(
            gm,
            submodule_hook=self.get_naive_decomposer_extractor(rel_model_path),
            split_positions=split_positions,
            subgraph_ranges=subgraph_ranges,
            group_head_and_tail=self.config.get("group_head_and_tail", False),
            chain_style=self.config.get("chain_style", False),
            use_all_inputs=self.config.get("use_all_inputs", False),
        )
        rewrited_gm(*inputs)

    def get_naive_decomposer_extractor(self, rel_model_path):
        def fn(submodule, seq_no):
            return NaiveDecomposerExtractorModule(
                config=self.config,
                parent_graph_rel_model_path=rel_model_path,
                submodule=submodule,
                seq_no=seq_no,
            )

        return fn


class NaiveDecomposerExtractorModule(torch.nn.Module):
    def __init__(
        self,
        config: dict,
        parent_graph_rel_model_path: str,
        submodule: torch.nn.Module,
        seq_no: int,
    ):
        super().__init__()
        self.config = config
        self.submodule = submodule
        self.seq_no = seq_no
        self.extracted = False
        self.parent_graph_rel_model_path = parent_graph_rel_model_path
        parent_graph_model_name = os.path.basename(parent_graph_rel_model_path)
        if self.seq_no is None:
            self.model_name = parent_graph_model_name
        else:
            submodule_name = f"{parent_graph_model_name}_{self.seq_no}"
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
        self.filter = self.make_filter(self.config)

    def _get_model_path(self):
        return os.path.join(
            self.config["output_dir"],
            f"{self.parent_graph_model_name}/_decomposed",
            self.model_name,
        )

    def forward(self, *args):
        # logger.warning("naive decomposer forwarding")
        if not self.extracted:
            if self.need_extract(self.submodule, args):
                self.builtin_extractor(self.submodule, args)
            self.extracted = True
        return self.submodule(*args)

    def need_extract(self, gm, sample_inputs):
        if self.filter is None:
            return True
        return self.filter(gm, sample_inputs)

    def make_filter(self, config):
        if config["filter_path"] is None:
            return None
        module = imp_util.load_module(config["filter_path"])
        return module.GraphFilter(config["filter_config"])
