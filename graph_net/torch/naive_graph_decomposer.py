import os
import torch
import json
import base64
import shutil
from typing import Union, Callable
from graph_net.torch import utils
from graph_net.torch.decompose_util import convert_to_submodules_graph
from graph_net.torch.extractor import GraphExtractor as BuiltinGraphExtractor


class GraphExtractor:
    def __init__(
        self,
        config_str: str,
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
        self.config = self.make_config(**self.convert_to_dict(config_str))

    def make_config(
        self,
        split_positions=(),
        group_head_and_tail=False,
        chain_style=False,
        output_dir="./tmp/naive_decomposer_dir",
    ):
        for pos in split_positions:
            assert isinstance(
                pos, int
            ), f"split_positions should be list of int, {split_positions=}"
        return {
            "split_positions": split_positions,
            "group_head_and_tail": group_head_and_tail,
            "chain_style": chain_style,
            "output_dir": output_dir,
        }

    def __call__(self, gm: torch.fx.GraphModule, sample_inputs):
        config = {
            k: v
            for k, v in self.config.items()
            if k in {"split_positions", "group_head_and_tail", "chain_style"}
        }
        rewrited_gm = convert_to_submodules_graph(
            gm,
            submodule_hook=self.get_naive_decomposer_extractor,
            **config,
        )
        return rewrited_gm

    def get_naive_decomposer_extractor(self, submodule, seq_no):
        return NaiveDecomposerExtractor(self, submodule, seq_no)

    def convert_to_dict(self, config_str):
        if config_str is None:
            return {}
        config_str = base64.b64decode(config_str).decode("utf-8")
        config = json.loads(config_str)
        assert isinstance(config, dict), f"config should be a dict. {config_str=}"
        return config


class NaiveDecomposerExtractor(torch.nn.Module):
    def __init__(self, parent_graph_extractor, submodule, seq_no):
        super().__init__()
        self.parent_graph_extractor = parent_graph_extractor
        self.submodule = submodule
        self.seq_no = seq_no
        self.extracted = False
        name = f"{parent_graph_extractor.name}_{self.seq_no}"
        self.builtin_extractor = BuiltinGraphExtractor(
            name=name,
            dynamic=False,
            mut_graph_codes=[],
            placeholder_auto_rename=parent_graph_extractor.placeholder_auto_rename,
            workspace_path=self.parent_graph_extractor.config["output_dir"],
        )

    def forward(self, *args):
        if not self.extracted:
            self.builtin_extractor(self.submodule, args)
            self.extracted = True
        return self.submodule(*args)
