import os
import torch
import json
import shutil
from typing import Union, Callable
from graph_net.torch import utils
from graph_net.torch.decompose_util import convert_to_submodules_graph
from graph_net.torch.extractor import GraphExtractor as BuiltinGraphExtractor


class GraphExtractor:
    def __init__(
        self, name, dynamic, mut_graph_codes=None, placeholder_auto_rename=False
    ):
        self.subgraph_counter = 0
        self.name = name
        self.dynamic = dynamic
        self.mut_graph_codes = mut_graph_codes
        self.placeholder_auto_rename = placeholder_auto_rename
        self.workspace_path = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
        if not self.workspace_path:
            raise EnvironmentError(
                "Environment variable 'GRAPH_NET_EXTRACT_WORKSPACE' is not set."
            )
        split_pos_str = os.environ.get("GRAPH_NET_NAIVE_DECOMPOSER_SPLIT_POS")
        if split_pos_str is None:
            raise EnvironmentError(
                "Environment variable 'GRAPH_NET_NAIVE_DECOMPOSER_SPLIT_POS' is not set."
            )
        self.split_positions = [int(pos) for pos in split_pos_str.split(",")]

    def __call__(self, gm: torch.fx.GraphModule, sample_inputs):
        return convert_to_submodules_graph(
            gm,
            split_positions=self.split_positions,
            submodule_hook=self.get_naive_decomposer_extractor,
            group_head_and_tail=False,
        )

    def get_naive_decomposer_extractor(self, submodule, seq_no):
        return NaiveDecomposerExtractor(self, submodule, seq_no)


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
        )

    def forward(self, *args):
        if not self.extracted:
            self.builtin_extractor(self.submodule, args)
            self.extracted = True
        return self.submodule(*args)
