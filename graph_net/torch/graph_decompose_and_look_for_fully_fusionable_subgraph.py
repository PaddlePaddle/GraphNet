import os
import torch
import copy
from graph_net.torch.decompose_util import convert_to_submodules_graph
from graph_net.torch.extractor import GraphExtractor as BuiltinGraphExtractor
import graph_net.imp_util as imp_util


class GraphExtractor:
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
        self.last_post_process_result = False
        self.decompose_finished = False

    def make_config(
        self,
        split_positions=(),
        group_head_and_tail=False,
        chain_style=False,
        output_dir="./tmp/naive_decomposer_dir",
        filter_path=None,
        filter_config=None,
        post_extract_process_path=None,
        post_extract_process_class_name=None,
        max_step=8,
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
            "filter_path": filter_path,
            "filter_config": filter_config if filter_config is not None else {},
            "post_extract_process_path": post_extract_process_path,
            "post_extract_process_class_name": post_extract_process_class_name,
            "max_step": max_step,
        }

    def __call__(self, gm: torch.fx.GraphModule, sample_inputs):
        for i in range(self.config["max_step"], -1, -1):
            start_pos = 0
            for start_pos in range(32 - i):
                end_pos = start_pos + i
                self.config["split_positions"] = [start_pos, end_pos]
                self.last_post_process_result = False
                config = {
                    k: v
                    for k, v in self.config.items()
                    if k in {"split_positions", "group_head_and_tail", "chain_style"}
                }
                print(f"Current Config: {config['split_positions']}")
                gm_to_process = copy.deepcopy(gm)
                rewrited_gm = convert_to_submodules_graph(
                    gm_to_process,
                    submodule_hook=self.get_naive_decomposer_extractor,
                    **config,
                )
                try:
                    rewrited_gm(*sample_inputs)
                except Exception as e:
                    print(f"Run failed: {e}")
                    self.last_post_process_result = False
                if self.last_post_process_result and self.decompose_finished:
                    print("Success! Subgraph is fully fusionable.")
                    break
            if self.last_post_process_result:
                break
        return rewrited_gm

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
        self.model_name = name
        self.builtin_extractor = BuiltinGraphExtractor(
            name=name,
            dynamic=False,
            mut_graph_codes=[],
            placeholder_auto_rename=parent_graph_extractor.placeholder_auto_rename,
            workspace_path=self.parent_graph_extractor.config["output_dir"],
        )
        self.filter = self.make_filter(self.parent_graph_extractor.config)
        self.post_extract_process = self.make_post_extract_process(
            self.parent_graph_extractor.config
        )

    def forward(self, *args):
        if not self.extracted:
            if self.need_extract(self.submodule, args):
                self.builtin_extractor(self.submodule, args)
                if self._post_extract_process() and self.seq_no == 1:
                    self.parent_graph_extractor.last_post_process_result = True
                    print("biggest fully fusionable subgraph found!!", self.model_name)
                if self.seq_no == len(
                    self.parent_graph_extractor.config["split_positions"]
                ):
                    self.parent_graph_extractor.decompose_finished = True
            self.extracted = True
        return self.submodule(*args)

    def need_extract(self, gm, sample_inputs):
        if self.filter is None:
            return True
        return self.filter(gm, sample_inputs)

    def _post_extract_process(self):
        model_path = os.path.join(
            self.parent_graph_extractor.config["output_dir"], self.model_name
        )
        if self.post_extract_process:
            result = self.post_extract_process(model_path)
        else:
            result = True  # 默认通过
        return result

    def make_filter(self, config):
        if config["filter_path"] is None:
            return None
        module = imp_util.load_module(config["filter_path"])
        return module.GraphFilter(config["filter_config"])

    def make_post_extract_process(self, config):
        if config["post_extract_process_path"] is None:
            return None
        module = imp_util.load_module(config["post_extract_process_path"])
        return module.GraphFullyFusionable(config["post_extract_process_path"])
