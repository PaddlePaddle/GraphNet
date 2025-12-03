import os
import torch
import graph_net
import tempfile
from graph_net.torch import constraint_util


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

    def make_config(
        self,
        split_positions=(),
        group_head_and_tail=False,
        chain_style=False,
        max_step=8,
        min_step=2,
        max_nodes=32,
    ):
        for pos in split_positions:
            assert isinstance(
                pos, int
            ), f"split_positions should be list of int, {split_positions=}"
        return {
            "split_positions": split_positions,
            "group_head_and_tail": group_head_and_tail,
            "chain_style": chain_style,
            "max_step": max_step,
            "min_step": min_step,
            "max_nodes": max_nodes,
        }

    def _get_sub_ranges(self):
        for step in reversed(
            range(self.config["min_step"], self.config["max_step"] + 1)
        ):
            for start_pos in range(self.config["max_nodes"] - step):
                end_pos = start_pos + step
                yield start_pos, end_pos

    def __call__(self, gm: torch.fx.GraphModule, sample_inputs):
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="_check_fusable_")
        temp_output_dir = temp_dir_obj.name
        found_fusable_subgraph = False
        print(f"Using temp output dir: {temp_output_dir}")
        for start_pos, end_pos in self._get_sub_ranges():
            self.config["split_positions"] = [start_pos, end_pos]
            print("current split_positions:", self.config["split_positions"])
            graph_net_root = os.path.dirname(graph_net.__file__)
            model_path = os.path.join(
                graph_net_root, "..", "samples", "timm", self.name
            )
            check_fusable_config = {
                "decorator_path": f"{graph_net_root}/torch/extractor.py",
                "decorator_config": {
                    "name": f"{self.name}",
                    "custom_extractor_path": f"{graph_net_root}/torch/naive_graph_decomposer.py",
                    "custom_extractor_config": {
                        "output_dir": temp_output_dir,
                        "split_positions": self.config["split_positions"],
                        "group_head_and_tail": False,
                        "filter_path": f"{graph_net_root}/torch/naive_subgraph_filter.py",
                        "filter_config": {},
                        "post_extract_process_path": f"{graph_net_root}/torch/post_extract_process_count_kernels.py",
                        "post_extract_process_class_name": "GraphFullyFusable",
                    },
                },
            }
            success = constraint_util.RunModelPredicator(check_fusable_config)(
                model_path
            )
            if success:
                found_fusable_subgraph = True
                temp_dir_obj.cleanup = lambda: None
                print(
                    f"SUCCESS in finding the biggest fully fusable subgraph saved in: {temp_output_dir}."
                )
                break
            else:
                print("Failed attempt. clean up the workspace and continue the search.")
                temp_dir_obj.cleanup()
                continue
        if not found_fusable_subgraph:
            print("No fusable subgraph found")
        return gm
