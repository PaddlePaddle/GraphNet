import os
import torch
import sys
import graph_net


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
        output_dir="./tmp/naive_decomposer_dir",
        filter_path=None,
        filter_config=None,
        post_extract_process_path=None,
        post_extract_process_class_name=None,
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
            "output_dir": output_dir,
            "filter_path": filter_path,
            "filter_config": filter_config if filter_config is not None else {},
            "post_extract_process_path": post_extract_process_path,
            "post_extract_process_class_name": post_extract_process_class_name,
            "max_step": max_step,
            "min_step": min_step,
            "max_nodes": max_nodes,
        }

    def _get_sub_ranges(self):
        kMinLenOps = self.config["min_step"]
        num_ops = self.config["max_nodes"]
        for length in reversed(range(kMinLenOps, num_ops)):
            for start_pos in range(num_ops - length):
                end_pos = start_pos + length
                yield start_pos, end_pos

    def __call__(self, gm: torch.fx.GraphModule, sample_inputs):
        import json
        import base64

        for start_pos, end_pos in self._get_sub_ranges():
            self.config["split_positions"] = [start_pos, end_pos]
            print("current split_positions:", self.config["split_positions"])
            graph_net_root = os.path.dirname(graph_net.__file__)
            model_path = f"{graph_net_root}/../samples//timm/{self.name}"
            check_fusable_config = {
                "decorator_path": f"{graph_net_root}/torch/extractor.py",
                "decorator_config": {
                    "name": f"{self.name}",
                    "custom_extractor_path": f"{graph_net_root}/torch/naive_graph_decomposer.py",
                    "custom_extractor_config": {
                        "output_dir": "/tmp/naive_decompose_workspace",
                        "split_positions": self.config["split_positions"],
                        "group_head_and_tail": False,
                        "filter_path": f"{graph_net_root}/torch/naive_subgraph_filter.py",
                        "filter_config": {},
                        "post_extract_process_path": f"{graph_net_root}/torch/post_extract_process_count_kernels.py",
                        "post_extract_process_class_name": "GraphFullyFusable",
                    },
                },
            }
            json_string = json.dumps(check_fusable_config)
            json_bytes = json_string.encode("utf-8")
            b64_encoded_bytes = base64.b64encode(json_bytes)
            checker_config = b64_encoded_bytes.decode("utf-8")
            cmd = f"{sys.executable} -m graph_net.torch.run_model --model-path {model_path} --decorator-config '{checker_config}'"
            res_code = os.system(cmd)
            if res_code == 0:
                print("find the biggest fully fusable subgraph")
                break
            else:
                continue
        return gm
