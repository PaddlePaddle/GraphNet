import os
import graph_net
import tempfile
import shutil
from graph_net.torch import constraint_util
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module
import logging

logger = logging.getLogger(__name__)


class FullyFusibleSubgraphExtractor:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)

    def _make_config(
        self,
        output_dir=None,
        split_positions=(),
        group_head_and_tail=False,
        chain_style=False,
        max_step=8,
        min_step=2,
        max_nodes=32,
        model_path_prefix="",
    ):
        for pos in split_positions:
            assert isinstance(
                pos, int
            ), f"split_positions should be list of int, {split_positions=}"
        return {
            "output_dir": output_dir,
            "split_positions": split_positions,
            "group_head_and_tail": group_head_and_tail,
            "chain_style": chain_style,
            "max_step": max_step,
            "min_step": min_step,
            "max_nodes": max_nodes,
            "model_path_prefix": model_path_prefix,
        }

    def _get_sub_ranges(self):
        assert self.config["min_step"] >= 1, "min_step must be greater than 1。"
        assert (
            self.config["max_step"] >= self.config["min_step"]
        ), "max_step must be greater than min_step。"
        for step in reversed(
            range(self.config["min_step"], self.config["max_step"] + 1)
        ):
            assert (
                self.config["min_step"] <= step <= self.config["max_step"]
            ), "Internal error: step exceeds configuration range."
            for start_pos in range(self.config["max_nodes"] - step):
                end_pos = start_pos + step
                assert (
                    0 <= start_pos < end_pos <= self.config["max_nodes"]
                ), f"Invalid range generated: start={start_pos}, end={end_pos}, max={self.config['max_nodes']}"
                yield start_pos, end_pos

    def _handle_success(self, temp_dir: str, start_pos: int, end_pos: int) -> str:
        target_name = f"_start{start_pos}_end{end_pos}"
        target_path = os.path.join(
            self.config["output_dir"],
            target_name,
        )
        os.makedirs(target_path, exist_ok=True)
        shutil.move(temp_dir, target_path)
        return target_path

    def _build_decompose_config(
        self, temp_dir: str, start_pos: int, end_pos: int, model_path_prefix
    ) -> dict:
        graph_net_root = os.path.dirname(graph_net.__file__)

        check_fusible_config = {
            "handler_path": f"{graph_net_root}/torch/naive_graph_decomposer.py",
            "handler_class_name": "NaiveDecomposerExtractor",
            "handler_config": {
                "model_path_prefix": model_path_prefix,
                "output_dir": temp_dir,
                "split_positions": [start_pos, end_pos],
                "group_head_and_tail": False,
                "post_extract_process_path": f"{graph_net_root}/torch/post_extract_process_count_kernels.py",
                "post_extract_process_class_name": "GraphFullyFusible",
            },
        }
        return check_fusible_config

    def __call__(self, rel_model_path):
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        module, inputs = get_torch_module_and_inputs(model_path)
        gm = parse_sole_graph_module(module, inputs)
        for start_pos, end_pos in self._get_sub_ranges():
            with tempfile.TemporaryDirectory(
                prefix="_find_fusible_subgraph_"
            ) as temp_dir:
                check_fusible_config = self._build_decompose_config(
                    temp_dir, start_pos, end_pos, self.config["model_path_prefix"]
                )
                predicator = constraint_util.FusibleSubgraphPredicator(
                    check_fusible_config
                )
                success = predicator(model_path)
                if success:
                    target_path = self._handle_success(temp_dir, start_pos, end_pos)
                    print(
                        f"SUCCESS in finding the biggest fully fusible subgraph. Result saved to: {target_path}"
                    )
                    break
        else:
            logger.warning("fail to find fully fusible subgraph")
        return gm.forward
