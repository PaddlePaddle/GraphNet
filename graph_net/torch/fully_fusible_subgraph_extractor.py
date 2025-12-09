import os
import torch
from pathlib import Path
import tempfile
import shutil
from graph_net.torch.graph_decomposer import NaiveDecomposerExtractor
from graph_net.torch.fully_fusible_graph_predicator import (
    FullyFusibleSubGraphPredicator,
)
import logging

logger = logging.getLogger(__name__)


class FullyFusibleSubgraphExtractor:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)

    def _make_config(
        self,
        nn_module_fully_fusible_decorator_path,
        nn_module_fully_fusible_decorator_class_name,
        nn_module_fully_fusible_decorator_config=None,
        output_dir=None,
        resume: bool = True,
        max_step=8,
        min_step=2,
        max_nodes=32,
        model_path_prefix="",
    ):
        return {
            "output_dir": output_dir,
            "resume": resume,
            "nn_module_fully_fusible_decorator_path": nn_module_fully_fusible_decorator_path,
            "nn_module_fully_fusible_decorator_class_name": nn_module_fully_fusible_decorator_class_name,
            "nn_module_fully_fusible_decorator_config": nn_module_fully_fusible_decorator_config,
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

    def _copy_from_tmp_dir_to_output_dir(
        self, temp_dir: str, rel_model_path: str
    ) -> str:
        subdirs = list(Path(temp_dir).iterdir())
        assert len(subdirs) == 1
        temp_dir = str(subdirs[0])
        target_path = os.path.join(
            self.config["output_dir"],
            rel_model_path,
        )
        os.makedirs(target_path, exist_ok=True)
        shutil.copytree(temp_dir, target_path, dirs_exist_ok=True)
        return target_path

    def _build_decompose_config(
        self, temp_dir: str, start_pos: int, end_pos: int
    ) -> dict:
        model_path_prefix = self.config["model_path_prefix"]
        decomposer_config = {
            "model_path_prefix": model_path_prefix,
            "output_dir": temp_dir,
            "split_positions": [start_pos, end_pos],
            "group_head_and_tail": False,
        }
        return decomposer_config

    def _get_fully_fusible_subgraph_predicator(self, model_path):
        config = {
            "model_path": model_path,
            "nn_module_fully_fusible_decorator_path": self.config[
                "nn_module_fully_fusible_decorator_path"
            ],
            "nn_module_fully_fusible_decorator_class_name": self.config[
                "nn_module_fully_fusible_decorator_class_name"
            ],
            "nn_module_fully_fusible_decorator_config": self.config[
                "nn_module_fully_fusible_decorator_config"
            ],
        }
        return FullyFusibleSubGraphPredicator(config)

    def _is_model_path_handled(self, rel_model_path):
        model_path = Path(self.config["output_dir"]) / rel_model_path
        return model_path.exists() and len(list(model_path.iterdir())) > 0

    def __call__(self, rel_model_path):
        if self.config["resume"] and self._is_model_path_handled(rel_model_path):
            return
        torch.cuda.empty_cache()
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        fully_fusible_subgraph_predicator = self._get_fully_fusible_subgraph_predicator(
            model_path
        )
        for start_pos, end_pos in self._get_sub_ranges():
            success = fully_fusible_subgraph_predicator(start_pos, end_pos)
            if not success:
                continue
            with tempfile.TemporaryDirectory(
                prefix="_find_fusible_subgraph_"
            ) as temp_dir:
                decomposer_config = self._build_decompose_config(
                    temp_dir, start_pos, end_pos
                )
                naive_graph_decomposer = NaiveDecomposerExtractor(decomposer_config)
                naive_graph_decomposer(rel_model_path)
                fully_fusible_destination_path = self._copy_from_tmp_dir_to_output_dir(
                    temp_dir, rel_model_path
                )
                print(f"{fully_fusible_destination_path=}")
            return
