"""
MultiDtypeGenerator: Initialize data type generalization passes for samples.

This module provides an initializer that tests which dtype conversion passes
work for a given computation graph, then writes the pass names to graph_net.json.

When the sample needs to be converted to low precision, it reads the pass names
from graph_net.json and applies them.
"""

import copy
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch.fx as fx

from graph_net.graph_net_json_file_util import kDataTypeGeneralizationPasses
from graph_net.torch.constraint_util import RunModelPredicator
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)
from graph_net.torch.fx_graph_serialize_util import serialize_graph_module_to_str
from graph_net.torch.multi_dtype_passes.pass_mgr import get_dtype_conversion_pass
from graph_net.torch import utils


# Weights that must remain float32 for numerical stability
FLOAT32_PRESERVED_WEIGHTS = {
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "bn_parameters_weight",
    "bn_parameters_bias",
    "ln_parameters_weight",
    "ln_parameters_bias",
}


class InitDataTypeGeneralizationPasses:
    """
    Initialize data type generalization passes for a computation graph.

    This class tests which dtype conversion passes work for a given graph
    and writes the successful pass names to graph_net.json.

    Config format:
        {
            "dtype_list": ["float16", "bfloat16"],
        }
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dtype_list = config.get("dtype_list", ["float16", "bfloat16"])

        # Validate dtypes
        valid_dtypes = {"float16", "bfloat16", "float8"}
        for dtype in self.dtype_list:
            if dtype not in valid_dtypes:
                raise ValueError(
                    f"Invalid dtype: {dtype}. Must be one of {valid_dtypes}"
                )

    def __call__(self, model_path: str) -> None:
        """
        Initialize dtype passes for the given model.

        Args:
            model_path: Path to the model directory
        """
        # Parse the computation graph
        traced_model = parse_immutable_model_path_into_sole_graph_module(model_path)

        # Test which dtype passes work
        dtype_pass_names = self._test_dtype_passes(model_path, traced_model)

        # Save pass names to graph_net.json
        if dtype_pass_names:
            self._save_dtype_pass_names(dtype_pass_names, model_path)
            logging.info(f"Saved {len(dtype_pass_names)} dtype passes to {model_path}")
        else:
            logging.warning(f"No dtype passes applicable for {model_path}")

    def _test_dtype_passes(
        self, model_path: str, traced_model: fx.GraphModule
    ) -> List[str]:
        """
        Test which dtype conversion passes work for this graph.

        Args:
            model_path: Path to model
            traced_model: Traced FX GraphModule

        Returns:
            List of pass names that work
        """
        working_passes = []

        for dtype in self.dtype_list:
            pass_name = "dtype_conversion_pass"  # Full pass file name without .py

            try:
                # Try to apply the pass
                dtype_pass_class = get_dtype_conversion_pass(pass_name)
                dtype_pass = dtype_pass_class(
                    target_dtype=dtype,
                    preserve_weights=FLOAT32_PRESERVED_WEIGHTS,
                )

                # Check if pass is needed
                if not dtype_pass.need_rewrite(traced_model):
                    continue

                # Apply the pass
                gm_copy = copy.deepcopy(traced_model)
                gm_copy = dtype_pass.rewrite(gm_copy)

                # Try to run the modified graph
                if self._test_graph_runnable(model_path, gm_copy, dtype):
                    working_passes.append(f"{pass_name}_{dtype}")
                    logging.info(f"Pass {pass_name}_{dtype} works for {model_path}")

            except (RuntimeError, ValueError, TypeError) as e:
                logging.warning(f"Pass {pass_name}_{dtype} failed: {e}")
                continue

        return working_passes

    def _test_graph_runnable(
        self, model_path: str, gm: fx.GraphModule, dtype: str
    ) -> bool:
        """
        Test if the modified graph can run.

        Args:
            model_path: Original model path
            gm: Modified GraphModule
            dtype: Target dtype

        Returns:
            True if graph runs successfully
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                model_code = serialize_graph_module_to_str(gm)
                write_code = utils.apply_templates(model_code)

                with open(f"{tmpdir}/model.py", "w") as f:
                    f.write(write_code)

                for fname in ["graph_net.json", "weight_meta.py", "input_meta.py"]:
                    src = Path(model_path) / fname
                    if src.exists():
                        shutil.copy(src, Path(tmpdir) / fname)

                predictor = RunModelPredicator({"use_dummy_inputs": True})
                return predictor(tmpdir)

            except (IOError, RuntimeError, ValueError) as e:
                logging.debug(f"Graph test failed for {dtype}: {e}")
                return False

    def _save_dtype_pass_names(
        self, dtype_pass_names: List[str], model_path: str
    ) -> None:
        """
        Save dtype pass names to graph_net.json atomically.

        Args:
            dtype_pass_names: List of working pass names
            model_path: Path to model directory
        """
        graph_net_json_path = Path(model_path) / "graph_net.json"

        with open(graph_net_json_path, "r") as f:
            metadata = json.load(f)

        metadata[kDataTypeGeneralizationPasses] = dtype_pass_names

        # Atomic write: write to temp file then rename
        temp_path = graph_net_json_path.with_suffix(".json.tmp")
        with open(temp_path, "w") as f:
            json.dump(metadata, f, indent=4)

        temp_path.replace(graph_net_json_path)


class MultiDtypeFilter:
    """
    Filter for graphs that cannot support dtype conversion.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.unsupported_ops = config.get("unsupported_ops", set())

    def __call__(self, gm: fx.GraphModule, sample_inputs: tuple) -> bool:
        """
        Check if graph can be converted to low precision.

        Args:
            gm: GraphModule to check
            sample_inputs: Sample inputs

        Returns:
            True if graph can be converted
        """
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                if any(unsup_op in target_str for unsup_op in self.unsupported_ops):
                    return False
        return True
