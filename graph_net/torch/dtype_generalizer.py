"""
DtypeGeneralizer: Data type generalization for samples.

This module provides two steps:
1. InitDataTypeGeneralizationPasses: Test and write pass names to graph_net.json
2. ApplyDataTypeGeneralizationPasses: Read pass names and generate new samples

When a sample needs to be converted to low precision, it reads the pass names
from graph_net.json and applies them to generate new samples.
"""

import copy
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch.fx as fx

from graph_net.graph_net_json_file_util import (
    kDataTypeGeneralizationPasses,
    kDtypeGeneralizationTargetDtype,
    kDtypeGeneralizationPrecision,
    kDtypeGeneralizationGenerated,
    update_json,
)
from graph_net.torch.constraint_util import RunModelPredicator
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)
from graph_net.torch.fx_graph_serialize_util import serialize_graph_module_to_str
from graph_net.torch.dtype_gen_passes.pass_mgr import get_dtype_generalization_pass
from graph_net.torch import utils
from graph_net.imp_util import load_module

from torch.fx.passes.shape_prop import ShapeProp
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module

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
    Step 1: Initialize data type generalization passes for a computation graph.

    This class tests which dtype generalization passes work for a given graph
    and writes the successful pass names to graph_net.json.

    Config format:
        {
            "dtype_list": ["float16", "bfloat16"],
            "model_path_prefix": "",
        }
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dtype_list = config.get("dtype_list", ["float16", "bfloat16"])
        self.model_path_prefix = config.get("model_path_prefix", "")

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
            model_path: Path to the model directory (may be relative to model_path_prefix)
        """
        # Apply model_path_prefix if provided
        if self.model_path_prefix:
            model_path = str(Path(self.model_path_prefix) / model_path)

        # Parse the computation graph
        # traced_model = parse_immutable_model_path_into_sole_graph_module(model_path)
        module, inputs = get_torch_module_and_inputs(model_path)
        traced_model = parse_sole_graph_module(module, inputs)
        ShapeProp(traced_model).propagate(*inputs)

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
        Test which dtype generalization passes work for this graph.

        Args:
            model_path: Path to model
            traced_model: Traced FX GraphModule

        Returns:
            List of pass names that work (pass file names without .py extension)
        """
        working_passes = []

        for dtype in self.dtype_list:
            # Pass name directly corresponds to file name (without .py)
            pass_name = f"dtype_generalization_pass_{dtype}"

            try:
                # Try to load and apply the pass
                dtype_pass_class = get_dtype_generalization_pass(pass_name)
                dtype_pass = dtype_pass_class()

                # Check if pass is needed
                if not dtype_pass.need_rewrite(traced_model):
                    continue

                # Apply the pass
                gm_copy = copy.deepcopy(traced_model)
                gm_copy = dtype_pass.rewrite(gm_copy)

                # Try to run the modified graph
                if self._test_graph_runnable(model_path, gm_copy, dtype):
                    working_passes.append(pass_name)
                    logging.info(f"Pass {pass_name} works for {model_path}")

            except (RuntimeError, ValueError, TypeError) as e:
                logging.warning(f"Pass {pass_name} failed: {e}")
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
        update_json(model_path, kDataTypeGeneralizationPasses, dtype_pass_names)


class ApplyDataTypeGeneralizationPasses:
    """
    Step 2: Apply data type generalization passes to generate new samples.

    This class reads the pass names from graph_net.json (written by Step 1)
    and applies them to generate low-precision samples.

    Config format:
        {
            "output_dir": "/path/to/output",
            "model_path_prefix": "",
            "model_runnable_predicator_filepath": "...",
            "model_runnable_predicator_class_name": "...",
            "model_runnable_predicator_config": {...},
        }
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get("output_dir")
        if not self.output_dir:
            raise ValueError("output_dir is required in config")

        self.model_path_prefix = config.get("model_path_prefix", "")

        # model_runnable_predicator is required to ensure generated code is runnable
        if "model_runnable_predicator_filepath" not in config:
            raise ValueError(
                "model_runnable_predicator_filepath is required in config. "
                "Generated code must be validated."
            )
        self.model_runnable_predicator = self._make_model_runnable_predicator(config)

    def _make_model_runnable_predicator(self, config: Dict[str, Any]):
        """Create model runnable predicator from config."""
        module = load_module(config["model_runnable_predicator_filepath"])
        cls = getattr(
            module,
            config.get("model_runnable_predicator_class_name", "RunModelPredicator"),
        )
        predicator_config = config.get("model_runnable_predicator_config", {})
        return cls(predicator_config)

    def __call__(self, model_path: str) -> List[str]:
        """
        Apply dtype passes to generate new samples.

        Args:
            model_path: Path to the original model directory (may be relative to model_path_prefix)

        Returns:
            List of generated sample directories
        """
        # Apply model_path_prefix if provided
        if self.model_path_prefix:
            model_path = str(Path(self.model_path_prefix) / model_path)

        # Read pass names from graph_net.json
        dtype_pass_names = self._read_dtype_pass_names(model_path)

        if not dtype_pass_names:
            logging.warning(f"No dtype passes found in {model_path}/graph_net.json")
            return []

        # Parse the computation graph
        traced_model = parse_immutable_model_path_into_sole_graph_module(model_path)

        # Generate samples for each pass
        generated_samples = []
        for pass_name in dtype_pass_names:
            try:
                sample_dir = self._apply_pass_and_generate(
                    model_path, traced_model, pass_name
                )
                generated_samples.append(sample_dir)
                logging.info(f"Generated sample: {sample_dir}")
            except Exception as e:
                logging.error(f"Failed to apply pass {pass_name}: {e}")
                continue

        return generated_samples

    def _read_dtype_pass_names(self, model_path: str) -> List[str]:
        """
        Read dtype pass names from graph_net.json.

        Args:
            model_path: Path to model directory

        Returns:
            List of pass names
        """
        graph_net_json_path = Path(model_path) / "graph_net.json"

        if not graph_net_json_path.exists():
            return []

        with open(graph_net_json_path, "r") as f:
            metadata = json.load(f)

        return metadata.get(kDataTypeGeneralizationPasses, [])

    def _apply_pass_and_generate(
        self, model_path: str, traced_model: fx.GraphModule, pass_name: str
    ) -> str:
        """
        Apply a specific pass and generate a new sample.

        Args:
            model_path: Original model path
            traced_model: Original traced model
            pass_name: Name of the pass file (without .py extension),
                       e.g., "dtype_generalization_pass_float16"

        Returns:
            Path to the generated sample directory
        """
        # Pass name directly corresponds to file name (without .py)
        # Extract dtype from pass name for output directory naming
        if not pass_name.startswith("dtype_generalization_pass_"):
            raise ValueError(
                f"Invalid pass name: {pass_name}. "
                f"Expected format: 'dtype_generalization_pass_<dtype>'"
            )

        dtype = pass_name.replace("dtype_generalization_pass_", "")

        # Load and apply the pass
        dtype_pass_class = get_dtype_generalization_pass(pass_name)
        dtype_pass = dtype_pass_class()

        gm_copy = copy.deepcopy(traced_model)
        gm_modified = dtype_pass.rewrite(gm_copy)

        # Generate output directory
        model_name = Path(model_path).name
        output_sample_dir = Path(self.output_dir) / f"{model_name}_{dtype}"
        output_sample_dir.mkdir(parents=True, exist_ok=True)

        # Write modified model.py
        model_code = serialize_graph_module_to_str(gm_modified)
        write_code = utils.apply_templates(model_code)
        with open(output_sample_dir / "model.py", "w") as f:
            f.write(write_code)

        # Copy metadata files
        for fname in ["graph_net.json", "weight_meta.py", "input_meta.py"]:
            src = Path(model_path) / fname
            if src.exists():
                shutil.copy(src, output_sample_dir / fname)

        # Update graph_net.json with dtype information
        self._update_sample_metadata(output_sample_dir, dtype)

        # Validate generated sample (required - generated code must be runnable)
        if not self.model_runnable_predicator(str(output_sample_dir)):
            raise RuntimeError(
                f"Generated sample failed validation: {output_sample_dir}"
            )
        logging.info(f"Generated sample validated: {output_sample_dir}")

        return str(output_sample_dir)

    def _update_sample_metadata(self, sample_dir: Path, dtype: str) -> None:
        """
        Update the generated sample's metadata.

        Args:
            sample_dir: Path to generated sample directory
            dtype: Target dtype
        """
        graph_net_json_path = sample_dir / "graph_net.json"
        update_json(graph_net_json_path, kDtypeGeneralizationTargetDtype, dtype)
        update_json(graph_net_json_path, kDtypeGeneralizationPrecision, dtype)
        update_json(graph_net_json_path, kDtypeGeneralizationGenerated, True)


class MultiDtypeFilter:
    """
    Filter for graphs that cannot support dtype generalization.
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
