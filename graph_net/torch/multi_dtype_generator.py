"""
MultiDtypeGenerator: Generate low-precision samples using FX Graph passes.

This generator converts existing float32 samples to lower precision dtypes by:
1. Using FX Graph passes to modify computation graphs (dtype conversion)
2. Preserving critical parameters (batch_norm/layer_norm buffers) as float32
3. Generating separate samples for each target dtype
"""

import copy
import json
import threading
import torch
import torch.fx as fx
from pathlib import Path
from typing import Dict, Any, Optional

import graph_net.imp_util as imp_util
from graph_net.torch.extractor import GraphExtractor as BuiltinGraphExtractor
from graph_net.torch.multi_dtype_passes.autocast_wrapper_pass import (
    AutocastWrapperPass,
)
from graph_net.torch.multi_dtype_passes.dtype_conversion_pass import (
    ConcretePass as DtypeConversionPass,
)


# Weights that must remain float32 for numerical stability
FLOAT32_PRESERVED_WEIGHTS = {
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "bn_parameters_weight",  # BatchNorm scale (gamma)
    "bn_parameters_bias",  # BatchNorm bias (beta)
    "ln_parameters_weight",  # LayerNorm weight
    "ln_parameters_bias",  # LayerNorm bias
}


class RunModelDecorator:
    """
    Decorator for run_model that generates low-precision samples.

    Config format:
        {
            "name": "sample_name",
            "dtype_list": ["float16", "bfloat16"],
            "output_dir": "/path/to/output",
            "filter_path": None,  # Optional filter
            "filter_config": {}
        }
    """

    def __init__(self, config):
        self.config = self.make_config(**config)

    def __call__(self, model):
        """Apply multi-dtype generation using GraphExtractor."""
        from graph_net.torch.extractor import extract

        return extract(**self.config)(model)

    def make_config(
        self,
        name=None,
        dtype_list=None,
        output_dir=None,
        filter_path=None,
        filter_config=None,
    ):
        if name is None:
            raise ValueError("name is required")
        if dtype_list is None:
            raise ValueError("dtype_list is required")

        # Validate dtypes
        valid_dtypes = {"float16", "bfloat16", "float8"}
        for dtype in dtype_list:
            if dtype not in valid_dtypes:
                raise ValueError(
                    f"Invalid dtype: {dtype}. Must be one of {valid_dtypes}"
                )

        return {
            "name": name,
            "dynamic": False,
            "placeholder_auto_rename": False,
            "extractor_config": {
                "custom_extractor_path": __file__,
                "custom_extractor_config": {
                    "dtype_list": dtype_list,
                    "output_dir": output_dir or "./multi_dtype_output",
                    "filter_path": filter_path,
                    "filter_config": filter_config or {},
                },
            },
        }


class GraphExtractor:
    """
    Custom GraphExtractor for multi-dtype generation.
    Compatible with graph_net.torch.extractor.extract() interface.

    This extractor:
    1. Applies FX Graph passes to convert dtypes
    2. Generates separate samples for each target dtype
    3. Preserves critical weights as float32
    """

    def __init__(
        self,
        config: dict,
        name: str,
        dynamic: bool,
        mut_graph_codes=None,
        placeholder_auto_rename=False,
    ):
        self.name = name
        self.dynamic = dynamic
        self.mut_graph_codes = mut_graph_codes
        self.placeholder_auto_rename = placeholder_auto_rename
        self.config = config

        # Extract config
        self.dtype_list = config.get("dtype_list", ["float16"])
        self.output_dir = config.get("output_dir", "./multi_dtype_output")
        self.filter = self._make_filter(config)

        # Thread-safe extraction tracking
        self._lock = threading.Lock()
        self._extracted_dtypes = set()

    def __call__(self, gm: fx.GraphModule, sample_inputs):
        """
        Process FX GraphModule and generate low-precision samples.

        This is called by torch.compile as a backend.

        Args:
            gm: FX GraphModule from torch.compile
            sample_inputs: Sample inputs captured by torch.compile

        Returns:
            Original forward function (unmodified)
        """
        with self._lock:
            # Check filter
            if self.filter and not self.filter(gm, sample_inputs):
                return gm.forward

            # Generate samples for each dtype
            for dtype in self.dtype_list:
                # Skip if already extracted
                if dtype in self._extracted_dtypes:
                    continue

                self._extracted_dtypes.add(dtype)
                self._generate_single_dtype(gm, sample_inputs, dtype)

        return gm.forward

    def _generate_single_dtype(
        self, gm: fx.GraphModule, sample_inputs: tuple, dtype: str
    ) -> None:
        """
        Generate a single low-precision sample using FX Graph pass.

        Args:
            gm: Original FX GraphModule
            sample_inputs: Original sample inputs
            dtype: Target dtype (e.g., 'float16')
        """
        # Create a copy of the graph for modification
        try:
            gm_copy = copy.deepcopy(gm)
        except Exception as e:
            print(f"Warning: Failed to deepcopy graph for {dtype}: {e}")
            return

        # Apply dtype conversion pass
        dtype_pass = DtypeConversionPass(
            target_dtype=dtype, preserve_weights=FLOAT32_PRESERVED_WEIGHTS
        )

        if dtype_pass.need_rewrite(gm_copy):
            gm_copy = dtype_pass.rewrite(gm_copy)

        # Apply autocast wrapper pass
        # This ensures operators that don't support low precision can still execute
        device_type = self._detect_device_type(gm_copy)
        autocast_pass = AutocastWrapperPass(target_dtype=dtype, device_type=device_type)
        gm_copy = autocast_pass.rewrite(gm_copy)

        # Convert sample inputs to target dtype
        converted_inputs = self._convert_inputs_dtype(sample_inputs, dtype)

        # Generate output directory name
        dtype_suffix = dtype.replace("float", "f").replace("bfloat", "bf")
        dtype_name = f"{self.name}_{dtype_suffix}"

        # Use builtin extractor to save the modified graph
        extractor = BuiltinGraphExtractor(
            name=dtype_name,
            dynamic=False,
            workspace_path=self.output_dir,
        )

        extractor(gm_copy, converted_inputs)

        # Update metadata
        output_path = Path(self.output_dir) / dtype_name
        if output_path.exists():
            self._update_metadata(output_path, dtype)

    def _convert_inputs_dtype(self, sample_inputs: tuple, dtype: str) -> tuple:
        """
        Convert sample inputs to target dtype.

        Args:
            sample_inputs: Original inputs
            dtype: Target dtype string

        Returns:
            Converted inputs
        """
        if not hasattr(torch, dtype):
            raise ValueError(f"Invalid dtype: {dtype}")

        target_dtype = getattr(torch, dtype)
        converted = []

        for inp in sample_inputs:
            if isinstance(inp, torch.Tensor) and inp.dtype == torch.float32:
                converted.append(inp.to(target_dtype))
            else:
                converted.append(inp)

        return tuple(converted)

    def _detect_device_type(self, gm: fx.GraphModule) -> str:
        """
        Detect device type from graph.

        Args:
            gm: GraphModule to inspect

        Returns:
            Device type ('cuda' or 'cpu')
        """
        # Check if any node mentions cuda
        for node in gm.graph.nodes:
            if "cuda" in str(node.target).lower():
                return "cuda"
            if hasattr(node, "meta") and "tensor_meta" in node.meta:
                tensor_meta = node.meta["tensor_meta"]
                if hasattr(tensor_meta, "device"):
                    device_str = str(tensor_meta.device)
                    if "cuda" in device_str:
                        return "cuda"

        # Check if CUDA is available
        if torch.cuda.is_available():
            return "cuda"

        # Fallback to CPU
        return "cpu"

    def _update_metadata(self, output_path: Path, dtype: str) -> None:
        """
        Update graph_net.json with dtype information.

        Args:
            output_path: Path to generated sample
            dtype: Target dtype
        """
        graph_net_json = output_path / "graph_net.json"
        if not graph_net_json.exists():
            return

        try:
            with open(graph_net_json, "r") as f:
                metadata = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to read metadata: {e}")
            return

        metadata["dtype"] = dtype
        metadata["generated_by"] = "MultiDtypeGenerator"

        # Add autocast recommendation
        metadata["autocast_recommended"] = True
        metadata["autocast_dtype"] = dtype

        if "tags" not in metadata:
            metadata["tags"] = []
        if "multi_dtype_generated" not in metadata["tags"]:
            metadata["tags"].append("multi_dtype_generated")

        try:
            with open(graph_net_json, "w") as f:
                json.dump(metadata, f, indent=4)
        except IOError as e:
            print(f"Warning: Failed to write metadata: {e}")

    def _make_filter(self, config: dict) -> Optional[Any]:
        """
        Create filter if specified in config.

        Args:
            config: Configuration dictionary

        Returns:
            Filter instance or None
        """
        filter_path = config.get("filter_path")
        if filter_path is None:
            return None

        filter_config = config.get("filter_config", {})
        module = imp_util.load_module(filter_path)
        return module.GraphFilter(filter_config)


class MultiDtypeFilter:
    """
    Filter for multi-dtype samples that cannot execute properly.

    This filter can be used to exclude graphs that:
    - Contain unsupported operations for low precision
    - Have numerical stability issues
    - Cannot be converted safely
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Filter configuration
        """
        self.config = config
        self.unsupported_ops = config.get("unsupported_ops", set())

    def __call__(self, gm: fx.GraphModule, sample_inputs: tuple) -> bool:
        """
        Check if the graph should be kept.

        Args:
            gm: GraphModule to check
            sample_inputs: Sample inputs

        Returns:
            True if graph should be kept, False to filter out
        """
        # Check for unsupported operations
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                if any(unsup_op in target_str for unsup_op in self.unsupported_ops):
                    return False

        return True
