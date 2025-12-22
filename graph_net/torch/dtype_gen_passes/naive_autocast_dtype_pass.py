"""
Autocast wrapper pass.

This pass wraps the forward method with torch.autocast context manager
to handle operators that don't support low-precision computation natively.
"""

import torch
import torch.fx as fx

from graph_net.torch.dtype_gen_passes.pass_base import DtypeGeneralizationPass


class AutocastWrapperPass(DtypeGeneralizationPass):
    """
    FX Graph pass that adds autocast context manager.

    This pass is applied AFTER dtype conversion to ensure operators
    that don't support low precision can still execute.
    """

    def __init__(self, target_dtype: str, device_type: str = None):
        """
        Args:
            target_dtype: Target dtype for autocast
            device_type: Device type ('cuda' or 'cpu'). If None, auto-detect.
        """
        super().__init__(target_dtype, preserve_weights=set())
        if device_type is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_type = device_type

    def get_pass_name(self) -> str:
        return f"autocast_wrapper_{self.target_dtype}"

    def need_rewrite(self, gm: fx.GraphModule) -> bool:
        """
        Always apply autocast wrapper for low-precision dtypes.
        """
        return True

    def rewrite(self, gm: fx.GraphModule) -> fx.GraphModule:
        """
        Wrap the entire graph with autocast context.

        Strategy:
        1. Create autocast context at the beginning
        2. Execute all operations within the context
        3. Exit context before return

        Note: This is a simplified approach. In practice, autocast
        is typically applied at the model level, not in FX Graph.
        For GraphNet samples, we document the need for autocast
        in metadata instead of modifying the graph.
        """
        # For GraphNet samples, we don't modify the graph structure
        # Instead, we mark in metadata that autocast should be used
        # The actual autocast wrapping happens at runtime

        # Add autocast metadata to the graph
        if not hasattr(gm, "_autocast_config"):
            gm._autocast_config = {
                "enabled": True,
                "dtype": self.target_dtype,
                "device_type": self.device_type,
            }

        return gm
