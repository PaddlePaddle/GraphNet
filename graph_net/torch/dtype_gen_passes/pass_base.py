"""
Base class for dtype generalization passes.
"""

import torch
import torch.fx as fx


class DtypeGeneralizationPass:
    """
    Base class for FX Graph passes that convert dtypes.

    Subclasses should implement:
    - need_rewrite(): Check if the graph needs dtype conversion
    - rewrite(): Apply dtype conversion to the graph
    """

    def __init__(self, target_dtype: str, preserve_weights: set = None):
        """
        Args:
            target_dtype: Target dtype (e.g., 'float16', 'bfloat16')
            preserve_weights: Set of weight name patterns to keep as float32
        """
        self.target_dtype = target_dtype
        self.preserve_weights = preserve_weights or set()

        # Validate dtype
        if not hasattr(torch, target_dtype):
            raise ValueError(f"Invalid dtype: {target_dtype}")

        self.torch_dtype = getattr(torch, target_dtype)

    def get_pass_name(self) -> str:
        """Return the name of this pass."""
        raise NotImplementedError()

    def need_rewrite(self, gm: fx.GraphModule) -> bool:
        """
        Check if the graph needs dtype conversion.

        Args:
            gm: GraphModule to check

        Returns:
            True if rewrite is needed
        """
        raise NotImplementedError()

    def rewrite(self, gm: fx.GraphModule) -> fx.GraphModule:
        """
        Apply dtype conversion to the graph.

        Args:
            gm: GraphModule to rewrite

        Returns:
            Modified GraphModule
        """
        raise NotImplementedError()

    def should_preserve_weight(self, weight_name: str) -> bool:
        """
        Check if a weight should be preserved as float32.

        Args:
            weight_name: Name of the weight parameter

        Returns:
            True if weight should remain float32
        """
        return any(pattern in weight_name for pattern in self.preserve_weights)
