"""
Concrete implementation of dtype generalization pass.

This pass converts tensor dtypes in FX Graph by:
1. Converting placeholder nodes (inputs) to target dtype
2. Converting get_attr nodes (weights) to target dtype, except preserved weights
3. Inserting .to(dtype) calls where needed
"""

import torch
import torch.fx as fx
from graph_net.torch.dtype_gen_passes.pass_base import DtypeGeneralizationPass


class ConcretePass(DtypeGeneralizationPass):
    """
    FX Graph pass that converts dtypes of tensors.

    This pass modifies the graph to:
    - Convert input tensors to target dtype
    - Convert weight tensors to target dtype (except preserved weights)
    - Insert dtype conversion nodes where necessary
    """

    def get_pass_name(self) -> str:
        return f"dtype_generalization_{self.target_dtype}"

    def need_rewrite(self, gm: fx.GraphModule) -> bool:
        """
        Check if graph has float32 tensors that need conversion.
        """
        for node in gm.graph.nodes:
            if self._node_need_rewrite(node):
                return True
        return False

    def _node_need_rewrite(self, node: fx.Node) -> bool:
        """
        Check if a specific node needs dtype conversion.

        Args:
            node: FX Node to check

        Returns:
            True if node should be rewritten
        """
        # Check placeholder nodes (inputs)
        if node.op == "placeholder":
            return self._is_float32_tensor(node)

        # Check get_attr nodes (weights)
        if node.op == "get_attr":
            if self._is_float32_tensor(node):
                # Only rewrite if not in preserve list
                attr_name = str(node.target)
                return not self.should_preserve_weight(attr_name)

        return False

    def rewrite(self, gm: fx.GraphModule) -> fx.GraphModule:
        """
        Rewrite the graph to convert dtypes.

        Strategy:
        1. For each placeholder (input), insert .to(target_dtype) after it
        2. For each get_attr (weight), insert .to(target_dtype) if not preserved
        3. Update the graph and recompile
        """
        new_graph = fx.Graph()
        val_map = {}

        def create_placeholder(node: fx.Node) -> fx.Node:
            """Create a placeholder node with dtype conversion if needed."""
            new_node = new_graph.node_copy(node, lambda x: val_map.get(x, x))
            if self._is_float32_tensor(node):
                return new_graph.call_method("to", args=(new_node, self.torch_dtype))
            return new_node

        def create_get_attr(node: fx.Node) -> fx.Node:
            """Create a get_attr node with dtype conversion if needed."""
            new_node = new_graph.node_copy(node, lambda x: val_map.get(x, x))
            attr_name = str(node.target)
            if self._is_float32_tensor(node) and not self.should_preserve_weight(
                attr_name
            ):
                return new_graph.call_method("to", args=(new_node, self.torch_dtype))
            return new_node

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                val_map[node] = create_placeholder(node)
            elif node.op == "get_attr":
                val_map[node] = create_get_attr(node)
            else:
                new_node = new_graph.node_copy(node, lambda x: val_map.get(x, x))
                val_map[node] = new_node

        # Replace the graph
        gm.graph = new_graph
        gm.recompile()

        return gm

    def _is_float32_tensor(self, node: fx.Node) -> bool:
        """
        Check if a node represents a float32 tensor.

        Args:
            node: FX Node to check

        Returns:
            True if node is a float32 tensor
        """
        # Check tensor_meta if available (most reliable)
        if "tensor_meta" in node.meta:
            tensor_meta = node.meta["tensor_meta"]
            if hasattr(tensor_meta, "dtype"):
                return tensor_meta.dtype == torch.float32

        # For placeholder and get_attr nodes without metadata,
        # we need to be conservative and only return True if explicitly float
        if node.op in ("placeholder", "get_attr"):
            # Check type annotation if available
            if node.type is not None:
                type_str = str(node.type).lower()

                # Explicitly check for integer types - these should NOT be converted
                integer_types = ["long", "int", "short", "byte", "bool"]
                if any(int_type in type_str for int_type in integer_types):
                    return False

                # Only return True if explicitly a floating point tensor
                # Check for explicit float types: FloatTensor, float32, float16, etc.
                float_indicators = ["float", "double", "half", "bfloat"]
                if any(
                    float_indicator in type_str for float_indicator in float_indicators
                ):
                    return True

                # For generic "Tensor" without explicit dtype, be conservative
                # Don't assume it's float32 - it might be integer
                return False

        return False
