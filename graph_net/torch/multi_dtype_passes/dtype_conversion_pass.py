"""
Concrete implementation of dtype conversion pass.

This pass converts tensor dtypes in FX Graph by:
1. Converting placeholder nodes (inputs) to target dtype
2. Converting get_attr nodes (weights) to target dtype, except preserved weights
3. Inserting .to(dtype) calls where needed
"""

import torch
import torch.fx as fx
from graph_net.torch.multi_dtype_passes.pass_base import DtypeConversionPass


class ConcretePass(DtypeConversionPass):
    """
    FX Graph pass that converts dtypes of tensors.

    This pass modifies the graph to:
    - Convert input tensors to target dtype
    - Convert weight tensors to target dtype (except preserved weights)
    - Insert dtype conversion nodes where necessary
    """

    def get_pass_name(self) -> str:
        return f"dtype_conversion_{self.target_dtype}"

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

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                # Copy placeholder node
                new_node = new_graph.node_copy(node, lambda x: val_map.get(x, x))

                # Insert dtype conversion if needed
                if self._is_float32_tensor(node):
                    converted_node = new_graph.call_method(
                        "to",
                        args=(new_node, self.torch_dtype),
                    )
                    val_map[node] = converted_node
                else:
                    val_map[node] = new_node

            elif node.op == "get_attr":
                # Copy get_attr node
                new_node = new_graph.node_copy(node, lambda x: val_map.get(x, x))

                # Insert dtype conversion if needed and not preserved
                attr_name = str(node.target)
                if self._is_float32_tensor(node) and not self.should_preserve_weight(
                    attr_name
                ):
                    converted_node = new_graph.call_method(
                        "to",
                        args=(new_node, self.torch_dtype),
                    )
                    val_map[node] = converted_node
                else:
                    val_map[node] = new_node

            else:
                # Copy other nodes
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
        # conservatively assume they might be float32
        # This is safe because:
        # 1. .to() on non-float tensors is a no-op for most cases
        # 2. Integer tensors (like input_ids) won't be affected
        if node.op in ("placeholder", "get_attr"):
            # Check type annotation if available
            if node.type is not None:
                type_str = str(node.type)
                # Only return True if it's explicitly a floating point tensor
                if "Tensor" in type_str and "int" not in type_str.lower():
                    return True

        return False
