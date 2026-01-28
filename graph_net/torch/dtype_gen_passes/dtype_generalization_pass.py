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
import operator

AMP_CALL_FUNCTION = {
    torch.matmul,
    torch.mm,
    torch.bmm,
    operator.matmul,
    torch.nn.functional.linear,
    torch.nn.functional.conv1d,
    torch.nn.functional.conv2d,
    torch.nn.functional.conv3d,
    torch.nn.functional.scaled_dot_product_attention,
    torch.addmm,
    torch.einsum,
}

AMP_CALL_METHOD = {
    "matmul",
    "mm",
    "bmm",
}


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
                attr_name = str(node.target)
                if self.should_preserve_weight(attr_name):
                    return new_node

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

        def create_new_args(node: fx.Node) -> list:
            """new_args of node with dtype conversion if needed."""
            new_args = []

            for arg in node.args:
                if isinstance(arg, fx.Node):
                    mapped = val_map[arg]
                    if self._is_float32_tensor(arg):
                        mapped = new_graph.call_method("to", (mapped, self.torch_dtype))
                    new_args.append(mapped)
                else:
                    new_args.append(arg)
            return new_args

        def create_new_kwargs(node: fx.Node) -> dict:
            """new_kwargs of node with dtype conversion if needed."""
            new_kwargs = {}

            for k, v in node.kwargs.items():
                if isinstance(v, fx.Node):
                    mapped = val_map[v]
                    if self._is_float32_tensor(v):
                        mapped = new_graph.call_method("to", (mapped, self.torch_dtype))
                    else:
                        new_kwargs[k] = mapped
                else:
                    new_kwargs[k] = v
            return new_kwargs

        def create_call_function(node: fx.Node) -> fx.Node:
            """Create a call_function node with dtype conversion if needed."""
            if node.target not in AMP_CALL_FUNCTION:
                return new_graph.node_copy(node, lambda x: val_map[x])

            new_args = create_new_args(node)

            new_kwargs = create_new_kwargs(node)

            return new_graph.call_function(
                node.target,
                args=tuple(new_args),
                kwargs=new_kwargs,
            )

        def create_call_method(node: fx.Node) -> fx.Node:
            """Create a call_method node with dtype conversion if needed."""
            if node.target not in AMP_CALL_METHOD:
                return new_graph.node_copy(node, lambda x: val_map[x])

            new_args = create_new_args(node)

            new_kwargs = create_new_kwargs(node)

            return new_graph.call_method(
                node.target,
                tuple(new_args),
                new_kwargs,
            )

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                val_map[node] = create_placeholder(node)
            elif node.op == "get_attr":
                val_map[node] = create_get_attr(node)
            elif node.op == "call_function":
                val_map[node] = create_call_function(node)
            elif node.op == "call_method":
                val_map[node] = create_call_method(node)
            else:
                val_map[node] = new_graph.node_copy(node, lambda x: val_map.get(x, x))

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
