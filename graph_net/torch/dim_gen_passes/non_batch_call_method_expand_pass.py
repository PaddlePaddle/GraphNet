import torch
import torch.fx as fx
from graph_net.torch.dim_gen_passes import DimensionGeneralizationPass
from collections import namedtuple


class ConcretePass(DimensionGeneralizationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def need_rewrite(self, traced_module: fx.GraphModule) -> bool:
        if 0 in self.axes:
            return False
        if self.dim <= 1:
            return False
        return any(self._node_need_rewrite(node) for node in traced_module.graph.nodes)

    def _node_need_rewrite(self, node) -> bool:
        return (
            node.op == "call_method"
            and node.target == "expand"
            and len(node.args) >= 2
            and any(arg == self.dim for arg in node.args[1:])
        )

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        """
        Fx Pass: Replaces hardcoded constants in 'expand' ops that match an input tensor dimension
        with a dynamic 'size()' call. The primary goal is to dynamicize the batch size (axis 0).
        """
        # Create a new graph to hold the rewritten nodes
        new_graph = fx.Graph()

        # Create a map to link nodes from the old graph to nodes in the new graph
        val_map = {}

        NodeAxis = namedtuple("NodeAxis", ["node", "shape_axis"])
        last_node_axis = None

        def try_reset_last_node_axis(node, new_node):
            nonlocal last_node_axis
            node_meta = node.meta.get("tensor_meta")
            if node_meta is None:
                return
            if not hasattr(node_meta, "shape"):
                return
            for axis, dim in enumerate(node_meta.shape):
                if not (dim == self.dim and axis > 0):
                    continue
                last_node_axis = NodeAxis(node=new_node, shape_axis=axis)
                return

        def get_new_node_arg(arg):
            if not (isinstance(arg, int) and arg == self.dim):
                return val_map[arg] if arg in val_map else arg

            assert arg == self.dim

            # Use the size() method to retrieve the dynamic dimension
            size_node = new_graph.call_method(
                "size", args=(last_node_axis.node, last_node_axis.shape_axis)
            )
            return size_node

        def create_new_node(node):
            if not (self._node_need_rewrite(node) and last_node_axis is not None):
                # Copy other nodes to the new graph
                new_node = new_graph.node_copy(node, lambda x: val_map[x])
                try_reset_last_node_axis(node=node, new_node=new_node)
                return new_node

            new_expand_args = tuple(get_new_node_arg(arg) for arg in node.args)

            # --- Rebuild the torch.arange node ---
            new_node = new_graph.call_method("expand", args=new_expand_args)

            return new_node

        for node in traced_module.graph.nodes:
            val_map[node] = create_new_node(node)

        # Replace the old graph with the new graph and return
        traced_module.graph = new_graph
        traced_module.recompile()
        return traced_module
