import torch.fx as fx
from graph_net.torch.dim_gen_passes import DimensionGeneralizationPass
from collections import namedtuple
import os


class ConcretePass(DimensionGeneralizationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_pass_name(cls) -> str:
        return os.path.basename(__file__)[:-3]

    def need_rewrite(self, traced_module: fx.GraphModule) -> bool:
        if 0 in self.axes:
            return False
        if self.dim <= 1:
            return False
        return any(self._node_need_rewrite(node) for node in traced_module.graph.nodes)

    def _node_need_rewrite(self, node) -> bool:
        if not (node.op == "call_method"):
            return False
        if not (node.target == "view"):
            return False
        if not (len(node.args) >= 2):
            return False
        view_args = node.args[1:]
        if not any(arg == self.dim for arg in view_args):
            return False
        if -1 in view_args:
            if len(view_args) == 2:
                return True
            return False
        return True

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        """
        Fx Pass: Dynamic the hard-coded non batch size in view ops.
        """
        # Create a new graph to hold the rewritten nodes
        new_graph = fx.Graph()

        # Create a map to link nodes from the old graph to nodes in the new graph
        val_map = {}

        NodeAxis = namedtuple("NodeAxis", ["node", "shape_axis"])
        last_node_axis = None

        def try_reset_last_node_axis(node, new_node):
            nonlocal last_node_axis
            axis = _find_matching_axis(node.meta.get("tensor_meta"))
            if axis is not None:
                last_node_axis = NodeAxis(node=new_node, shape_axis=axis)

        def _find_matching_axis(node_meta):
            if node_meta is None or not hasattr(node_meta, "shape"):
                return None
            for axis, dim in enumerate(node_meta.shape):
                if dim == self.dim and axis > 0:
                    return axis
            return None

        def _get_target_axis_info(node):
            input_node = node.args[0]
            axis = _find_matching_axis(input_node.meta.get("tensor_meta"))
            if axis is not None:
                new_input_node = val_map.get(input_node, input_node)
                return NodeAxis(node=new_input_node, shape_axis=axis)
            return None

        def create_new_node(node):
            # Try to find the dimension from the input tensor itself first
            target_axis_info = None
            if self._node_need_rewrite(node):
                target_axis_info = _get_target_axis_info(node)

            # Fallback to last_node_axis if not found in input tensor
            if target_axis_info is None:
                target_axis_info = last_node_axis

            if not (self._node_need_rewrite(node) and target_axis_info is not None):
                # Copy other nodes to the new graph
                new_node = new_graph.node_copy(node, lambda x: val_map[x])
                try_reset_last_node_axis(node=node, new_node=new_node)
                return new_node

            def get_new_node_arg(arg):
                if not (isinstance(arg, int) and arg == self.dim):
                    return val_map[arg] if arg in val_map else arg

                assert arg == self.dim

                # Use the size() method to retrieve the dynamic dimension
                size_node = new_graph.call_method(
                    "size",
                    args=(target_axis_info.node, target_axis_info.shape_axis),
                )
                return size_node

            new_view_args = tuple(get_new_node_arg(arg) for arg in node.args)

            # --- Rebuild the node ---
            new_node = new_graph.call_method("view", args=new_view_args)

            return new_node

        for node in traced_module.graph.nodes:
            val_map[node] = create_new_node(node)

        # Replace the old graph with the new graph and return
        traced_module.graph = new_graph
        traced_module.recompile()
        return traced_module
