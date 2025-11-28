import torch
import torch.fx as fx
from graph_net.torch.dim_gen_passes import DimensionGeneralizationPass
from collections import namedtuple
import os
import operator


class ConcretePass(DimensionGeneralizationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_pass_name(cls) -> bool:
        return os.path.basename(__file__)[:-3]

    def need_rewrite(self, traced_module: fx.GraphModule) -> bool:
        # non batch
        if 0 in self.axes:
            return False
        if not any(self._is_pivote_node(node) for node in traced_module.graph.nodes):
            return False
        return any(self._node_need_rewrite(node) for node in traced_module.graph.nodes)

    def _node_target(self):
        return torch.arange

    def _is_pivote_node(self, node):
        if not (node.op == "call_function"):
            return False
        if not (self._is_pivote_target(node.target)):
            return False
        if not (len(node.args) == 1):
            return False
        input_node = node.args[0]
        input_node_meta = input_node.meta.get("tensor_meta")
        if not (input_node_meta is not None):
            return False
        if not (hasattr(input_node_meta, "shape")):
            return False
        shape = input_node_meta.shape
        if not any(dim == self.dim + self._dyn_dim_delta() for dim in shape):
            return False
        return True

    def _is_pivote_target(self, target):
        return target == torch.triu

    def _node_need_rewrite(self, node) -> bool:
        if not (node.op == "call_function"):
            return False
        if not (node.target == self._node_target()):
            return False
        if not (len(node.args) == 1):
            return False
        if not (node.args[0] == self.dim + self._dyn_dim_delta()):
            return False
        return True

    def _dyn_dim_delta(self):
        return 1

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
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

        def get_new_node_dim(dim):
            if not (dim == self.dim + self._dyn_dim_delta()):
                return val_map[dim] if dim in val_map else dim
            assert dim == self.dim + self._dyn_dim_delta()

            # Use the size() method to retrieve the dynamic dimension
            size_node = new_graph.call_method(
                "size", args=(last_node_axis.node, last_node_axis.shape_axis)
            )
            plus_one_node = new_graph.call_function(operator.add, args=(size_node, 1))
            return plus_one_node

        def create_new_node(node):
            if not (self._node_need_rewrite(node) and last_node_axis is not None):
                # Copy other nodes to the new graph
                new_node = new_graph.node_copy(node, lambda x: val_map[x])
                try_reset_last_node_axis(node=node, new_node=new_node)
                return new_node

            new_node_dim = get_new_node_dim(dim=node.args[0])

            new_node = new_graph.call_function(
                self._node_target(), args=(new_node_dim,), kwargs=node.kwargs
            )

            return new_node

        for node in traced_module.graph.nodes:
            val_map[node] = create_new_node(node)

        # Replace the old graph with the new graph and return
        traced_module.graph = new_graph
        traced_module.recompile()
        return traced_module
