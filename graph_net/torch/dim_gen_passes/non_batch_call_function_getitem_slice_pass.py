import torch
import torch.fx as fx
from graph_net.torch.dim_gen_passes import DimensionGeneralizationPass
from collections import namedtuple
import operator
import copy
import os


class ConcretePass(DimensionGeneralizationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_pass_name(cls) -> bool:
        return os.path.basename(__file__)[:-3]

    def need_rewrite(self, traced_module: fx.GraphModule) -> bool:
        # non batch
        if 0 in self.axes:
            return False
        return any(self._node_need_rewrite(node) for node in traced_module.graph.nodes)

    def _node_need_rewrite(self, node) -> bool:
        if not (node.op == "call_function"):
            return False
        if not (node.target == operator.getitem):
            return False
        if not isinstance(node.args[1], tuple):
            return False
        if not any(self._slice_need_rewrite(slice_obj) for slice_obj in node.args[1]):
            return False
        return True

    def _slice_need_rewrite(self, slice_obj) -> bool:
        if not isinstance(slice_obj, slice):
            return False
        return self._head_slice_need_rewrite(
            slice_obj
        ) or self._tail_slice_need_rewrite(slice_obj)

    def _head_slice_need_rewrite(self, slice_obj: slice) -> bool:
        return (
            slice_obj.stop == self.dim
            and (slice_obj.start is None or slice_obj.start == 0)
            and (slice_obj.step is None or slice_obj.step == 1)
        )

    def _tail_slice_need_rewrite(self, slice_obj: slice) -> bool:
        return (
            slice_obj.start == -self.dim
            and (slice_obj.stop is None)
            and (slice_obj.step is None or slice_obj.step == 1)
        )

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        """
        Fx Pass: Replaces hardcoded constants in 'operator.getitem' ops that match an input tensor dimension
        with a dynamic 'size()' call. The primary goal is to dynamicize the batch size.
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

        def val_map_contains(key):
            if isinstance(key, slice):
                return False
            return key in val_map

        def get_new_getitem_tuple_elem(elem):
            if not (
                isinstance(elem, slice)
                and self._slice_need_rewrite(elem)
                and last_node_axis is not None
            ):
                return val_map[elem] if val_map_contains(elem) else elem

            if self._head_slice_need_rewrite(elem):
                slice_obj = copy.deepcopy(elem)
                assert slice_obj.stop == self.dim

                # Use the size() method to retrieve the dynamic dimension
                size_node = new_graph.call_method(
                    "size", args=(last_node_axis.node, last_node_axis.shape_axis)
                )
                return slice(slice_obj.start, size_node, slice_obj.step)
            elif self._tail_slice_need_rewrite(elem):
                slice_obj = copy.deepcopy(elem)
                assert slice_obj.start == -self.dim

                # Use the size() method to retrieve the dynamic dimension
                size_node = new_graph.call_method(
                    "size", args=(last_node_axis.node, last_node_axis.shape_axis)
                )
                negation_node = new_graph.call_function(
                    operator.neg,
                    args=(
                        size_node,
                    ),  # The single positional argument is the input node
                    kwargs={},  # Unary operators typically have no keyword arguments
                )
                return slice(negation_node, slice_obj.stop, slice_obj.step)
            else:
                raise NotImplementedError("Dead code.")

        def get_new_getitem_arg(arg):
            if not isinstance(arg, tuple):
                return val_map[arg] if arg in val_map else arg
            return tuple(get_new_getitem_tuple_elem(elem) for elem in arg)

        def create_new_node(node):
            if not self._node_need_rewrite(node):
                # Copy other nodes to the new graph
                new_node = new_graph.node_copy(node, lambda x: val_map[x])
                try_reset_last_node_axis(node=node, new_node=new_node)
                return new_node

            new_gettiem_args = tuple(get_new_getitem_arg(arg) for arg in node.args)

            # --- Rebuild the operator.getitem node ---
            new_node = new_graph.call_function(operator.getitem, args=new_gettiem_args)

            return new_node

        for node in traced_module.graph.nodes:
            val_map[node] = create_new_node(node)

        # Replace the old graph with the new graph and return
        traced_module.graph = new_graph
        traced_module.recompile()
        return traced_module
