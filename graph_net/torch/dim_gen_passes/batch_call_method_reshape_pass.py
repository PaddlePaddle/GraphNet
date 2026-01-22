import torch.fx as fx
from graph_net.torch.dim_gen_passes import DimensionGeneralizationPass
import os


class ConcretePass(DimensionGeneralizationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_pass_name(cls) -> str:
        return os.path.basename(__file__)[:-3]

    def need_rewrite(self, traced_module: fx.GraphModule) -> bool:
        if 0 not in self.axes:
            return False
        return any(self._node_need_rewrite(node) for node in traced_module.graph.nodes)

    def _node_need_rewrite(self, node) -> bool:
        if not (node.op == "call_method"):
            return False
        if not (node.target == "reshape"):
            return False
        if len(node.args) < 2:
            return False
        reshape_arg = node.args[1]

        if isinstance(reshape_arg, (tuple, list)):
            if (
                len(reshape_arg) > 0
                and isinstance(reshape_arg[0], int)
                and reshape_arg[0] == 1
            ):
                return True

        if isinstance(reshape_arg, int) and reshape_arg == 1:
            return True

        return False

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        """
        Fx Pass: Dynamic the hard-coded batch size in reshape ops.
        """
        # Create a new graph to hold the rewritten nodes
        new_graph = fx.Graph()

        # Create a map to link nodes from the old graph to nodes in the new graph
        val_map = {}
        batch_size_node = None

        def create_batch_size_from_node(node):
            return new_graph.call_method("size", args=(val_map[node], 0))

        for node in traced_module.graph.nodes:
            if self._node_need_rewrite(node):
                # Get the input tensor node
                input_tensor_node = node.args[0]

                # Map the input tensor node to the new graph node
                new_input_node = val_map[input_tensor_node]

                if batch_size_node is None:
                    batch_size_node = create_batch_size_from_node(input_tensor_node)

                # Get the target shape arguments for reshape
                reshape_args = node.args[1]

                if isinstance(reshape_args, (tuple, list)):
                    # reshape((1, 659, 768))
                    new_reshape_args = [batch_size_node] + list(reshape_args[1:])
                    new_node = new_graph.call_method(
                        "reshape", args=(new_input_node, tuple(new_reshape_args))
                    )
                else:
                    # reshape(1, 659, 768)
                    new_reshape_args = [batch_size_node] + list(reshape_args[1:])
                    new_node = new_graph.call_method(
                        "reshape", args=(new_input_node, *new_reshape_args)
                    )

                val_map[node] = new_node

            else:
                new_node = new_graph.node_copy(node, lambda x: val_map[x])
                val_map[node] = new_node

                if batch_size_node is None and node.op == "placeholder":
                    batch_size_node = create_batch_size_from_node(node)

        traced_module.graph = new_graph
        traced_module.recompile()
        return traced_module
