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
        if not (node.target == "view"):
            return False
        if not (len(node.args) == 4):
            return False
        if not (isinstance(node.args[1], int)):
            return False
        if not node.args[2] == -1:
            return False
        return self._input_is_missing_batch_dim(node.args[0])

    def _input_is_missing_batch_dim(self, input_node: fx.Node) -> bool:
        meta = input_node.meta.get("tensor_meta")
        return meta is not None and len(meta.shape) == 4

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        """
        Fx Pass: Restore batch dimension in view ops.
        e.g., view(16, -1, 1) to view(batch, 16, -1, 1)
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

                # Get the target shape arguments for view (e.g., 16, -1, 1)
                view_args = node.args[1:]

                # Prepend batch_size to view arguments
                #  (batch_size, 16, -1, 1)
                new_view_args = (batch_size_node,) + view_args

                # Insert the new view node into the new graph
                new_node = new_graph.call_method(
                    "view", args=(new_input_node, *new_view_args)
                )
                # Map the old node to the new node
                val_map[node] = new_node
            else:
                # Copy other nodes to the new graph
                new_node = new_graph.node_copy(node, lambda x: val_map[x])
                val_map[node] = new_node

                # Use first placeholder as anchor for batch size
                if batch_size_node is None and node.op == "placeholder":
                    batch_size_node = create_batch_size_from_node(node)

        traced_module.graph = new_graph
        traced_module.recompile()
        return traced_module
