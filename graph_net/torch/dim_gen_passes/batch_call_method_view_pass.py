import torch.fx as fx
from graph_net.torch.dim_gen_passes import DimensionGeneralizationPass
import os


class ConcretePass(DimensionGeneralizationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_pass_name(cls) -> bool:
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
        if not (len(node.args) >= 2):
            return False
        if not (isinstance(node.args[1], int)):
            return False
        if not (self.dim == node.args[1]):
            return False
        return True

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        """
        Fx Pass: Replaces hardcoded constants in 'view' ops that match an input tensor dimension
        with a dynamic 'size()' call. The primary goal is to dynamicize the batch size (axis 0).
        """
        # Create a new graph to hold the rewritten nodes
        new_graph = fx.Graph()

        # Create a map to link nodes from the old graph to nodes in the new graph
        val_map = {}

        def get_new_tuple_args(input_tensor_node, view_args):
            # --- Dependency on ShapeProp Results ---
            # input_shape is the static shape (e.g., batch_size, C, H, W)
            input_meta = input_tensor_node.meta.get("tensor_meta")
            if input_meta is None:
                raise RuntimeError(
                    f"Node {input_tensor_node.name} lacks tensor_meta. Did ShapeProp run?"
                )

            input_shape = input_meta.shape

            # Find the new list of view arguments
            new_view_args = []
            for axis_idx, target_dim in enumerate(view_args):
                if not isinstance(target_dim, int) or target_dim < 1:
                    new_view_args.append(
                        val_map[target_dim] if target_dim in val_map else target_dim
                    )
                    continue

                if axis_idx == 0 and target_dim == input_shape[axis_idx]:
                    new_input_node = val_map[input_tensor_node]
                    size_node = new_graph.call_method(
                        "size", args=(new_input_node, axis_idx)
                    )
                    best_match = size_node
                else:
                    best_match = target_dim
                new_view_args.append(best_match)
            return tuple(new_view_args)

        for node in traced_module.graph.nodes:
            if self._node_need_rewrite(node):
                # Get the input tensor node
                input_tensor_node = node.args[0]
                # Get the target shape arguments for view (e.g., 1, -1, 6, 64)
                view_args = node.args[1:]
                new_view_args = get_new_tuple_args(input_tensor_node, view_args)

                # --- Rebuild the view node ---
                # 1. Map the input tensor node to the new graph node
                new_input_node = val_map[input_tensor_node]

                # 2. Insert the new view node into the new graph
                # with new_graph.inserting_after(new_input_node):
                new_node = new_graph.call_method(
                    "view", args=(new_input_node, *new_view_args)
                )

                # 3. Map the old node to the new node
                val_map[node] = new_node

            else:
                # Copy other nodes to the new graph
                new_node = new_graph.node_copy(node, lambda x: val_map[x])
                val_map[node] = new_node

        # Replace the old graph with the new graph and return
        traced_module.graph = new_graph
        traced_module.recompile()
        return traced_module
