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
        for node in traced_module.graph.nodes:
            if node.op == "call_method" and node.target == "expand":
                return True
        return False

    def _node_need_rewrite(self, node) -> bool:
        if not (node.op == "call_method"):
            return False
        if not (node.op == "expand"):
            return False
        input_tensor_node = node.args[0]
        input_meta = input_tensor_node.meta.get("tensor_meta")
        if input_meta is None:
            return False
        expand_args = node.args[1:]
        input_shape = input_meta.shape
        if not (len(expand_args) == len(input_shape)):
            return False
        return True

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        """
        Fx Pass: Replaces hardcoded constants in 'expand' ops that match an input tensor dimension
        with a dynamic 'size()' call. The primary goal is to dynamicize the batch size (axis 0).
        """
        # Create a new graph to hold the rewritten nodes
        new_graph = fx.Graph()

        # Create a map to link nodes from the old graph to nodes in the new graph
        val_map = {}

        for node in traced_module.graph.nodes:
            if self._node_need_rewrite(node):
                # Get the input tensor node
                input_tensor_node = node.args[0]
                # Get the target shape arguments for expand (e.g., 1, 4, 6, 64)
                expand_args = node.args[1:]

                # --- Dependency on ShapeProp Results ---
                # input_shape is the static shape (e.g., batch_size, C, H, W)
                input_meta = input_tensor_node.meta.get("tensor_meta")
                if input_meta is None:
                    raise RuntimeError(
                        f"Node {input_tensor_node.name} lacks tensor_meta. Did ShapeProp run?"
                    )

                input_shape = input_meta.shape

                # Find the new list of expand arguments
                new_expand_args = []

                # Iterate over the target dimensions of expand (dim0, dim1, ...)
                for i, target_dim in enumerate(expand_args):
                    # 1. Handle dynamic dimensions (e.g., -1 or non-integer values)
                    if not isinstance(target_dim, int) or target_dim < 1:
                        new_expand_args.append(
                            val_map[target_dim] if target_dim in val_map else target_dim
                        )
                        continue

                    # 2. Handle hardcoded constants (e.g., 1, 6, 64)

                    # --- Core Logic: Find the matching dynamic axis ---

                    # Default: Keep the hardcoded constant if no matching dynamic axis is found
                    best_match = target_dim
                    matched_axis = -1

                    axis_idx = i
                    input_dim_size = input_shape[i]
                    if target_dim == input_dim_size:
                        if axis_idx == 0:
                            matched_axis = axis_idx
                        elif axis_idx > 0 and input_dim_size > 1:
                            matched_axis = axis_idx
                        else:
                            # Do nothing.
                            pass

                    if matched_axis != -1:
                        # Found a matching dynamic axis (matched_axis), replace it with a size() call

                        # 1. Create a call to size(axis) in the new graph
                        # NOTE: input_tensor_node must first be mapped to a new graph node via val_map
                        new_input_node = val_map[input_tensor_node]

                        # Use the size() method to retrieve the dynamic dimension
                        size_node = new_graph.call_method(
                            "size", args=(new_input_node, matched_axis)
                        )

                        best_match = size_node

                    new_expand_args.append(best_match)

                # --- Rebuild the expand node ---
                # 1. Map the input tensor node to the new graph node
                new_input_node = val_map[input_tensor_node]

                # 2. Insert the new expand node into the new graph
                # with new_graph.inserting_after(new_input_node):
                new_node = new_graph.call_method(
                    "expand", args=(new_input_node, *new_expand_args)
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
