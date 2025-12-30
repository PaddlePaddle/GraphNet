import operator
from collections import defaultdict, deque
import torch.fx as fx
from graph_net.torch.dim_gen_passes import DimensionGeneralizationPass
import os


class ConcretePass(DimensionGeneralizationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_pass_name(cls) -> bool:
        return os.path.basename(__file__)[:-3]

    def need_rewrite(self, traced_module: fx.GraphModule) -> bool:
        if 0 in self.axes:
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
        if -1 in node.args[1:]:
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

        def get_index_map_of_common_dim(input_shape, view_args):
            dim2input_indices = defaultdict(deque)
            for input_index, dim in enumerate(input_shape):
                dim2input_indices[dim].append(input_index)

            # arg_index: input_index
            common_arg_index2input_index = {}
            for arg_index, arg in enumerate(view_args):
                if arg in dim2input_indices.keys() and dim2input_indices[arg]:
                    input_index = dim2input_indices[arg].popleft()
                    common_arg_index2input_index[arg_index] = input_index
            return common_arg_index2input_index

        def get_new_tuple_args(input_shape, view_args):
            common_arg_index2input_index = get_index_map_of_common_dim(
                input_shape, view_args
            )
            rest_view_indices = list(
                set(range(len(view_args))) - set(common_arg_index2input_index.keys())
            )
            rest_input_indices = list(
                set(range(len(input_shape)))
                - set(common_arg_index2input_index.values())
            )

            new_view_args_dict = {}
            for arg_index, input_index in common_arg_index2input_index.items():
                if arg_index == 0:
                    new_view_args_dict[arg_index] = view_args[arg_index]
                else:
                    new_input_node = val_map[input_tensor_node]
                    size_node = new_graph.call_method(
                        "size", args=(new_input_node, input_index)
                    )
                    new_view_args_dict[arg_index] = size_node

            size_nodes = []
            for input_index in sorted(rest_input_indices):
                new_input_node = val_map[input_tensor_node]
                size_nodes.append(
                    new_graph.call_method("size", args=(new_input_node, input_index))
                )

            if len(rest_view_indices) == 1 and len(rest_input_indices) > 1:
                # Merge the reset input dims into 1.
                # e.g. input_shape=[1, 226, 4, 8], view_args=[1, 226, 32]
                mul_node = new_graph.call_function(
                    operator.mul, args=(size_nodes[0], size_nodes[1])
                )
                for i in range(2, len(size_nodes)):
                    mul_node = new_graph.call_function(
                        operator.mul, args=(mul_node, size_nodes[i])
                    )
                new_view_args_dict[rest_view_indices[0]] = mul_node
            elif (
                len(rest_input_indices) == 1
                and len(rest_view_indices) == 2
                and view_args[rest_view_indices[0]] == view_args[rest_view_indices[1]]
            ):
                # Factorize the input dim with sqrt.
                # e.g. input_shape=[1, 9216, 128], view_args=[1, 96, 96, 128]
                pow_node = new_graph.call_function(
                    operator.pow, args=(size_nodes[0], 0.5)
                )
                int_node = new_graph.call_function(int, args=(pow_node,))
                for arg_index in rest_view_indices:
                    new_view_args_dict[arg_index] = int_node
            else:
                print(f"Not Support rewriting for {input_shape=}, {view_args=}")
                for arg_index in rest_view_indices:
                    new_view_args_dict[arg_index] = view_args[arg_index]

            new_view_args = dict(sorted(new_view_args_dict.items())).values()
            return tuple(new_view_args)

        for node in traced_module.graph.nodes:
            if self._node_need_rewrite(node):
                # Get the input tensor node
                input_tensor_node = node.args[0]

                # --- Dependency on ShapeProp Results ---
                # input_shape is the static shape (e.g., batch_size, C, H, W)
                input_meta = input_tensor_node.meta.get("tensor_meta")
                if input_meta is None:
                    raise RuntimeError(
                        f"Node {input_tensor_node.name} lacks tensor_meta. Did ShapeProp run?"
                    )

                # Get the target shape arguments for view (e.g., 1, -1, 6, 64)
                input_shape = input_tensor_node.meta["tensor_meta"].shape
                view_args = node.args[1:]
                new_view_args = get_new_tuple_args(input_shape, view_args)

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
