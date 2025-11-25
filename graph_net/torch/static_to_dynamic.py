import torch
import torch.fx as fx
from graph_net.torch.utils import convert_tensor_meta_attrs_list_to_named_tensors
from torch.fx.passes.shape_prop import ShapeProp
from graph_net.torch.utils import apply_templates
from pathlib import Path
import inspect
from typing import Any
from contextlib import contextmanager
from torch.export import export
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module


# used as configuration of python3 -m graph_net.torch.run_model
class StaticToDynamic:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config

    def __call__(self, module):
        return StaticToDynamicModule(self.config, module)


class StaticToDynamicModule(torch.nn.Module):
    def __init__(self, config, module):
        super().__init__()
        config = {} if config is None else config
        self.config = self.make_config(**config)
        self.module = module

    def make_config(self):
        return {}

    def need_rewrite(self):
        try:
            traced_module = torch.fx.symbolic_trace(self.module)
        except:
            return False
        return any(
            predicator(traced_module) for predicator, _ in self.get_conditional_passes()
        )

    def save_graph_module(self, graph_module, model_path):
        py_code = apply_templates(graph_module.code)
        (Path(model_path) / "model.py").write_text(py_code)

    def rewrite_with_tensor_meta_attrs_list(self, tensor_meta_attrs_list):
        named_tensors = convert_tensor_meta_attrs_list_to_named_tensors(
            tensor_meta_attrs_list
        )
        ret = self.rewrite(**{k: v for k, v in named_tensors})
        return ret

    def rewrite(self, *args, **kwargs):
        assert len(args) == 0
        inputs = tuple(
            kwargs[name] for name in inspect.signature(self.module.forward).parameters
        )
        traced_module = parse_sole_graph_module(self.module, inputs)

        for predicator, pass_fn in self.get_conditional_passes():
            if predicator(traced_module):
                ShapeProp(traced_module).propagate(*inputs)
                traced_module = pass_fn(traced_module)

        return traced_module

    @classmethod
    def get_conditional_passes(cls):
        def call_method_predicator(method_name):
            return lambda module: has_call_method(module, method_name)

        return [
            (call_method_predicator("view"), dynamic_view_rewrite_pass),
            (call_method_predicator("reshape"), dynamic_reshape_rewrite_pass),
        ]

    def forward(self, *args, **kwargs):
        traced_module = self.rewrite(*args, **kwargs)

        inputs = [
            kwargs[name] for name in inspect.signature(self.module.forward).parameters
        ]
        ShapeProp(traced_module).propagate(*inputs)

        # return traced_module(*args, **kwargs)


def dynamic_view_rewrite_pass(traced_module: fx.GraphModule) -> fx.GraphModule:
    """
    Fx Pass: Replaces hardcoded constants in 'view' ops that match an input tensor dimension
    with a dynamic 'size()' call. The primary goal is to dynamicize the batch size (axis 0).
    """
    # Create a new graph to hold the rewritten nodes
    new_graph = fx.Graph()

    # Create a map to link nodes from the old graph to nodes in the new graph
    val_map = {}

    for node in traced_module.graph.nodes:
        if node.op == "call_method" and node.target == "view":
            # Get the input tensor node
            input_tensor_node = node.args[0]
            # Get the target shape arguments for view (e.g., 1, -1, 6, 64)
            view_args = node.args[1:]

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

            # Iterate over the target dimensions of view (dim0, dim1, ...)
            for i, target_dim in enumerate(view_args):
                # 1. Handle dynamic dimensions (e.g., -1 or non-integer values)
                if not isinstance(target_dim, int) or target_dim < 1:
                    new_view_args.append(
                        val_map[target_dim] if target_dim in val_map else target_dim
                    )
                    continue

                # 2. Handle hardcoded constants (e.g., 1, 6, 64)

                # --- Core Logic: Find the matching dynamic axis ---

                # Default: Keep the hardcoded constant if no matching dynamic axis is found
                best_match = target_dim
                matched_axis = -1

                # Iterate over all dimensions of the input tensor node
                for axis_idx, input_dim_size in enumerate(input_shape):
                    # If the view target dimension equals the size of one of the input tensor dimensions
                    # and this dimension is one we wish to generalize (e.g., batch size, axis 0)
                    if target_dim == input_dim_size:
                        # Prioritize matching the batch size (axis 0)
                        if axis_idx == 0:
                            matched_axis = 0
                            break
                        # If not axis 0, but still a match, consider generalization
                        elif matched_axis == -1:
                            matched_axis = axis_idx

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

                new_view_args.append(best_match)

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


def dynamic_reshape_rewrite_pass(traced_module: fx.GraphModule) -> fx.GraphModule:
    """
    Fx Pass: Replaces hardcoded constants in 'reshape' ops that match an input tensor dimension
    with a dynamic 'size()' call. The primary goal is to dynamicize the batch size (axis 0).
    """
    # Create a new graph to hold the rewritten nodes
    new_graph = fx.Graph()

    # Create a map to link nodes from the old graph to nodes in the new graph
    val_map = {}

    for node in traced_module.graph.nodes:
        if node.op == "call_method" and node.target == "reshape":
            # Get the input tensor node
            input_tensor_node = node.args[0]
            # Get the target shape arguments for reshape (e.g., 1, -1, 6, 64)
            reshape_args = node.args[1:]

            # --- Dependency on ShapeProp Results ---
            # input_shape is the static shape (e.g., batch_size, C, H, W)
            input_meta = input_tensor_node.meta.get("tensor_meta")
            if input_meta is None:
                raise RuntimeError(
                    f"Node {input_tensor_node.name} lacks tensor_meta. Did ShapeProp run?"
                )

            input_shape = input_meta.shape

            # Find the new list of reshape arguments
            new_reshape_args = []

            # Iterate over the target dimensions of reshape (dim0, dim1, ...)
            for i, target_dim in enumerate(reshape_args):
                # 1. Handle dynamic dimensions (e.g., -1 or non-integer values)
                if not isinstance(target_dim, int) or target_dim < 1:
                    new_reshape_args.append(
                        val_map[target_dim] if target_dim in val_map else target_dim
                    )
                    continue

                # 2. Handle hardcoded constants (e.g., 1, 6, 64)

                # --- Core Logic: Find the matching dynamic axis ---

                # Default: Keep the hardcoded constant if no matching dynamic axis is found
                best_match = target_dim
                matched_axis = -1

                # Iterate over all dimensions of the input tensor node
                for axis_idx, input_dim_size in enumerate(input_shape):
                    # If the reshape target dimension equals the size of one of the input tensor dimensions
                    # and this dimension is one we wish to generalize (e.g., batch size, axis 0)
                    if target_dim == input_dim_size:
                        # Prioritize matching the batch size (axis 0)
                        if axis_idx == 0:
                            matched_axis = 0
                            break
                        # If not axis 0, but still a match, consider generalization
                        elif matched_axis == -1:
                            matched_axis = axis_idx

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

                new_reshape_args.append(best_match)

            # --- Rebuild the reshape node ---
            # 1. Map the input tensor node to the new graph node
            new_input_node = val_map[input_tensor_node]

            # 2. Insert the new reshape node into the new graph
            # with new_graph.inserting_after(new_input_node):
            new_node = new_graph.call_method(
                "reshape", args=(new_input_node, *new_reshape_args)
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


def has_call_method(traced_module: fx.GraphModule, method_name) -> bool:
    for node in traced_module.graph.nodes:
        if node.op == "call_method" and node.target == method_name:
            return True
    return False
