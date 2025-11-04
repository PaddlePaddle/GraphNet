import torch
import copy
import operator
from collections import defaultdict
from dataclasses import dataclass


def fold_range_to_submodule(
    original_gm: torch.fx.GraphModule,
    start_node_idx: int,
    end_node_idx: int,
    submodule_hook=None,
    submodule_name="extraced_submodule",
):
    original_gm = copy.deepcopy(original_gm)
    submodule_body_nodes = list(original_gm.graph.nodes)[start_node_idx:end_node_idx]

    def get_body_nodes():
        return submodule_body_nodes

    assert len(get_body_nodes()) > 0

    for idx, original_node in enumerate(get_body_nodes()):
        assert original_node.op not in {
            "placeholder",
            "output",
        }, f"{idx=}, {original_node.op=}"

    submodule_input_nodes, submodule_output_nodes = _get_submodule_inputs_and_outputs(
        original_gm=original_gm,
        start_node_idx=start_node_idx,
        end_node_idx=end_node_idx,
    )

    def get_input_nodes():
        return submodule_input_nodes

    def get_output_nodes():
        return submodule_output_nodes

    def get_name2sub_submodule():
        used_module_names = set()
        for node in get_body_nodes():
            if node.op == "call_module":
                used_module_names.add(node.target)
        return {
            name: module
            for name, module in original_gm.named_modules()
            if name in used_module_names
        }

    new_graph = torch.fx.Graph()
    # Create a mapping for nodes from original graph to new graph
    node_map = {}

    # Add placeholder nodes for inputs
    for original_node in get_input_nodes():
        new_node = new_graph.placeholder(original_node.name)
        node_map[original_node] = new_node

    # Copy body nodes
    for original_node in get_body_nodes():
        print(original_node)
        new_node = new_graph.node_copy(original_node, lambda x: node_map[x])
        node_map[original_node] = new_node

    # Add output nodes
    output_args = []
    for original_node in get_output_nodes():
        output_args.append(node_map[original_node])
    new_graph.output(tuple(output_args))

    # Create the new GraphModule
    # This assumes no submodules are being extracted, or they are handled separately
    new_sub_module = torch.fx.GraphModule(get_name2sub_submodule(), new_graph)
    if submodule_hook is not None:
        new_sub_module = submodule_hook(new_sub_module)
    # Replace with submodule node
    original_gm.add_submodule(submodule_name, new_sub_module)
    with original_gm.graph.inserting_after(get_body_nodes()[-1]):
        submodule_node = original_gm.graph.call_module(
            submodule_name, tuple(get_input_nodes())
        )
    prev_node = submodule_node
    for idx, original_output in enumerate(get_output_nodes()):
        with original_gm.graph.inserting_after(prev_node):
            new_output_node = original_gm.graph.call_function(
                operator.getitem, (submodule_node, idx)
            )
            node_map[original_output] = new_output_node
            prev_node = new_output_node

    # Replace all use of outputs
    for original_output in get_output_nodes():
        original_output.replace_all_uses_with(node_map[original_output])

    # Erase old nodes
    for node in reversed(get_body_nodes()):
        original_gm.graph.erase_node(node)

    original_gm.recompile()

    return original_gm


@dataclass
class NodeProducedOrConsumedCountCtx:
    node2before_input: defaultdict(int)
    node2body: defaultdict(int)
    node2after_output: defaultdict(int)


def _get_submodule_inputs_and_outputs(
    original_gm: torch.fx.GraphModule,
    start_node_idx: int,
    end_node_idx: int,
):
    count_ctx = NodeProducedOrConsumedCountCtx(
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )
    node_list = list(original_gm.graph.nodes)

    def get_related_node(node):
        yield from node.args
        yield node

    for node in node_list[0:start_node_idx]:
        for related_node in get_related_node(node):
            count_ctx.node2before_input[related_node] += 1

    for node in node_list[start_node_idx:end_node_idx]:
        for related_node in get_related_node(node):
            count_ctx.node2body[related_node] += 1

    for node in node_list[end_node_idx:]:
        for related_node in get_related_node(node):
            count_ctx.node2after_output[related_node] += 1

    input_nodes = [
        node
        for node in node_list
        if count_ctx.node2before_input[node] > 0
        if count_ctx.node2body[node] > 0
    ]

    output_nodes = [
        node
        for node in node_list
        if not (count_ctx.node2before_input[node] > 0)
        if count_ctx.node2body[node] > 0
        if count_ctx.node2after_output[node] > 0
    ]

    return input_nodes, output_nodes
