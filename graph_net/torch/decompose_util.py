import torch
import copy
import operator
from collections import defaultdict
from dataclasses import dataclass


def convert_to_submodules_graph(
    original_gm: torch.fx.GraphModule,
    split_positions: list[int],
    submodule_hook=None,
    submodule_name_prefix="extracted_submodule",
    chain_style=False,
    group_head_and_tail=True,
):
    """
    chain_style=True: decompose original_gm into g0 * g1 * g2 * g3
    """
    original_gm = copy.deepcopy(original_gm)
    num_placeholders = len(
        [node for node in original_gm.graph.nodes if node.op == "placeholder"]
    )
    submodules_body_nodes = [
        node
        for node in original_gm.graph.nodes
        if node.op
        not in {
            "placeholder",
            "output",
        }
    ]
    split_positions = (
        [0, *split_positions, len(submodules_body_nodes)]
        if group_head_and_tail
        else split_positions
    )
    split_positions = [
        max(0, min(pos, len(submodules_body_nodes))) for pos in split_positions
    ]
    range_idx2submodule_body_nodes = [
        submodules_body_nodes[start:end]
        for i in range(len(split_positions) - 1)
        for start in [split_positions[i]]
        for end in [split_positions[i + 1]]
        if end > start
    ]

    def get_body_nodes(range_idx):
        return range_idx2submodule_body_nodes[range_idx]

    def get_name2sub_submodule():
        used_module_names = set(
            [node.target for node in submodules_body_nodes if node.op == "call_module"]
        )
        return {
            name: module
            for name, module in original_gm.named_modules()
            if name in used_module_names
        }

    def get_start_node_idx(range_idx):
        start_node = get_body_nodes(range_idx)[0]
        for i, node in enumerate(original_gm.graph.nodes):
            if node == start_node:
                return i
        raise NotImplementedError("Dead code.")

    def get_end_node_idx(range_idx):
        last_node = get_body_nodes(range_idx)[-1]
        for i, node in enumerate(original_gm.graph.nodes):
            if node == last_node:
                return i + 1
        raise NotImplementedError("Dead code.")

    def print_submodule_call(prompt, gm):
        submodule_call_stmts = [
            stmt for stmt in gm.code.split("\n") if "self.extracted_submodule" in stmt
        ]
        print(f"{prompt} ", submodule_call_stmts)

    for range_idx in range(len(range_idx2submodule_body_nodes)):
        (
            submodule_input_nodes,
            submodule_output_nodes,
        ) = _get_submodule_inputs_and_outputs(
            original_gm=original_gm,
            start_node_idx=get_start_node_idx(range_idx),
            end_node_idx=get_end_node_idx(range_idx),
            chain_style=chain_style,
        )

        def get_input_nodes(range_idx):
            return submodule_input_nodes

        def get_output_nodes(range_idx):
            return submodule_output_nodes

        submodule_name = (
            f"{submodule_name_prefix}_{range_idx}"
            if range_idx > 0
            else submodule_name_prefix
        )
        new_graph = torch.fx.Graph()
        # Create a mapping for nodes from original graph to new graph
        node_map = {}

        # Add placeholder nodes for inputs
        for original_node in get_input_nodes(range_idx):
            new_node = new_graph.placeholder(original_node.name)
            node_map[original_node] = new_node

        # Copy body nodes
        for original_node in get_body_nodes(range_idx):
            new_node = new_graph.node_copy(original_node, lambda x: node_map[x])
            node_map[original_node] = new_node

        # Add output nodes
        output_args = []
        for original_node in get_output_nodes(range_idx):
            output_args.append(node_map[original_node])
        new_graph.output(tuple(output_args))

        # Create the new GraphModule
        # This assumes no submodules are being extracted, or they are handled separately
        new_sub_module = torch.fx.GraphModule(get_name2sub_submodule(), new_graph)
        if submodule_hook is not None:
            new_sub_module = submodule_hook(new_sub_module, range_idx)
        # Replace with submodule node
        original_gm.add_submodule(submodule_name, new_sub_module)
        with original_gm.graph.inserting_after(get_body_nodes(range_idx)[-1]):
            submodule_node = original_gm.graph.call_module(
                submodule_name, tuple(get_input_nodes(range_idx))
            )
        prev_node = submodule_node
        for idx, original_output in enumerate(get_output_nodes(range_idx)):
            with original_gm.graph.inserting_after(prev_node):
                new_output_node = original_gm.graph.call_function(
                    operator.getitem, (submodule_node, idx)
                )
                node_map[original_output] = new_output_node
                prev_node = new_output_node

        # Replace all use of outputs
        for original_output in get_output_nodes(range_idx):
            original_output.replace_all_uses_with(node_map[original_output])

        # Erase old nodes
        for node in reversed(get_body_nodes(range_idx)):
            original_gm.graph.erase_node(node)
        # print_submodule_call("(fx) after Erase old nodes", original_gm)

    # print_submodule_call("(fx) before recompile", original_gm)

    original_gm.recompile()

    # print_submodule_call("(fx) after recompile", original_gm)

    return original_gm


def fold_range_to_submodule(
    original_gm: torch.fx.GraphModule,
    start_node_idx: int,
    end_node_idx: int,
    submodule_hook=None,
    submodule_name="extracted_submodule",
    group_head_and_tail=True,
):
    return convert_to_submodules_graph(
        original_gm,
        split_positions=[start_node_idx, end_node_idx],
        submodule_hook=submodule_hook,
        submodule_name_prefix=submodule_name,
        group_head_and_tail=group_head_and_tail,
    )


@dataclass
class NodeProducedOrConsumedCountCtx:
    node2before_input: defaultdict(int)
    node2body: defaultdict(int)
    node2after_output: defaultdict(int)


def _get_submodule_inputs_and_outputs(
    original_gm: torch.fx.GraphModule,
    start_node_idx: int,
    end_node_idx: int,
    chain_style=False,
):
    count_ctx = NodeProducedOrConsumedCountCtx(
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )
    node_list = list(original_gm.graph.nodes)

    def get_related_node(node):
        for arg in node.args:
            if isinstance(arg, tuple):
                yield from arg
            else:
                yield arg
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

    if chain_style:
        input_nodes = [
            node
            for node in node_list
            if (count_ctx.node2before_input[node] > 0)
            if (count_ctx.node2body[node] > 0 or count_ctx.node2after_output[node] > 0)
        ]
        input_nodes_set = set(input_nodes)
        output_nodes = [
            node
            for node in node_list
            if (count_ctx.node2before_input[node] > 0 or count_ctx.node2body[node] > 0)
            if (count_ctx.node2after_output[node] > 0)
        ]
    else:
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
