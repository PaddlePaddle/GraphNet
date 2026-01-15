import torch
import copy
import operator
from collections import defaultdict
from dataclasses import dataclass


def gen_submodule_input_nodes(
    gm: torch.fx.GraphModule,
    subgraph_ranges: list[(int, int)],
    chain_style=False,
    group_head_and_tail=True,
    use_all_inputs=True,
):
    """
    chain_style=True: decompose gm into g0 * g1 * g2 * g3
    """
    submodules_body_nodes = [
        node
        for node in gm.graph.nodes
        if node.op
        not in {
            "placeholder",
            "output",
        }
    ]

    def get_range_idx2range_by_split_positions():
        split_positions = sorted(
            set(pos for subgraph_range in subgraph_ranges for pos in subgraph_range)
        )
        split_positions = (
            [0, *split_positions, len(submodules_body_nodes)]
            if group_head_and_tail
            else split_positions
        )
        split_positions = [
            max(0, min(pos, len(submodules_body_nodes))) for pos in split_positions
        ]
        return [
            (start, end)
            for i in range(len(split_positions) - 1)
            for start in [split_positions[i]]
            for end in [split_positions[i + 1]]
            if end > start
        ]

    def get_range_idx2range_by_subgraph_ranges():
        assert subgraph_ranges is not None
        num_nodes = len(submodules_body_nodes)
        for i in range(len(subgraph_ranges)):
            start, end = subgraph_ranges[i]
            assert start >= 0
            assert start < end
            assert end <= num_nodes
            # check disjoint
            assert i == 0 or start >= subgraph_ranges[i - 1][1], f"{i=}"
        return subgraph_ranges

    range_idx2range = (
        get_range_idx2range_by_split_positions()
        if chain_style
        else get_range_idx2range_by_subgraph_ranges()
    )
    range_idx2submodule_body_nodes = [
        submodules_body_nodes[start:end] for start, end in range_idx2range
    ]

    def get_body_nodes(range_idx):
        return range_idx2submodule_body_nodes[range_idx]

    def get_start_node_idx(range_idx):
        start_node = get_body_nodes(range_idx)[0]
        for i, node in enumerate(gm.graph.nodes):
            if node == start_node:
                return i
        raise NotImplementedError("Dead code.")

    def get_end_node_idx(range_idx):
        last_node = get_body_nodes(range_idx)[-1]
        for i, node in enumerate(gm.graph.nodes):
            if node == last_node:
                return i + 1
        raise NotImplementedError("Dead code.")

    num_subgraphs = len(range_idx2submodule_body_nodes)
    for range_idx in range(num_subgraphs):
        use_all_inputs = use_all_inputs and range_idx == 0
        start, end = range_idx2range[range_idx]
        (
            submodule_input_nodes,
            submodule_output_nodes,
            identity_nodes,
        ) = _get_submodule_inputs_and_outputs(
            gm=gm,
            start_node_idx=get_start_node_idx(range_idx),
            end_node_idx=get_end_node_idx(range_idx),
            chain_style=chain_style,
            use_all_inputs=use_all_inputs,
        )
        yield start, end, submodule_input_nodes


def convert_to_submodules_graph(
    gm: torch.fx.GraphModule,
    split_positions: list[int],
    subgraph_ranges: list[(int, int)] = None,
    submodule_hook=None,
    submodule_name_prefix="extracted_submodule",
    chain_style=False,
    group_head_and_tail=True,
    use_all_inputs=True,
):
    """
    chain_style=True: decompose gm into g0 * g1 * g2 * g3
    """
    gm = copy.deepcopy(gm)
    submodules_body_nodes = [
        node
        for node in gm.graph.nodes
        if node.op
        not in {
            "placeholder",
            "output",
        }
    ]

    def get_range_idx2range_by_split_positions():
        nonlocal split_positions
        split_positions = (
            [0, *split_positions, len(submodules_body_nodes)]
            if group_head_and_tail
            else split_positions
        )
        split_positions = [
            max(0, min(pos, len(submodules_body_nodes))) for pos in split_positions
        ]
        return [
            (start, end)
            for i in range(len(split_positions) - 1)
            for start in [split_positions[i]]
            for end in [split_positions[i + 1]]
            if end > start
        ]

    def get_range_idx2range_by_subgraph_ranges():
        assert subgraph_ranges is not None
        num_nodes = len(submodules_body_nodes)
        for i in range(len(subgraph_ranges)):
            start, end = subgraph_ranges[i]
            assert start >= 0
            assert start < end
            assert end <= num_nodes
            # check disjoint
            assert i == 0 or start >= subgraph_ranges[i - 1][1], f"{i=}"
        return subgraph_ranges

    range_idx2range = (
        get_range_idx2range_by_split_positions()
        if (chain_style or subgraph_ranges is None)
        else get_range_idx2range_by_subgraph_ranges()
    )
    range_idx2submodule_body_nodes = [
        submodules_body_nodes[start:end] for start, end in range_idx2range
    ]

    def get_body_nodes(range_idx):
        return range_idx2submodule_body_nodes[range_idx]

    def get_name2sub_submodule():
        used_module_names = set(
            [node.target for node in submodules_body_nodes if node.op == "call_module"]
        )
        return {
            name: module
            for name, module in gm.named_modules()
            if name in used_module_names
        }

    def get_start_node_idx(range_idx):
        start_node = get_body_nodes(range_idx)[0]
        for i, node in enumerate(gm.graph.nodes):
            if node == start_node:
                return i
        raise NotImplementedError("Dead code.")

    def get_end_node_idx(range_idx):
        last_node = get_body_nodes(range_idx)[-1]
        for i, node in enumerate(gm.graph.nodes):
            if node == last_node:
                return i + 1
        raise NotImplementedError("Dead code.")

    def print_submodule_call(prompt, gm):
        submodule_call_stmts = [
            stmt for stmt in gm.code.split("\n") if "self.extracted_submodule" in stmt
        ]
        print(f"{prompt} ", submodule_call_stmts)

    new_node2original_node = {}
    for node in gm.graph.nodes:
        new_node2original_node[node] = node

    def sort_key(node):
        return new_node2original_node[node].name

    num_subgraphs = len(range_idx2submodule_body_nodes)
    for range_idx in range(num_subgraphs):
        use_all_inputs = use_all_inputs and range_idx == 0
        (
            submodule_input_nodes,
            submodule_output_nodes,
            identity_nodes,
        ) = _get_submodule_inputs_and_outputs(
            gm=gm,
            start_node_idx=get_start_node_idx(range_idx),
            end_node_idx=get_end_node_idx(range_idx),
            chain_style=chain_style,
            use_all_inputs=use_all_inputs,
        )
        identity_node_set = set(identity_nodes)

        def get_input_nodes(range_idx):
            return sorted(submodule_input_nodes, key=sort_key)

        def get_output_nodes(range_idx):
            end = range_idx2range[range_idx][1]
            if end >= len(submodules_body_nodes):
                return submodule_output_nodes
            return sorted(submodule_output_nodes, key=sort_key)

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
            name = new_node2original_node[original_node].name
            new_node = new_graph.placeholder(name)
            node_map[original_node] = new_node

        # Copy body nodes
        for original_node in get_body_nodes(range_idx):
            new_node = new_graph.node_copy(original_node, lambda x: node_map[x])
            node_map[original_node] = new_node

        # Add output nodes
        output_args = [
            node_map[original_node] for original_node in get_output_nodes(range_idx)
        ]
        new_graph.output(tuple(output_args))

        # Create the new GraphModule
        # This assumes no submodules are being extracted, or they are handled separately
        new_sub_module = torch.fx.GraphModule(get_name2sub_submodule(), new_graph)
        if submodule_hook is not None:
            new_sub_module = submodule_hook(new_sub_module, range_idx)
        # Replace with submodule node
        gm.add_submodule(submodule_name, new_sub_module)
        with gm.graph.inserting_after(get_body_nodes(range_idx)[-1]):
            submodule_node = gm.graph.call_module(
                submodule_name, tuple(get_input_nodes(range_idx))
            )
        prev_node = submodule_node
        for idx, original_output in enumerate(get_output_nodes(range_idx)):
            with gm.graph.inserting_after(prev_node):
                new_output_node = gm.graph.call_function(
                    operator.getitem, (submodule_node, idx)
                )
                node_map[original_output] = new_output_node
                prev_node = new_output_node

        # Replace all use of outputs
        for original_output in get_output_nodes(range_idx):
            if original_output in identity_node_set:
                continue
            original_output.replace_all_uses_with(node_map[original_output])
            new_node2original_node[node_map[original_output]] = new_node2original_node[
                original_output
            ]

        # Erase old nodes
        for node in reversed(get_body_nodes(range_idx)):
            gm.graph.erase_node(node)
        # print_submodule_call("(fx) after Erase old nodes", gm)

    # print_submodule_call("(fx) before recompile", gm)

    gm.recompile()

    # print_submodule_call("(fx) after recompile", gm)

    return gm


def fold_range_to_submodule(
    gm: torch.fx.GraphModule,
    start_node_idx: int,
    end_node_idx: int,
    submodule_hook=None,
    submodule_name="extracted_submodule",
    group_head_and_tail=False,
):
    return convert_to_submodules_graph(
        gm,
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
    gm: torch.fx.GraphModule,
    start_node_idx: int,
    end_node_idx: int,
    chain_style=False,
    use_all_inputs=False,
):
    if not chain_style:
        (
            minimal_input_nodes,
            minimal_output_nodes,
        ) = _get_minimal_submodule_inputs_and_outputs(
            gm=gm, start_node_idx=start_node_idx, end_node_idx=end_node_idx
        )
        if use_all_inputs:
            node_list = list(gm.graph.nodes)
            input_nodes, _ = _get_minimal_submodule_inputs_and_outputs(
                gm=gm, start_node_idx=start_node_idx, end_node_idx=len(node_list)
            )
        else:
            input_nodes = minimal_input_nodes
        return input_nodes, minimal_output_nodes, []
    else:
        node_list = list(gm.graph.nodes)
        if _is_node_idx_out_of_range(gm, start_node_idx):
            input_nodes = list(_get_return_nodes(gm))
        else:
            input_nodes, _ = _get_minimal_submodule_inputs_and_outputs(
                gm=gm, start_node_idx=start_node_idx, end_node_idx=len(node_list)
            )
        if _is_node_idx_out_of_range(gm, end_node_idx):
            output_nodes = list(_get_return_nodes(gm))
        else:
            output_nodes, _ = _get_minimal_submodule_inputs_and_outputs(
                gm=gm, start_node_idx=end_node_idx, end_node_idx=len(node_list)
            )
        identity_nodes_set = set(input_nodes) & set(output_nodes)
        identity_nodes = [node for node in input_nodes if node in identity_nodes_set]
        return input_nodes, output_nodes, identity_nodes


def _get_minimal_submodule_inputs_and_outputs(
    gm: torch.fx.GraphModule,
    start_node_idx: int,
    end_node_idx: int,
):
    count_ctx = NodeProducedOrConsumedCountCtx(
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )
    node_list = list(gm.graph.nodes)
    assert end_node_idx <= len(node_list)

    def get_args_node(arg):
        if isinstance(arg, torch.fx.Node):
            yield arg
        elif isinstance(arg, (tuple, list)):
            for x in arg:
                yield from get_args_node(x)
        elif isinstance(arg, slice):
            yield arg.start
            yield arg.stop
            yield arg.step
        elif isinstance(arg, torch.device):
            pass
        elif isinstance(arg, torch.dtype):
            pass
        else:
            assert isinstance(
                arg,
                (
                    int,
                    bool,
                    float,
                    str,
                    type(...),
                    type(None),
                    torch.dtype,
                    torch.device,
                ),
            ), f"{type(arg)=}"

    def get_args_node_and_self_node(node):
        for arg in node.args:
            yield from get_args_node(arg)
        for name, values in node.kwargs.items():
            yield from get_args_node(values)
        yield node

    for node in node_list[0:start_node_idx]:
        for related_node in get_args_node_and_self_node(node):
            count_ctx.node2before_input[related_node] += 1

    for node in node_list[start_node_idx:end_node_idx]:
        for related_node in get_args_node_and_self_node(node):
            count_ctx.node2body[related_node] += 1

    for node in node_list[end_node_idx:]:
        for related_node in get_args_node_and_self_node(node):
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


def _get_return_nodes(gm):
    for node in gm.graph.nodes:
        if node.op != "output":
            continue
        for arg in node.args:
            if isinstance(arg, (tuple, list)):
                yield from arg
            else:
                yield arg


def _is_node_idx_out_of_range(gm, node_idx: int):
    node_list = list(gm.graph.nodes)
    num_nodes = len(node_list)
    if node_idx < 0:
        return True
    if node_idx >= num_nodes:
        return True
    return node_list[node_idx].op in {"output", "placeholder"}
