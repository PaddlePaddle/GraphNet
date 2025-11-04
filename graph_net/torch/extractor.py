import os
import copy
import torch
import json
import shutil
import operator
from typing import Union, Callable
from . import utils
from collections import defaultdict
from dataclasses import dataclass

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_sparse_compute = True
torch._dynamo.config.raise_on_ctx_manager_usage = False
torch._dynamo.config.allow_rnn = True


class GraphExtractor:
    def __init__(
        self, name, dynamic, mut_graph_codes=None, placeholder_auto_rename=False
    ):
        self.subgraph_counter = 0
        self.name = name
        self.dynamic = dynamic
        self.mut_graph_codes = mut_graph_codes
        self.placeholder_auto_rename = placeholder_auto_rename
        self.workspace_path = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
        if not self.workspace_path:
            raise EnvironmentError(
                "Environment variable 'GRAPH_NET_EXTRACT_WORKSPACE' is not set."
            )

    def move_files(self, source_dir, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            if os.path.isfile(source_path):
                target_path = os.path.join(target_dir, item)
                shutil.move(source_path, target_path)

    def __call__(self, gm: torch.fx.GraphModule, sample_inputs):
        # 1. Get model path
        model_path = os.path.join(self.workspace_path, self.name)
        os.makedirs(model_path, exist_ok=True)

        if self.subgraph_counter == 0:
            subgraph_path = model_path
        else:
            if self.subgraph_counter == 1:
                subgraph_0_path = os.path.join(model_path, f"subgraph_0")
                self.move_files(model_path, subgraph_0_path)

            subgraph_path = os.path.join(
                model_path, f"subgraph_{self.subgraph_counter}"
            )
            os.makedirs(subgraph_path, exist_ok=True)

        self.subgraph_counter += 1

        # 2. Get full params
        params = {}
        input_idx = 0
        unique_id = 0

        def try_rename_placeholder(node):
            assert node.op == "placeholder"
            if not self.placeholder_auto_rename:
                return
            nonlocal unique_id
            node.target = f"v{unique_id}"
            unique_id += 1
            node.name = f"v{unique_id}"
            unique_id += 1

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                try_rename_placeholder(node)
                input = sample_inputs[input_idx]
                if isinstance(input, torch.SymInt):
                    input = torch.tensor(4)
                params[node.target] = input
                input_idx += 1

            if node.op == "call_function" and hasattr(node.target, "__name__"):
                if node.target.__name__ in [
                    "_enter_autocast",
                    "_exit_autocast",
                ]:
                    node.replace_all_uses_with(node.args[0])
                    gm.graph.erase_node(node)

        assert input_idx == len(sample_inputs)
        if self.mut_graph_codes is not None:
            assert isinstance(self.mut_graph_codes, list)
            self.mut_graph_codes.append(gm.code)
        # 3. Generate and save model code
        base_code = gm.code
        # gm.graph.print_tabular()
        write_code = utils.apply_templates(base_code)
        with open(os.path.join(subgraph_path, "model.py"), "w") as fp:
            fp.write(write_code)

        # 4. Save metadata
        metadata = {
            "framework": "torch",
            "num_devices_required": 1,
            "num_nodes_required": 1,
            "dynamic": bool(self.dynamic),
            "model_name": self.name,
        }
        with open(os.path.join(subgraph_path, "graph_net.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # 5. Save tensor metadata
        # Adapt to different input structures (e.g., single tensor vs. dict/tuple of tensors)
        converted = utils.convert_state_and_inputs(params, [])
        utils.save_converted_to_text(converted, file_path=subgraph_path)
        utils.save_constraints_text(
            converted,
            file_path=os.path.join(subgraph_path, "input_tensor_constraints.py"),
        )

        print(
            f"Graph and tensors for '{self.name}' extracted successfully to: {model_path}"
        )

        return gm.forward


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
        assert original_node.op not in {"placeholder", "output"}, f"{idx=}, {original_node.op=}"

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
            name:module
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
        submodule_node = original_gm.graph.call_module(submodule_name, tuple(get_input_nodes())) 
    prev_node = submodule_node
    for idx, original_output in enumerate(get_output_nodes()):
        with original_gm.graph.inserting_after(prev_node):
            new_output_node = original_gm.graph.call_function(operator.getitem, (submodule_node, idx))
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
    node2before_input:  defaultdict(int)
    node2body:          defaultdict(int)
    node2after_output:  defaultdict(int)

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


def extract(name, dynamic=True, mut_graph_codes=None, placeholder_auto_rename=False):
    """
    Extract computation graphs from PyTorch nn.Module.
    The extracted computation graphs will be saved into directory of env var $GRAPH_NET_EXTRACT_WORKSPACE.

    Args:
        name (str): The name of the model, used as the directory name for saving.
        dynamic (bool): Enable dynamic shape support in torch.compile.

    Returns:
        wrapper or decorector

    Examples:
        >>> # wrapper style:
        >>> from graph_net.torch.extractor import extract
        >>> import torch
        >>> import os
        >>> class Foo(torch.nn.Module):
        ...     def forward(self, x):
        ...         return x * 2 + 1
        ...
        >>> os.environ['GRAPH_NET_EXTRACT_WORKSPACE'] = '/tmp'
        >>> foo = extract("foo")(Foo())
        >>> foo(torch.tensor([1, 2, 3]))
        Graph and tensors for 'foo' extracted successfully to: /tmp/foo
        tensor([3, 5, 7])
        >>> print(open('/tmp/foo/model.py').read())
        import torch

        class GraphModule(torch.nn.Module):



            def forward(self, s0 : torch.SymInt, L_x_ : torch.Tensor):
                l_x_ = L_x_
                mul = l_x_ * 2;  l_x_ = None
                add = mul + 1;  mul = None
                return (add,)

        >>> # decorator style:
        >>> from graph_net.torch.extractor import extract
        >>> import torch
        >>> import os
        >>> os.environ['GRAPH_NET_EXTRACT_WORKSPACE'] = '/tmp'
        >>> @extract('bar')
        ... class Bar(torch.nn.Module):
        ...     def forward(self, x):
        ...             return x * 2 + 1
        ...
        >>> bar = Bar()
        >>> bar(torch.tensor([1, 2, 3]))
        Graph and tensors for 'bar' extracted successfully to: /tmp/bar
        tensor([3, 5, 7])
        >>> print(open("/tmp/bar/model.py").read())
        import torch

        class GraphModule(torch.nn.Module):



            def forward(self, s0 : torch.SymInt, L_x_ : torch.Tensor):
                l_x_ = L_x_
                mul = l_x_ * 2;  l_x_ = None
                add = mul + 1;  mul = None
                return (add,)

        >>>
    """

    def wrapper(model: torch.nn.Module):
        assert isinstance(model, torch.nn.Module), f"{type(model)=}"
        extractor = GraphExtractor(
            name, dynamic, mut_graph_codes, placeholder_auto_rename
        )
        # return torch.compile(backend=extractor, dynamic=dynamic)
        compiled_model = torch.compile(model, backend=extractor, dynamic=dynamic)
        return compiled_model

    def decorator(module_class):
        def constructor(*args, **kwargs):
            return wrapper(module_class(*args, **kwargs))

        return constructor

    def decorator_or_wrapper(obj):
        if isinstance(obj, torch.nn.Module):
            return wrapper(obj)
        elif issubclass(obj, torch.nn.Module):
            return decorator(obj)
        else:
            raise NotImplementedError(
                "Only torch.nn.Module instance or subclass supported."
            )

    return decorator_or_wrapper
