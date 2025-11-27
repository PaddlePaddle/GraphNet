import os
import torch
import json
import shutil
from typing import Union, Callable
from graph_net.torch import utils
from graph_net.torch.fx_graph_serialize_util import serialize_graph_module_to_str

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_sparse_compute = True
torch._dynamo.config.raise_on_ctx_manager_usage = False
torch._dynamo.config.allow_rnn = True


# used as configuration of python3 -m graph_net.torch.run_model
class RunModelDecorator:
    def __init__(self, config):
        self.config = self.make_config(**config)

    def __call__(self, model):
        return extract(**self.config)(model)

    def make_config(
        self,
        name=None,
        dynamic=True,
        placeholder_auto_rename=False,
        custom_extractor_path: str = None,
        custom_extractor_config: dict = None,
    ):
        assert name is not None
        return {
            "name": name,
            "dynamic": dynamic,
            "placeholder_auto_rename": placeholder_auto_rename,
            "extractor_config": {
                "custom_extractor_path": custom_extractor_path,
                "custom_extractor_config": custom_extractor_config,
            },
        }


class GraphExtractor:
    def __init__(
        self,
        name,
        dynamic,
        mut_graph_codes=None,
        placeholder_auto_rename=False,
        workspace_path=None,
    ):
        self.subgraph_counter = 0
        self.name = name
        self.dynamic = dynamic
        self.mut_graph_codes = mut_graph_codes
        self.placeholder_auto_rename = placeholder_auto_rename
        self.workspace_path = (
            workspace_path
            if workspace_path is not None
            else os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
        )
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
            self.mut_graph_codes.append(serialize_graph_module_to_str(gm))
        # 3. Generate and save model code
        base_code = serialize_graph_module_to_str(gm)
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


def extract(
    name,
    dynamic=True,
    mut_graph_codes=None,
    placeholder_auto_rename=False,
    extractor_config: dict = None,
):
    """
    Extract computation graphs from PyTorch nn.Module.
    The extracted computation graphs will be saved into directory of env var $GRAPH_NET_EXTRACT_WORKSPACE.

    Args:
        name (str): The name of the model, used as the directory name for saving.
        dynamic (bool): Enable dynamic shape support in torch.compile.

    Returns:
        wrapper or decorator

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

    extractor_config = make_extractor_config(extractor_config)

    def get_graph_extractor_maker():
        custom_extractor_path = extractor_config["custom_extractor_path"]
        custom_extractor_config = extractor_config["custom_extractor_config"]
        if custom_extractor_path is None:
            return GraphExtractor
        import importlib.util as imp

        spec = imp.spec_from_file_location("graph_extractor", custom_extractor_path)
        graph_extractor = imp.module_from_spec(spec)
        spec.loader.exec_module(graph_extractor)
        cls = graph_extractor.GraphExtractor
        return lambda *args, **kwargs: cls(custom_extractor_config, *args, **kwargs)

    def wrapper(model: torch.nn.Module):
        assert isinstance(model, torch.nn.Module), f"{type(model)=}"
        extractor = get_graph_extractor_maker()(
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


def make_extractor_config(extractor_config):
    kwargs = extractor_config if extractor_config is not None else {}
    return make_extractor_config_impl(**kwargs)


def make_extractor_config_impl(
    custom_extractor_path: str = None, custom_extractor_config: dict = None
):
    config = custom_extractor_config if custom_extractor_config is not None else {}
    return {
        "custom_extractor_path": custom_extractor_path,
        "custom_extractor_config": config,
    }
