import torch
import torch.nn as nn
import os
import sys
import inspect
import importlib.util
import itertools
from typing import List, Tuple, Dict, Any, Callable


class ComposedModel(nn.Module):
    def __init__(self, graph: nn.Module, subgraph: List[nn.Module]):
        super().__init__()
        self.graph = graph
        self.subgraph = nn.ModuleList(subgraph)
        self.subgraph_param_names = [
            list(inspect.signature(sm.forward).parameters.keys())
            for sm in self.subgraph
        ]
        self.extract_node = []

    def _serialize_arg(self, arg: Any) -> Any:
        if isinstance(arg, torch.fx.Node):
            return arg.name
        if isinstance(arg, (list, tuple)):
            return type(arg)(self._serialize_arg(elem) for elem in arg)
        if isinstance(arg, dict):
            return {
                self._serialize_arg(k): self._serialize_arg(v) for k, v in arg.items()
            }
        return arg

    def _extract_operators_from_graph(
        self, gm: nn.Module, example_inputs: List[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        operator_list = []
        for node in gm.graph.nodes:
            if node.op in ("call_method", "call_function", "call_module"):
                operator_info = {
                    "op_type": node.op,
                    "target": node.target,
                    "name": node.name,
                    "kwargs": self._serialize_arg(node.kwargs),
                }

                if isinstance(node.target, Callable):
                    try:
                        operator_info["target_name"] = node.target.__name__
                    except AttributeError:
                        operator_info["target_name"] = str(node.target)
                else:
                    operator_info["target_name"] = str(node.target)

                operator_list.append(operator_info)

        return operator_list

    def extract_compiler(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
        operator = self._extract_operators_from_graph(gm, inputs)
        self.extract_node.append(operator)
        return gm.forward

    def forward(self, **kwargs):
        current_args = kwargs
        compiled_model = torch.compile(self.graph, backend=self.extract_compiler)
        compiled_model(**current_args)
        graph_node_list = list(itertools.chain.from_iterable(self.extract_node))
        self.extract_node = []

        for i, (sm, param_names) in enumerate(
            zip(self.subgraph, self.subgraph_param_names)
        ):
            call_kwargs = {}
            if i > 0:
                first_param_name = param_names[0]
                call_kwargs[first_param_name] = current_args
                remaining_params = param_names[1:]
            else:
                remaining_params = param_names

            for name in remaining_params:
                if name in kwargs:
                    call_kwargs[name] = kwargs[name]

            compiled_model = torch.compile(sm, backend=self.extract_compiler)
            outputs = compiled_model(**call_kwargs)
            current_args = outputs[0]

        subgraph_node_list = list(itertools.chain.from_iterable(self.extract_node))
        self.extract_node = []

        if graph_node_list != subgraph_node_list:
            diff_in_graph = [
                item for item in graph_node_list if item not in subgraph_node_list
            ]
            diff_in_subgraph = [
                item for item in subgraph_node_list if item not in graph_node_list
            ]

            error_msg = f"Subgraph segmentation verification failed\n"
            error_msg += f"Nodes in graph but not in subgraph: {diff_in_graph}\n"
            error_msg += f"Nodes in subgraph but not in graph: {diff_in_subgraph}"
            raise ValueError(error_msg)

        return (current_args,)


class RangeDecomposerValidatorBackend:
    def _load_model_instance(self, path: str, device: str) -> torch.nn.Module:
        class_name = "GraphModule"
        model_file = os.path.join(path, "model.py")

        spec = importlib.util.spec_from_file_location(class_name, model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ModelClass = getattr(module, class_name)
        instance = ModelClass().to(device)
        return instance

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        model_file_path = model.__class__.__file_path__
        model_dir = os.path.dirname(model_file_path)
        decomposed_parent_dir = model_dir + "_decomposed"
        subgraph_paths = []
        for name in sorted(os.listdir(decomposed_parent_dir)):
            full_path = os.path.join(decomposed_parent_dir, name)
            if os.path.isdir(full_path) and name[-1].isdigit():
                subgraph_paths.append(full_path)

        print(
            f"[RangeDecomposerValidatorBackend] Found subgraphs: {[os.path.basename(p) for p in subgraph_paths]}"
        )

        device = model.__class__.__device__
        graph_instances = self._load_model_instance(model_dir, device)
        subgraph_instances = []

        for path in subgraph_paths:
            instance = self._load_model_instance(path, device)
            subgraph_instances.append(instance)
            dir_name = os.path.basename(path)
            print(
                f"[RangeDecomposerValidatorBackend] Loaded and instantiated '{dir_name}'"
            )

        composed_model = ComposedModel(graph_instances, subgraph_instances)
        return composed_model.eval()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
