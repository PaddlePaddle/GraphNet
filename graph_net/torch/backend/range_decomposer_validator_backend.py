import torch
import inspect
import torch.nn as nn
import os
import importlib.util
from typing import List


class ComposedModel(nn.Module):
    def __init__(self, subgraphs: List[nn.Module]):
        super().__init__()
        self.subgraphs = nn.ModuleList(subgraphs)

    def forward(self, **kwargs):
        output = None
        for i, subgraph in enumerate(self.subgraphs):
            if output is None:
                output = subgraph(**self._convert_inputs(subgraph, kwargs))
            else:
                output = subgraph(*output)

        return output

    def _convert_inputs(self, subgraph, input_kwargs):
        input_keywords = set(name for name, _ in input_kwargs.items())
        sub_graph_arg_names = set(inspect.signature(subgraph.forward).parameters)
        assert (
            len(sub_graph_arg_names - input_keywords) == 0
        ), f"{(sub_graph_arg_names - input_keywords)=}"
        for remainder in input_keywords - sub_graph_arg_names:
            assert remainder.startswith("s")
            assert remainder[1:].isdigit()
        return {
            name: value
            for name, value in input_kwargs.items()
            if name in sub_graph_arg_names
        }


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

    def _make_config(
        self,
        model_path_prefix: str,
        decomposed_root: str,
        decomposed_dentry: str = "_decomposed",
    ):
        return {
            "model_path_prefix": model_path_prefix,
            "decomposed_root": decomposed_root,
            "decomposed_dentry": decomposed_dentry,
        }

    def _get_rel_model_path(self, model_path) -> str:
        model_path = os.path.realpath(model_path)
        model_path_prefix = os.path.realpath(self.config["model_path_prefix"])
        assert model_path.startswith(model_path_prefix)
        rel_model_path = model_path[len(model_path_prefix) :]
        if rel_model_path.startswith("/"):
            rel_model_path = rel_model_path[1:]
        assert not rel_model_path.startswith("/")
        return rel_model_path

    def _get_model_name_order(self, name):
        lst = name.split("_")
        if not (len(lst) > 0):
            return -1
        if not (lst[-1].isdigit()):
            return -1
        return int(lst[-1])

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        config = self._make_config(**self.config)
        model_path = os.path.dirname(model.__class__.__graph_net_file_path__)
        rel_model_path = self._get_rel_model_path(model_path)
        decomposed_parent_dir = os.path.join(
            config["decomposed_root"], rel_model_path, config["decomposed_dentry"]
        )
        subgraph_paths = []
        dentries = os.listdir(decomposed_parent_dir)
        for name in sorted(dentries, key=self._get_model_name_order):
            full_path = os.path.join(decomposed_parent_dir, name)
            if os.path.isdir(full_path) and self._get_model_name_order(name) >= 0:
                subgraph_paths.append(full_path)

        device = model.__class__.__graph_net_device__
        subgraph_instances = []

        for path in subgraph_paths:
            instance = self._load_model_instance(path, device)
            subgraph_instances.append(instance)

        composed_model = ComposedModel(subgraph_instances)
        return composed_model.eval()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
