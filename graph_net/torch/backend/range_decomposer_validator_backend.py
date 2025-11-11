import torch
import torch.nn as nn
import os
import sys
import inspect
import importlib.util
import itertools
from typing import List, Tuple, Dict, Any, Callable


class ComposedModel(nn.Module):
    def __init__(self, subgraph: List[nn.Module]):
        super().__init__()
        self.subgraphs = nn.ModuleList(subgraph)

    def forward(self, **kwargs):
        subgraph_intput = {
            key.replace("L", "l_l", 1): value
            for key, value in kwargs.items()
            if key.startswith("L")
        }

        output = None
        for subgraph in self.subgraphs:
            if output is None:
                output = subgraph(**subgraph_intput)
            else:
                output = subgraph(*output)

        return output


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
        model_file_path = model.__class__.__graph_net_file_path__
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

        device = model.__class__.__graph_net_device__
        subgraph_instances = []

        for path in subgraph_paths:
            instance = self._load_model_instance(path, device)
            subgraph_instances.append(instance)
            dir_name = os.path.basename(path)
            print(
                f"[RangeDecomposerValidatorBackend] Loaded and instantiated '{dir_name}'"
            )

        composed_model = ComposedModel(subgraph_instances)
        return composed_model.eval()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
