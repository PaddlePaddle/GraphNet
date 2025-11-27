import torch
from typing import Union, Callable
from torch.fx.passes.shape_prop import ShapeProp
import inspect
import os
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)


# used as configuration of python3 -m graph_net.torch.run_model
class ShapePropagate:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config

    def __call__(self, module):
        return ShapePropModule(self.config, module)


class ShapePropModule(torch.nn.Module):
    def __init__(self, config, module):
        super().__init__()
        self.config = config
        self.module = module
        assert hasattr(module, "__graph_net_file_path__")

    def forward(self, *args, **kwargs):
        assert len(args) == 0
        model_path = os.path.dirname(self.module.__graph_net_file_path__)
        traced_model = parse_immutable_model_path_into_sole_graph_module(model_path)
        inputs = [
            kwargs[name] for name in inspect.signature(self.module.forward).parameters
        ]
        propagated_model = ShapeProp(traced_model).propagate(*inputs)
