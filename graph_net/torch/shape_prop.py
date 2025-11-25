import torch
from typing import Union, Callable
from torch.fx.passes.shape_prop import ShapeProp
import inspect


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

    def forward(self, *args, **kwargs):
        assert len(args) == 0
        traced_model = torch.fx.symbolic_trace(self.module)
        inputs = [
            kwargs[name] for name in inspect.signature(self.module.forward).parameters
        ]
        propagated_model = ShapeProp(traced_model).propagate(*inputs)
