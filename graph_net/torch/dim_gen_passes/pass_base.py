import torch
import torch.fx as fx
import inspect
import os


class DimensionGeneralizationPass:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def get_pass_name(cls):
        return os.path.basename(os.path.dirname(inspect.getfile(cls)))

    def need_rewrite(self, traced_module: fx.GraphModule) -> bool:
        raise NotImplementedError()

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError()
