import torch
from .graph_compiler_backend import GraphCompilerBackend

try:
    import torch_xla
except ImportError:
    torch_xla = None


class XlaCompiledModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.counter = 0
        self.xla_input = {}

    def forward(self, **kwargs):
        if self.counter == 0:
            self.xla_input = {k: v.to("xla") for k, v in kwargs.items()}
            self.module = torch.compile(self.module, backend="openxla")

        ret = self.module(**self.xla_input)
        self.counter += 1
        return ret


class XlaBackend(GraphCompilerBackend):
    def __call__(self, model):
        if torch_xla is None:
            raise ImportError("torch_xla not installed")
        return XlaCompiledModule(model)

    def synchronize(self):
        torch_xla.sync()

    def version(self):
        return torch_xla.version
