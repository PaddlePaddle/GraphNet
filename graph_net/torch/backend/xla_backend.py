import torch
from .graph_compiler_backend import GraphCompilerBackend

try:
    import torch_xla
except ImportError:
    torch_xla = None


class XlaBackend(GraphCompilerBackend):
    def __call__(self, model):
        if torch_xla is None:
            raise ImportError("torch_xla not installed")
        return torch.compile(model, backend="openxla")

    def synchronize(self):
        torch_xla.sync()

    def version(self):
        return torch_xla.version
