import torch
from .graph_compiler_backend import GraphCompilerBackend

try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None


class TensorRTBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model):
        if torch_tensorrt is None:
            raise ImportError("torch_tensorrt not installed")
        return torch.compile(model, backend="tensorrt")

    def synchronize(self):
        torch.cuda.synchronize()

    def version(self):
        return torch_tensorrt.version
