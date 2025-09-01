import torch

try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None

from .graph_compiler_backend import GraphCompilerBackend


class TensorRTBackend(GraphCompilerBackend):
    def __call__(self, model):
        if torch_tensorrt is None:
            raise ImportError("torch_tensorrt not installed")
        return torch.compile(model, backend="tensorrt")

    def synchronize(self):
        torch.cuda.synchronize()
