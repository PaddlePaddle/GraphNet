import torch
from .graph_compiler_backend import GraphCompilerBackend


class InductorBackend(GraphCompilerBackend):
    def __call__(self, model):
        return torch.compile(model, backend="inductor")

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
