# graph_net/torch/backend/nope_backend.py
import torch
from .graph_compiler_backend import GraphCompilerBackend


class NopeBackend(GraphCompilerBackend):
    def __call__(self, model):
        return model

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
