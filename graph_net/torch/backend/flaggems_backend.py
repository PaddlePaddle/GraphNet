import torch
from .graph_compiler_backend import GraphCompilerBackend

try:
    import flag_gems
except ImportError:
    flag_gems = None


class FlagGemsBackend(GraphCompilerBackend):
    def __call__(self, model):
        with flag_gems.use_gems():
            return model

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
