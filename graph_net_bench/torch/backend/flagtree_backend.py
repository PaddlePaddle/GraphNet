import torch
from .graph_compiler_backend import GraphCompilerBackend

try:
    import flagtree
except ImportError:
    flagtree = None


class FlagtreeBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)
        self.flagtree_backend = None

    def __call__(self, model):
        if flagtree is None:
            raise ImportError("flagtree not installed")

        if self.flagtree_backend is None:
            self.flagtree_backend = flagtree.create_backend()

        return self.flagtree_backend.compile(model)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def version(self):
        if flagtree is None:
            return "not installed"
        return flagtree.__version__
