import torch
from .graph_compiler_backend import GraphCompilerBackend

try:
    import torch_blade
except ImportError:
    torch_blade = None


class BladeDISCCompiledModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.counter = 0

    def forward(self, *args, **kwargs):
        if self.counter == 0:
            self.module = self.compile(self.module, *args, **kwargs)
        ret = self.module(*args, **kwargs)
        self.counter += 1
        return ret

    def compile(self, module, *args, **kwargs):
        dummy_input = tuple([*args, *kwargs.values()])
        return torch_blade.optimize(
            module, allow_tracing=True, model_inputs=dummy_input
        )


class BladeDISCBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model):
        return BladeDISCCompiledModule(model)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def version(self):
        return torch_blade.version
