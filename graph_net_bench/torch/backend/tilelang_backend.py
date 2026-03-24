import torch
import inspect
from .graph_compiler_backend import GraphCompilerBackend

try:
    from tilelang import compile as tilelang_compile
except ImportError:
    tilelang_compile = None


class TilelangCompiledModule(torch.nn.Module):
    def __init__(self, module, cuda_graph=False, native=False):
        super().__init__()
        self.module = module
        self.cuda_graph = cuda_graph
        self.native = native
        self.runner = None
        self.param_names = list(inspect.signature(module.forward).parameters.keys())

    def forward(self, **kwargs):
        args = [kwargs[name] for name in self.param_names if name in kwargs]
        if self.runner is None:
            self.runner = tilelang_compile(
                self.module.forward,
                mode="graph",
                cuda_graph=self.cuda_graph,
                native=self.native,
                example_args=tuple(args),
            )
        return self.runner(*args)


class TilelangBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)
        if tilelang_compile is None:
            raise ImportError(
                "tilelang is not installed. "
                "Please install it to use the tilelang backend."
            )

    def __call__(self, model, **kwargs):
        cuda_graph = self.config.get("cuda_graph", False)
        native = self.config.get("native", False)
        return TilelangCompiledModule(model, cuda_graph=cuda_graph, native=native)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def version(self):
        try:
            from importlib.metadata import version

            return version("tilelang")
        except ImportError:
            return "unknown"
