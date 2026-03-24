import torch
import inspect
from .graph_compiler_backend import GraphCompilerBackend

try:
    from tilelang.torch_compile import torch_compile
except ImportError:
    torch_compile = None


class TilelangCompiledModule(torch.nn.Module):
    def __init__(self, module, arch=None, executor="auto"):
        super().__init__()
        self.module = module
        self.arch = arch
        self.executor = executor
        self.runner = None
        self.param_names = list(inspect.signature(module.forward).parameters.keys())

    def forward(self, **kwargs):
        args = [kwargs[name] for name in self.param_names if name in kwargs]
        if self.runner is None:
            compiled_forward = torch_compile(
                self.module.forward,
                arch=self.arch,
                executor=self.executor,
            )
            self.runner = compiled_forward.compile(*args)
        return self.runner(*args)


class TilelangBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)
        if torch_compile is None:
            raise ImportError(
                "tilelang is not installed. "
                "Please install it to use the tilelang backend."
            )

    def __call__(self, model, **kwargs):
        arch = self.config.get("arch", None)
        executor = self.config.get("executor", "auto")
        return TilelangCompiledModule(model, arch=arch, executor=executor)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def version(self):
        try:
            from importlib.metadata import version

            return version("tilelang")
        except ImportError:
            return "unknown"
