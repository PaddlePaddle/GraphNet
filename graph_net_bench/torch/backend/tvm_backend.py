import torch
import inspect
from .graph_compiler_backend import GraphCompilerBackend

try:
    import tvm
    from tvm import relax
    from tvm import dlight as dl
    from tvm.relax.frontend.torch import dynamo_capture_subgraphs
except ImportError:
    tvm = None
    relax = None
    from_exported_program = None


class TvmCompiledModule(torch.nn.Module):
    def __init__(self, module, device):
        super().__init__()
        self.module = module
        self.counter = 0
        self.tvm_input = []
        self.compiled_vm = None
        self.dev = tvm.device(device)
        self.target = tvm.target.Target.from_device(self.dev)
        self.param_names = list(inspect.signature(module.forward).parameters.keys())

    def forward(self, **kwargs):
        if self.counter == 0:
            self.compiled_vm = self.compile(self.module, **kwargs)
            for name in self.param_names:
                if name in kwargs and name != "s1":
                    self.tvm_input.append(tvm.nd.from_dlpack(kwargs[name]))

        output = self.compiled_vm["subgraph_0"](*self.tvm_input)
        self.counter += 1
        return torch.from_dlpack(output)

    def compile(self, module, **kwargs):
        with torch.no_grad():
            mod = dynamo_capture_subgraphs(module, **kwargs, keep_params_as_input=True)
        mod, _ = relax.frontend.detach_params(mod)
        with self.target:
            mod = tvm.ir.transform.Sequential(
                [
                    relax.get_pipeline("zero"),
                    dl.ApplyDefaultSchedule(
                        dl.gpu.Matmul(),
                        dl.gpu.GEMV(),
                        dl.gpu.Reduction(),
                        dl.gpu.GeneralReduction(),
                        dl.gpu.Fallback(),
                    ),
                ]
            )(mod)
        ex = tvm.compile(mod, target=self.target)
        vm = relax.VirtualMachine(ex, self.dev)
        return vm


class TvmBackend(GraphCompilerBackend):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model, **kwargs):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "llvm"
        return TvmCompiledModule(model, device=device)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def version(self):
        try:
            from importlib.metadata import version

            return version("tvm")
        except ImportError:
            return "unknown"
