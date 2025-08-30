import torch
from .graph_compiler_backend import GraphCompilerBackend

try:
    import torch_blade
except ImportError:
    torch_blade = None


class BladeDISCBackend(GraphCompilerBackend):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __call__(self, model):
        torch_config = torch_blade.config.Config()
        torch_config.enable_mlir_amp = False
        with torch.no_grad(), torch_config:
            dummy_input = tuple(self.input_dict.values())
            compiled_model = torch_blade.optimize(
                model, allow_tracing=True, model_inputs=dummy_input
            )
        return compiled_model

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
