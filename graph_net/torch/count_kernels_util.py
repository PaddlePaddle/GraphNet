import torch
from graph_net.optional import Optional
from torch.profiler import profile, record_function, ProfilerActivity

from graph_net.torch.graph_fusibility_status import (
    GraphFusibilityStatus,
    GraphFusibility,
)


class TorchSubModuleFullyFusibleDecorator:
    def __init__(self, config):
        self.config = config

    def __call__(self, module, sub_module_idx):
        return TorchNNModuleFullyFusiblePredicator(module)


class CountNumKernelsNNModule(torch.nn.Module):
    def __init__(self, module, mut_opt_num_kernels: Optional):
        super().__init__()
        self.module = module
        self.compiled_module = torch.compile(self.module)
        self.mut_opt_num_kernels = mut_opt_num_kernels

    def forward(self, *inputs):
        ret_tensors, compiled_num_of_kernels = count_kernels(
            self.compiled_module, inputs
        )
        self.mut_opt_num_kernels.reset(Optional(compiled_num_of_kernels))
        return ret_tensors


class TorchNNModuleFullyFusiblePredicator(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs):
        # print(self.module)
        try:
            # compiled_model = self.module
            compiled_model = torch.compile(self.module)
        except Exception:
            raise GraphFusibilityStatus(GraphFusibility.kNotFullyFusible)
        ret_tensors, compiled_num_of_kernels = count_kernels(compiled_model, inputs)
        print("!!!!! count two keys = ", compiled_num_of_kernels)
        if compiled_num_of_kernels == 1:
            raise GraphFusibilityStatus(GraphFusibility.kFullyFusible)
        else:
            raise GraphFusibilityStatus(GraphFusibility.kNotFullyFusible)
        return ret_tensors


def count_kernels(model, sample_inputs) -> int:
    """
    Count the number of CUDA kernel launches performed during a model's forward pass.
    """
    model.eval()
    # Use PyTorch Profiler
    with torch.no_grad():
        if isinstance(sample_inputs, dict):
            _ = model(**sample_inputs)
        elif isinstance(sample_inputs, (list, tuple)):
            _ = model(*sample_inputs)
        else:
            raise NotImplementedError(f"{type(sample_inputs)=}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            if isinstance(sample_inputs, dict):
                ret_tensors = model(**sample_inputs)
            elif isinstance(sample_inputs, (list, tuple)):
                ret_tensors = model(*sample_inputs)
            else:
                raise NotImplementedError(f"{type(sample_inputs)=}")
            torch.cuda.synchronize()
    print(prof.key_averages())
    events = prof.key_averages()

    total_count = 0
    for e in events:
        if e.key == "cuLaunchKernel" or e.key == "cudaLaunchKernel":
            total_count += e.count
    return ret_tensors, total_count
