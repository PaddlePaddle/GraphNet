import torch
from graph_net.optional import Optional
from torch.profiler import profile, record_function, ProfilerActivity


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


def count_kernels(model, sample_inputs) -> int:
    """
    Count the number of CUDA kernel launches performed during a model's forward pass.
    """
    model.eval()

    def run():
        if isinstance(sample_inputs, dict):
            ret_tensors = model(**sample_inputs)
        elif isinstance(sample_inputs, (list, tuple)):
            ret_tensors = model(*sample_inputs)
        else:
            raise NotImplementedError(f"{type(sample_inputs)=}")
        return ret_tensors

    # warmup
    for _ in range(3):
        run()

    # Use PyTorch Profiler
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            ret_tensors = run()

    events = prof.key_averages()

    total_count = 0
    for e in events:
        if e.key == "cuLaunchKernel" or e.key == "cudaLaunchKernel":
            total_count += e.count
    return ret_tensors, total_count
