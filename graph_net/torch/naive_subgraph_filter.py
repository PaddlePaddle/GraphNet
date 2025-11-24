from torch.profiler import profile, record_function, ProfilerActivity


class GraphFilter:
    def __init__(self, config):
        self.config = config

    def __call__(self, gm, sample_inputs):
        print("GraphFilter")
        # print(f"GraphFilter\n{gm.code}")
        kernels_num = count_kernels(gm, sample_inputs)
        print("number of kernels is ", kernels_num)
        return True


def count_kernels(model, tensors) -> int:
    """
    Count the number of CUDA kernel launches performed during a model's forward pass.

    Args:
        model (torch.nn.Module)
        tensors

    Returns:
        int: The number of kernel launches recorded by PyTorch profiler.

    Behavior:
        - Runs the model once inside a PyTorch profiler context.
        - Identifies the event with key = 'cudaLaunchKernel', which corresponds
        to the number of CUDA kernel launches.
    """
    # Use PyTorch Profiler
    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            output = model(*tensors)
    # print(prof.key_averages().table()) #print a table of profiler result
    events = prof.key_averages()
    for e in events:
        if e.key == "cudaLaunchKernel":
            return e.count
