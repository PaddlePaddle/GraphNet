from graph_net.torch import utils
import importlib.util
import torch
import sys
from typing import Type
from torch.profiler import profile, record_function, ProfilerActivity


class GraphFullyFusible:
    def __init__(self, config):
        self.config = config

    def __call__(self, model_path=None):
        torch._dynamo.reset()
        if model_path is None:
            sys.exit(1)
        # model
        model_class = load_class_from_file(
            f"{model_path}/model.py", class_name="GraphModule"
        )
        assert model_class is not None
        model = model_class()
        # print(f"{model_path=}")

        inputs_params = utils.load_converted_from_text(f"{model_path}")
        params = inputs_params["weight_info"]
        state_dict = {k: utils.replay_tensor(v) for k, v in params.items()}

        # try to run the model
        try:
            model(**state_dict)
        except Exception:
            sys.exit(1)
        # try to compile the model
        try:
            compiled_model = torch.compile(model)
        except Exception:
            sys.exit(1)
        compiled_num_of_kernels = count_kernels(compiled_model, state_dict)
        if compiled_num_of_kernels == 1:
            sys.exit(0)
        else:
            sys.exit(1)


def load_class_from_file(file_path: str, class_name: str) -> Type[torch.nn.Module]:
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    return model_class


def count_kernels(model, sample_inputs) -> int:
    """
    Count the number of CUDA kernel launches performed during a model's forward pass.

    Args:
        model(graph models)
        sample_inputs(tensors)

    Returns:
        int: The number of kernels used.

    Behavior:
        - Runs the model once inside a PyTorch profiler context.
        - Identifies the event with key = 'cudaLaunchKernel', which corresponds
        to the number of CUDA kernel launches.
    """
    model.eval()
    # Use PyTorch Profiler

    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            _ = model(**sample_inputs)
    events = prof.key_averages()

    total_count = 0
    for e in events:
        if e.key == "cuLaunchKernel" or e.key == "cudaLaunchKernel":
            total_count += e.count
    return total_count
