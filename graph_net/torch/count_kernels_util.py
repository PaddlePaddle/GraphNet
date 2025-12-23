import traceback
from graph_net.torch import utils
import importlib.util
import torch
from graph_net.optional import Optional
import sys
from typing import Type
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
        try:
            compiled_model = torch.compile(self.module)
        except Exception:
            raise GraphFusibilityStatus(GraphFusibility.kNotFullyFusible)
        ret_tensors, compiled_num_of_kernels = count_kernels(compiled_model, inputs)
        if compiled_num_of_kernels == 1:
            raise GraphFusibilityStatus(GraphFusibility.kFullyFusible)
        else:
            raise GraphFusibilityStatus(GraphFusibility.kNotFullyFusible)
        return ret_tensors


class ThrowExitStatusIfGraphFullyFusible:
    def __init__(self, config):
        self.config = config

    def __call__(self, model_path=None):
        # def callback = lambda: logger.warning("post-extract-process-call-end")
        # logger.warning("post-extract-process-call-begin")
        # atexit.register(callback)
        torch._dynamo.reset()
        if model_path is None:
            raise GraphFusibilityStatus(GraphFusibility.kNotFullyFusible)
        # model
        model_class = load_class_from_file(
            f"{model_path}/model.py", class_name="GraphModule"
        )
        assert model_class is not None
        model = model_class()
        # print(f"{model_path=}")

        inputs_params = utils.load_converted_from_text(f"{model_path}")
        params = inputs_params["weight_info"]
        state_dict = {k: utils.get_dummy_tensor(v) for k, v in params.items()}

        # try to run the model
        try:
            model(**state_dict)
        except Exception:
            raise GraphFusibilityStatus(GraphFusibility.kNotFullyFusible)
        # try to compile the model
        try:
            compiled_model = torch.compile(model)
        except Exception:
            raise GraphFusibilityStatus(GraphFusibility.kNotFullyFusible)
        _, compiled_num_of_kernels = count_kernels(compiled_model, state_dict)
        if compiled_num_of_kernels == 1:
            raise GraphFusibilityStatus(GraphFusibility.kFullyFusible)
        else:
            raise GraphFusibilityStatus(GraphFusibility.kNotFullyFusible)


class GraphFullyFusible:
    def __init__(self, config):
        self.predicator = ThrowExitStatusIfGraphFullyFusible(config)

    def __call__(self, model_path=None):
        try:
            self.predicator(model_path)
        except GraphFusibilityStatus as status:
            if status.graph_fusibility == GraphFusibility.kFullyFusible:
                sys.exit(0)
            elif status.graph_fusibility == GraphFusibility.kNotFullyFusible:
                sys.exit(1)
            else:
                raise NotImplementedError(f"{status.graph_fusibility=}")
        except Exception:
            traceback.print_exc()
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
