from graph_net.torch import utils
import argparse
import importlib.util
import inspect
import shutil
import torch
import logging
from pathlib import Path
from typing import Type, Any
import sys
import json
import base64
from contextlib import contextmanager

from torch.profiler import profile, record_function, ProfilerActivity


class PostExtractProcess:
    def __init__(self, config):
        self.config = config

    def __call__(self, model_path=None):
        print("PostExtractProcess")
        if model_path is None:
            return False
        import json
        import base64
        import sys
        import os

        json_string = json.dumps(self.config)
        json_bytes = json_string.encode("utf-8")
        b64_encoded_bytes = base64.b64encode(json_bytes)
        decorator_config = b64_encoded_bytes.decode("utf-8")

        # args
        parser = argparse.ArgumentParser(description="load and run model")
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to folder e.g '../../samples/torch/resnet18'",
        )
        parser.add_argument(
            "--decorator-config",
            type=str,
            required=False,
            default=None,
            help="decorator configuration string",
        )
        args = parser.parse_args()

        # model
        model_class = load_class_from_file(
            f"{model_path}/model.py", class_name="GraphModule"
        )
        assert model_class is not None
        model = model_class()
        print(f"{model_path=}")

        model = _get_decorator(args)(model)

        inputs_params = utils.load_converted_from_text(f"{model_path}")
        params = inputs_params["weight_info"]
        state_dict = {k: utils.replay_tensor(v) for k, v in params.items()}

        compiled_num_of_kernels = compile_and_count_kernels(model, state_dict)
        print("compiled: nums_of_kernels = ", compiled_num_of_kernels)
        if compiled_num_of_kernels == 1:
            print("Graph is fully fusionable")
            return True
        else:
            print(f"Graph is not fully fusionable ({compiled_num_of_kernels} kernels)")
            shutil.rmtree(model_path)
            return False


def _convert_to_dict(config_str):
    if config_str is None:
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


def _get_decorator(args):
    if args.decorator_config is None:
        return lambda model: model
    decorator_config = _convert_to_dict(args.decorator_config)
    if "decorator_path" not in decorator_config:
        return lambda model: model
    class_name = decorator_config.get("decorator_class_name", "RunModelDecorator")
    decorator_class = load_class_from_file(
        decorator_config["decorator_path"],
        class_name=class_name,
    )
    return decorator_class(decorator_config.get("decorator_config", {}))


def load_class_from_file(file_path: str, class_name: str) -> Type[torch.nn.Module]:
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    return model_class


def compile_and_count_kernels(gm, sample_inputs) -> int:
    """
    Count the number of CUDA kernel launches performed during a model's forward pass.

    Args:
        gm(graph models)
        sample_inputs(tensors)

    Returns:
        int: The number of kernels used.

    Behavior:
        - Runs the model once inside a PyTorch profiler context.
        - Identifies the event with key = 'cudaLaunchKernel', which corresponds
        to the number of CUDA kernel launches.
    """
    gm.eval()
    # Use PyTorch Profiler
    compiled_gm = torch.compile(gm)
    _ = compiled_gm(**sample_inputs)

    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        with record_function("model_inference"):
            output = compiled_gm(**sample_inputs)
    print(prof.key_averages().table())  # print a table of profiler result
    events = prof.key_averages()
    if_compile_work = any(e.key == "TorchDynamo Cache Lookup" for e in events)
    if not if_compile_work:
        print("Compile failed")
        return -1
    for e in events:
        if e.key == "cuLaunchKernel":
            return e.count
