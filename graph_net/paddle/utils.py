import re
from collections import OrderedDict
import uuid
import json
import os
import argparse
import importlib
import inspect
import paddle


def get_limited_precision_float_str(value):
    if not isinstance(value, float):
        return value
    return f"{value:.3f}"


def convert_state_and_inputs_impl(state_dict, example_inputs):
    def tensor_info(tensor):
        is_float = tensor.dtype.is_floating_point
        mean = float(tensor.mean().item()) if is_float else None
        std = float(tensor.std().item()) if is_float else None
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": get_limited_precision_float_str(mean),
            "std": get_limited_precision_float_str(std),
        }

    def process_tensor(tensor):
        if not isinstance(tensor, paddle.Tensor):
            return {"type": "unknown", "value": tensor}

        info = tensor_info(tensor)
        if tensor.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64]:
            if tensor.numel() < 1024:
                return {
                    "type": "small_int_tensor",
                    "data": tensor.clone(),
                    "info": info,
                }
            else:
                return {"type": "big_int_tensor", "data": tensor.clone(), "info": info}
        elif tensor.numel() < 1024:
            return {"type": "small_tensor", "data": tensor.clone(), "info": info}
        else:
            return {"type": "random_tensor", "info": info}

    if isinstance(example_inputs, paddle.Tensor):
        processed_inputs = process_tensor(example_inputs)
    elif isinstance(example_inputs, (list, tuple)):
        processed_inputs = [process_tensor(t) for t in example_inputs]
    else:
        processed_inputs = {"type": "unknown", "value": example_inputs}

    def handle_named_tensors(tensor):
        data_value = None
        data_type = "random_tensor"
        if tensor.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64]:
            if tensor.numel() < 1024:
                data_type = "small_int_tensor"
                data_value = tensor.clone()
            else:
                data_type = "big_int_tensor"
        info = tensor_info(tensor)
        return {"info": info, "data": data_value, "type": data_type}

    processed_weights = {
        key: handle_named_tensors(tensor) for key, tensor in state_dict.items()
    }

    # dynamic_shapes = extract_dynamic_shapes(example_inputs)
    return {
        "input_info": processed_inputs,
        "weight_info": processed_weights,
        "dynamic_shapes": None,
    }


def convert_state_and_inputs(state_dict, example_inputs):
    return convert_state_and_inputs_impl(state_dict, example_inputs)


def save_constraints_text(converted, file_path):
    lines = []
    if converted["dynamic_shapes"] is not None:
        raise NotImplementedError("Handling constraints is not implemented yet.")
    with open(file_path, "w") as f:
        f.write("\n".join(lines))


def load_converted_from_text(file_path):
    input_info = {
        data["name"]: data
        for data in convert_meta_classes_to_tensors(f"{file_path}/input_meta.py")
    }

    weight_info = {
        data["name"]: data
        for data in convert_meta_classes_to_tensors(f"{file_path}/weight_meta.py")
    }

    return {
        "input_info": input_info,
        "weight_info": weight_info,
        "dynamic_shapes": None,
    }


def load_converted_list_from_text(file_path):
    input_info = [
        data for data in convert_meta_classes_to_tensors(f"{file_path}/input_meta.py")
    ]
    weight_info = [
        data for data in convert_meta_classes_to_tensors(f"{file_path}/weight_meta.py")
    ]

    return [*input_info, *weight_info]


def convert_meta_classes_to_tensors(file_path):
    for name, cls in _get_classes(file_path):
        attrs = {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
        data_value = None
        data_type = getattr(paddle, attrs.get("dtype", "paddle.float").split(".")[-1])
        if attrs.get("data") is not None:
            if isinstance(attrs.get("data"), str):
                raise ValueError("Unimplemented")
            else:
                data_value = paddle.tensor(attrs["data"], dtype=data_type).reshape(
                    attrs.get("shape"), []
                )
        yield {
            "info": {
                "shape": attrs.get("shape", []),
                "dtype": data_type,
                "device": attrs.get("device", "gpu"),
                "mean": attrs.get("mean", 0.0),
                "std": attrs.get("std", 1.0),
            },
            "data": data_value,
            "name": attrs.get("name"),
        }


def _get_classes(file_path):
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    yield from inspect.getmembers(unnamed, inspect.isclass)


def extract_dynamic_shapes(example_inputs):
    pass


def replay_tensor(info):
    device = info["info"]["device"]
    dtype = info["info"]["dtype"]
    shape = info["info"]["shape"]
    if None in shape:
        shape = list(map(lambda i: i if i is not None else 1, shape))
    mean = info["info"]["mean"]
    std = info["info"]["std"]
    if "data" in info and info["data"] is not None:
        return info["data"].to(device)

    return paddle.randn(shape).to(dtype).to(device) * std * 1e-3 + 1e-2
