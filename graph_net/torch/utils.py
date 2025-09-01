import re
import torch
import torch.nn as nn
from collections import OrderedDict
import uuid
import json
import os
import argparse
import importlib
import inspect
import math

kLiteralTensorSize = 64


def apply_templates(forward_code: str) -> str:
    tab = "    "
    forward_code = f"\n{tab}".join(forward_code.split("\n"))
    imports = "import torch"
    if "device" in forward_code:
        imports += "\n\nfrom torch import device"
    if "inf" in forward_code:
        imports += "\n\nfrom torch import inf"
    return f"{imports}\n\nclass GraphModule(torch.nn.Module):\n{tab}{forward_code}"


def get_limited_precision_float_str(value):
    if not isinstance(value, float):
        return value
    return f"{value:.3f}"


def convert_state_and_inputs_impl(state_dict, example_inputs):
    def tensor_info(tensor):
        is_float = tensor.dtype.is_floating_point
        mean = float(tensor.mean().item()) if is_float else None
        std = None
        if is_float:
            if tensor.numel() <= 1:
                std = 0.0
            else:
                std = float(tensor.std().item())
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": get_limited_precision_float_str(mean),
            "std": get_limited_precision_float_str(std),
        }

    def process_tensor(tensor):
        if not isinstance(tensor, torch.Tensor):
            return {"type": "unknown", "value": tensor}

        info = tensor_info(tensor)
        if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            if tensor.numel() < kLiteralTensorSize:
                return {
                    "type": "small_int_tensor",
                    "data": tensor.clone(),
                    "info": info,
                }
            else:
                return {
                    "type": "big_int_tensor_by_range",
                    "min_val": tensor.min().item(),
                    "max_val": tensor.max().item(),
                    "info": info,
                }
        elif tensor.numel() < kLiteralTensorSize:
            return {"type": "small_tensor", "data": tensor.clone(), "info": info}
        else:
            return {"type": "random_tensor", "info": info}

    if isinstance(example_inputs, torch.Tensor):
        processed_inputs = process_tensor(example_inputs)
    elif isinstance(example_inputs, (list, tuple)):
        processed_inputs = [process_tensor(t) for t in example_inputs]
    else:
        processed_inputs = {"type": "unknown", "value": example_inputs}

    def handle_named_tensors(tensor):
        info = tensor_info(tensor)
        if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            if tensor.numel() < kLiteralTensorSize:
                return {
                    "info": info,
                    "data": tensor.clone(),
                    "type": "small_int_tensor",
                }
            else:
                return {
                    "info": info,
                    "min_val": tensor.min().item(),
                    "max_val": tensor.max().item(),
                    "type": "big_int_tensor_by_range",
                }
        if tensor.numel() < kLiteralTensorSize:
            return {"info": info, "data": tensor.clone(), "type": "small_tensor"}
        else:
            return {"info": info, "data": None, "type": "random_tensor"}

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


def save_converted_to_text(converted, file_path):
    def format_data(data):
        if data is None:
            return "None"
        elif isinstance(data, torch.Tensor):
            if data.dtype.is_floating_point:
                return "[{}]".format(
                    ", ".join(f"{x:.6f}" for x in data.flatten().tolist())
                )
            else:
                return "[{}]".format(", ".join(f"{x}" for x in data.flatten().tolist()))
        else:
            return repr(data)

    def process_tensor_info(tensor_info, name_prefix="example_input"):
        tensor_type = tensor_info.get("type")
        info = tensor_info.get("info", {})
        dtype = info.get("dtype", "torch.float")
        shape = info.get("shape", [])
        device = info.get("device", "cpu")
        mean = info.get("mean", 0.0)
        std = info.get("std", 1.0)
        uid = f"{name_prefix}_tensor_meta_{tensor_info.get('name', '')}"

        lines = [
            (f"class {uid}:"),
            (f"\tname = \"{tensor_info.get('name', '')}\""),
            (f"\tshape = {shape}"),
            (f'\tdtype = "{dtype}"'),
            (f'\tdevice = "{device}"'),
            (f"\tmean = {get_limited_precision_float_str(mean)}"),
            (f"\tstd = {get_limited_precision_float_str(std)}"),
        ]
        if tensor_type == "big_int_tensor_by_range":
            lines.append(f"\tmin_val = {tensor_info['min_val']}")
            lines.append(f"\tmax_val = {tensor_info['max_val']}")
        elif "data" in tensor_info:
            data_list = (
                tensor_info["data"].flatten()
                if isinstance(tensor_info["data"], torch.Tensor)
                else tensor_info["data"]
            )
            lines.append(f"\tdata = {format_data(data_list)}")

        lines.append("")
        return lines

    input_infos = converted["input_info"]
    if isinstance(input_infos, dict):
        input_infos = [input_infos]

    input_lines = []
    for idx, input_info in enumerate(input_infos):
        input_info["name"] = f"input_{idx}"
        input_lines.extend(process_tensor_info(input_info, name_prefix="Program_input"))

    with open(f"{file_path}/input_meta.py", "w") as f:
        f.write("\n".join(input_lines))

    weight_lines = []
    for name, weight_info in converted["weight_info"].items():
        weight_info["name"] = name
        weight_lines.extend(
            process_tensor_info(weight_info, name_prefix="Program_weight")
        )

    with open(f"{file_path}/weight_meta.py", "w") as f:
        f.write("\n".join(weight_lines))


def load_converted_from_text(file_path):
    input_info = list(convert_meta_classes_to_tensors(f"{file_path}/input_meta.py"))

    weight_info = {
        data["name"]: data
        for data in convert_meta_classes_to_tensors(f"{file_path}/weight_meta.py")
    }

    return {
        "input_info": input_info,
        "weight_info": weight_info,
        "dynamic_shapes": None,
    }


def convert_meta_classes_to_tensors(file_path):
    for name, cls in _get_classes(file_path):
        attrs = {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
        data_value = None
        data_type = getattr(torch, attrs.get("dtype", "torch.float").split(".")[-1])
        shape = attrs.get("shape", [])

        if "min_val" in attrs and "max_val" in attrs:
            min_val = attrs["min_val"]
            max_val = attrs["max_val"]
            # torch.randint's upper bound is exclusive, so add 1
            data_value = torch.randint(
                min_val, max_val + 1, size=shape, dtype=data_type
            )
        elif attrs.get("data") is not None:
            if isinstance(attrs.get("data"), str):
                raise ValueError("Unimplemented")
            else:
                data_value = torch.tensor(attrs["data"], dtype=data_type).reshape(
                    attrs.get("shape"), []
                )
        yield {
            "info": {
                "shape": attrs.get("shape", []),
                "dtype": data_type,
                "device": attrs.get("device", "cpu"),
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

    mean = info["info"]["mean"]
    std = info["info"]["std"]
    if "data" in info and info["data"] is not None:
        return info["data"].to(device)
    if dtype is torch.bool:
        return (torch.randn(size=shape) > 0.5).to(dtype).to(device)
    return torch.randn(size=shape).to(dtype).to(device) * std * 0.2 + mean


def update_device(code, device):
    if device == "cuda":
        pattern = r'device\(type="cpu"\)'
        replacement = f'device(type="cuda", index={torch.cuda.current_device()})'
        updated_code = re.sub(pattern, replacement, code)
        return updated_code
    else:
        pattern = r'device\(type="cuda"(?:, index=\d+)?\)'
        replacement = 'device(type="cpu")'
        updated_code = re.sub(pattern, replacement, code)
        return updated_code
