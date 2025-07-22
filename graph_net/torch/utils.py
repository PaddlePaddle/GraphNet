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

dyn_template = """
%MODULE
"""

def indent_with_tab(code: str) -> str:
    lines = code.splitlines()
    indented_lines = [f"    {line}" for line in lines]
    return "\n".join(indented_lines)

def apply_templates(code: str) -> str:
    code = indent_with_tab(code)
    code = code.replace("    GraphModule()", "class GraphModule(torch.nn.Module):")
    code = code.replace("    \n" * 3, "\n")
    py_code = dyn_template.replace('%MODULE', code)
    return py_code

def convert_state_and_inputs_and_ctrls(state_dict, example_inputs, ctrls):
    def tensor_info(tensor):
        is_float = tensor.dtype.is_floating_point
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": float(tensor.mean().item()) if is_float else None,
            "std": float(tensor.std().item()) if is_float else None,
        }

    def process_tensor(tensor):
        if not isinstance(tensor, torch.Tensor):
            return {"type": "unknown", "value": tensor}

        info = tensor_info(tensor)
        if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            if tensor.numel() < 1024:
                return {"type": "small_int_tensor", "data": tensor.clone(), "info": info}
            else:
                return {"type": "big_int_tensor", "data": tensor.clone(), "info": info}
        elif tensor.numel() < 1024:
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
        data_value = None
        data_type = "random_tensor"
        if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            if tensor.numel() < 1024:
                data_type = "small_int_tensor"
                data_value = tensor.clone()
            else:
                data_type = "big_int_tensor"
        info = tensor_info(tensor)
        return {
            "info": info,
            "data": data_value,
            "type": data_type
        }
 
    processed_weights = {
        key: handle_named_tensors(tensor) for key, tensor in state_dict.items()
    }

    processed_ctrls = {
        key: handle_named_tensors(tensor) for key, tensor in ctrls.items()
    }

    # dynamic_shapes = extract_dynamic_shapes(example_inputs)
    return {
        "input_info": processed_inputs,
        "weight_info": processed_weights,
        "ctrl_info": processed_ctrls,
        "dynamic_shapes": None
    }

def convert_state_and_inputs(state_dict, example_inputs):
    return convert_state_and_inputs_and_ctrls(state_dict, example_inputs, {})

def save_constraints_text(converted, file_path):
    lines = []
    if converted["dynamic_shapes"] is not None:
        raise NotImplementedError("Handling constraints is not implemented yet.")
    with open(file_path, 'w') as f:
        f.write("\n".join(lines))

def save_converted_to_text(converted, file_path):
    def generate_uid():
        return str(uuid.uuid4()).replace('-', '')

    def format_data(data):
        if data is None:
            return "None"
        elif isinstance(data, torch.Tensor):
            if data.dtype.is_floating_point:
                return "[{}]".format(", ".join(f'{x:.6f}' for x in data.tolist()))
            else:
                return "[{}]".format(", ".join(f'{x}' for x in data.tolist()))
        else:
            return repr(data)

    def process_tensor_info(tensor_info, name_prefix="example_input"):
        data_list = None
        if "input_" in tensor_info["name"]:
            if tensor_info["type"] in ["small_tensor", "small_int_tensor"]:
                data_list = tensor_info["data"].flatten()
            elif tensor_info["type"] == "big_int_tensor":
                data_list = f'pt-filename:xxx-key'
            else:
                pass
        else:
            if tensor_info["type"] ==  "small_int_tensor":
                data_list = tensor_info["data"].flatten()
            if tensor_info["type"] ==  "big_int_tensor":
                raise ValueError("Unexpected cases: there are weights in big tensor of int type ")
        info = tensor_info.get("info", {})
        dtype = info.get("dtype", "torch.float")
        shape = info.get("shape", [])
        device = info.get("device", "cpu")
        mean = info.get("mean", 0.0)
        std = info.get("std", 1.0)
        uid = f"{name_prefix}_tensor_meta_{generate_uid()}"
        return [
            (f"class {uid}:"),
            (f"\tname = \"{tensor_info.get('name', '')}\""),
            (f"\tshape = {shape}"),
            (f"\tdtype = \"{dtype}\""),
            (f"\tdevice = \"{device}\""),
            (f"\tmean = {mean}"),
            (f"\tstd = {std}"),
            (f"\tdata = {format_data(data_list)}"),
            ("")
        ]

    input_infos = converted["input_info"]
    if isinstance(input_infos, dict):
        input_infos = [input_infos]

    input_lines = []
    for idx, input_info in enumerate(input_infos):
        input_info["name"] = f"input_{idx}"
        input_lines.extend(process_tensor_info(input_info, name_prefix="Program_input"))

    with open(f"{file_path}/input_meta.py", 'w') as f:
        f.write("\n".join(input_lines))

    weight_lines = []
    for name, weight_info in converted["weight_info"].items():
        weight_info["name"] = name
        weight_lines.extend(process_tensor_info(weight_info, name_prefix="Program_weight"))

    with open(f"{file_path}/weight_meta.py", 'w') as f:
        f.write("\n".join(weight_lines))

    ctrl_lines = []
    for name, ctrl_info in converted["ctrl_info"].items():
        ctrl_info["name"] = name
        ctrl_lines.extend(process_tensor_info(ctrl_info, name_prefix="Program_ctrl"))

    with open(f"{file_path}/ctrl_meta.py", 'w') as f:
        f.write("\n".join(ctrl_lines))

def load_converted_from_text(file_path):

    def parse_value(value_str):
        value_str = value_str.strip()
        if value_str == "None":
            return None
        if value_str == "[]":
            return []
        elif value_str.startswith('"') or value_str.startswith("'"):
            return value_str[1:-1]
        elif value_str.startswith('['):
            elements = value_str[1:-1].split(',')
            result = []
            for e in elements:
                e = e.strip()
                try:
                    result.append(eval(e))
                except:
                    result.append(e)
            return result
        else:
            try:
                return eval(value_str)
            except:
                return value_str

    with open(f"{file_path}/input_meta.py", 'r') as f:
        lines = f.readlines()
    with open(f"{file_path}/weight_meta.py", 'r') as f:
        lines += f.readlines()
    classes = []
    current_class = None

    for line in lines:
        line = line.strip()
        if line.startswith("class "):
            if current_class is not None:
                classes.append(current_class)
            current_class = {}
        elif "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            current_class[key] = parse_value(val)

    if current_class is not None:
        classes.append(current_class)

    input_info = []
    weight_info = {}

    def get_input_info(cls):
        item = {
            "type": "random_tensor",
            "info": {
                "shape": cls.get("shape", []),
                "dtype": getattr(torch, cls.get("dtype", "torch.float").split('.')[-1]),
                "device": cls.get("device", "cpu"),
                "mean": cls.get("mean", 0.0),
                "std": cls.get("std", 1.0),
            }
        }
        if cls.get("data") is not None:
            if isinstance(cls.get("data"), str):
                pass
            else:
                item["data"] = torch.tensor(cls["data"], dtype=item["info"]["dtype"]).reshape(cls.get("shape"), [])
        return item
    
    def get_weight_info(cls):
        data_value = None
        data_type = getattr(torch, cls.get("dtype", "torch.float").split('.')[-1])
        if cls.get("data") is not None:
            if isinstance(cls.get("data"), str):
                raise ValueError("Unimplemented")
            else:
                data_value = torch.tensor(cls["data"], dtype=data_type).reshape(cls.get("shape"), [])
        return {
            "info": {
                "shape": cls.get("shape", []),
                "dtype": data_type,
                "device": cls.get("device", "cpu"),
                "mean": cls.get("mean", 0.0),
                "std": cls.get("std", 1.0),
            },
            "data": data_value,
        }

    for cls in classes:
        if 'input_' in cls["name"]:
            input_info.append(get_input_info(cls))
        else:
            weight_info[cls["name"]] = get_weight_info(cls)

    ctrl_info = {
        data['name']: data
        for data in convert_meta_classes_to_tensors(f"{file_path}/ctrl_meta.py")
    }

    return {
        "input_info": input_info if len(input_info) > 0 else None,
        "weight_info": weight_info,
        "ctrl_info": ctrl_info,
        "dynamic_shapes": None 
    }

def convert_meta_classes_to_tensors(file_path):
    for name, cls in _get_classes(file_path):
        attrs = {
            k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v)
        }
        data_value = None
        data_type = getattr(torch, attrs.get("dtype", "torch.float").split('.')[-1])
        if attrs.get("data") is not None:
            if isinstance(attrs.get("data"), str):
                raise ValueError("Unimplemented")
            else:
                data_value = torch.tensor(attrs["data"], dtype=data_type).reshape(attrs.get("shape"), [])
        yield {
            "info": {
                "shape": attrs.get("shape", []),
                "dtype": data_type,
                "device": attrs.get("device", "cpu"),
                "mean": attrs.get("mean", 0.0),
                "std": attrs.get("std", 1.0),
            },
            "data": data_value,
            "name": attrs.get("name")
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
    return torch.randn(size=shape).to(dtype).to(device) * std * 0.2 + mean
