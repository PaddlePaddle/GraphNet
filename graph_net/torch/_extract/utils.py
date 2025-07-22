import re
import torch
import torch.nn as nn
from collections import OrderedDict
import uuid
import json
import os
import argparse

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
