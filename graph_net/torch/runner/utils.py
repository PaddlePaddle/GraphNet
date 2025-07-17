import torch
import torch.nn as nn
from collections import OrderedDict
import os
import re

def convert_param_name(original_name):
    if original_name.endswith(('.weight', '.bias')):
        prefix = 'p_'
        base_name = original_name

    elif any(x in original_name for x in ['running_mean', 'running_var', 'num_batches_tracked']):
        prefix = 'b_'
        base_name = original_name
    else:
        raise ValueError(f"Unrecognized parameter type: {original_name}")
    
    if '.' in base_name:
        parts = base_name.split('.')
        if len(parts) == 2 and not parts[0].startswith('layer'):
            return prefix + parts[0] + '_' + parts[1]
        else:
            # layer1.0 -> layer1___0___
            pattern = r'(layer\d+)\.(\d+)\.'
            replacement = r'\1___\2___'
            converted = re.sub(pattern, replacement, base_name)
            converted = converted.replace('.', '_')
            return f"{prefix}getattr_l__self___{converted}"
    else:
        return prefix + base_name

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

    with open(file_path, 'r') as f:
        lines = f.readlines()

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

    for cls in classes:
        if 'input_' in cls["name"]:
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
            input_info.append(item)
        else:
            data_value = None
            data_type = getattr(torch, cls.get("dtype", "torch.float").split('.')[-1])
            if cls.get("data") is not None:
                if isinstance(cls.get("data"), str):
                    raise ValueError("Unimplemented")
                else:
                    data_value = torch.tensor(cls["data"], dtype=data_type).reshape(cls.get("shape"), [])
            weight_info[cls["name"]] = {
                "info": {
                    "shape": cls.get("shape", []),
                    "dtype": data_type,
                    "device": cls.get("device", "cpu"),
                    "mean": cls.get("mean", 0.0),
                    "std": cls.get("std", 1.0),
                },
                "data": data_value,
            }

    return {
        "input_info": input_info if len(input_info) > 0 else None,
        "weight_info": weight_info,
        "dynamic_shapes": None 
    }

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
