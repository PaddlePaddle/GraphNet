import torch
import torch.nn as nn
from collections import OrderedDict
import os
import re
import importlib
import inspect

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
