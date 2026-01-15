import torch
import ast
import math
import inspect
import importlib

kLiteralTensorSize = 64


def get_limited_precision_float_str(value):
    if not isinstance(value, float):
        return value
    if math.isnan(value):
        return "float('nan')"
    if math.isinf(value):
        return "float('inf')" if value > 0 else "float('-inf')"
    return f"{value:.3f}"


def convert_meta_classes_to_tensors(file_path):
    tensor_meta_attrs_list = [
        {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
        for name, cls in _get_classes(file_path)
    ]

    return convert_tensor_meta_attrs_list_to_tensors_wrappers(tensor_meta_attrs_list)


def convert_tensor_meta_attrs_list_to_tensors_wrappers(tensor_meta_attrs_list):
    for i, attrs in enumerate(tensor_meta_attrs_list):
        data_value = None
        data_type = getattr(torch, attrs.get("dtype", "torch.float").split(".")[-1])
        shape = attrs.get("shape", [])

        if (
            "min_val" in attrs
            and attrs.get("min_val") is not None
            and "max_val" in attrs
            and attrs.get("max_val") is not None
            and data_type
            in [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ]
        ):
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
                    attrs.get("shape", [])
                )
        info_dict = {
            "shape": attrs.get("shape", []),
            "dtype": data_type,
            "device": attrs.get("device", "cpu"),
            "mean": attrs.get("mean", 0.0),
            "std": attrs.get("std", 1.0),
        }
        # Include constraints if present (floats will be clamped in replay_tensor)
        if attrs.get("min_val") is not None:
            info_dict["min_val"] = attrs.get("min_val")
        if attrs.get("max_val") is not None:
            info_dict["max_val"] = attrs.get("max_val")

        yield {
            "info": info_dict,
            "data": data_value,
            "name": attrs.get("name"),
        }


def _get_classes(file_path):
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    yield from inspect.getmembers(unnamed, inspect.isclass)


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
    if std is None:
        std = 0.1
    if mean is None:
        mean = 0
    # Handle std = 0 case to avoid generating identical values
    if std == 0:
        tensor = torch.full(size=shape, fill_value=mean, dtype=dtype, device=device)
    else:
        tensor = torch.randn(size=shape).to(dtype).to(device) * std * 0.2 + mean

    # Apply lower/upper bound constraints if present
    if "min_val" in info["info"]:
        min_val = info["info"]["min_val"]
        tensor = torch.clamp(tensor, min=min_val)
    if "max_val" in info["info"]:
        max_val = info["info"]["max_val"]
        tensor = torch.clamp(tensor, max=max_val)

    # Additional numerical stability checks
    if dtype.is_floating_point:
        # Replace any inf or nan values with small random values
        tensor = torch.where(
            torch.isfinite(tensor), tensor, torch.randn_like(tensor) * 0.01
        )
        # Ensure no extremely large values
        tensor = torch.clamp(tensor, min=-100.0, max=100.0)

    return tensor


def get_dummy_tensor(info):
    device = info["info"]["device"]
    dtype = info["info"]["dtype"]
    shape = info["info"]["shape"]

    if "data" in info and info["data"] is not None:
        return info["data"].to(device)
    return torch.empty(shape, dtype=dtype, device=device)


def modify_code_by_device(code, new_device_str):
    tree = ast.parse(code)

    class DeviceReplacer(ast.NodeTransformer):
        def __init__(self, new_device):
            super().__init__()
            self.new_device = new_device

        def visit_Call(self, node):
            # device.type("device")
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "device"
                and node.func.attr == "type"
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    node.args[0].value = self.new_device
                return node

            # .to(device(type="device"))
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == "device"
            ):
                device_call = node.args[0]
                for keyword in device_call.keywords:
                    if (
                        keyword.arg == "type"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        keyword.value.value = self.new_device
                return node

            # device=device(type="device")
            new_keywords = []
            for keyword in node.keywords:
                if (
                    keyword.arg == "device"
                    and isinstance(keyword.value, ast.Call)
                    and isinstance(keyword.value.func, ast.Name)
                    and keyword.value.func.id == "device"
                ):
                    device_call = keyword.value
                    for kw in device_call.keywords:
                        if (
                            kw.arg == "type"
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ):
                            kw.value.value = self.new_device
                    new_keywords.append(keyword)
                else:
                    new_keywords.append(keyword)
            node.keywords = new_keywords
            return self.generic_visit(node)

    transformer = DeviceReplacer(new_device_str)
    modified_tree = transformer.visit(tree)
    ast.fix_missing_locations(modified_tree)
    return ast.unparse(modified_tree)
