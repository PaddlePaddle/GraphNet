import importlib
import ast
import numpy as np
import paddle


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
    return [*weight_info, *input_info]


def convert_meta_classes_to_tensors(file_path):
    current_device = paddle.device.get_device()
    for name, cls in get_meta_classes(file_path):
        attrs = {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
        data_value = None
        data_type = getattr(paddle, attrs.get("dtype", "float32"))
        if attrs.get("data") is not None:
            if isinstance(attrs.get("data"), str):
                raise ValueError("Unimplemented")
            else:
                data_value = paddle.reshape(
                    paddle.to_tensor(attrs.get("data"), dtype=data_type),
                    attrs.get("shape", []),
                )
        yield {
            "info": {
                "shape": attrs.get("shape", []),
                "dtype": data_type,
                "device": attrs.get("device", current_device),
                "mean": attrs.get("mean", None),
                "std": attrs.get("std", None),
                "min_val": attrs.get("min_val", 0),
                "max_val": attrs.get("max_val", 2),
            },
            "data": data_value,
            "name": attrs.get("name"),
            "original_name": attrs.get("original_name", None),
        }


def get_meta_classes(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    class_names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]

    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)

    classes = [(name, getattr(unnamed, name)) for name in class_names]
    return classes


def extract_dynamic_shapes(example_inputs):
    pass


def init_integer_tensor(dtype, shape, min_val, max_val, use_numpy):
    if use_numpy:
        array = np.random.randint(
            low=min_val, high=max_val + 1, size=shape, dtype=dtype
        )
        return paddle.to_tensor(array)
    else:
        return paddle.randint(low=min_val, high=max_val + 1, shape=shape, dtype=dtype)


def init_float_tensor(shape, mean, std, min_val, max_val, use_numpy):
    tensor = None
    if use_numpy:
        if mean is not None and std is not None:
            # NumPy does not support truncated normal, we simulate it here.
            array = np.random.normal(0, 1, shape) * std * 0.2 + mean
            array = np.clip(array, min_val, max_val)
        else:
            array = np.random.uniform(low=min_val, high=max_val, size=shape)
        tensor = paddle.to_tensor(array)
    else:
        if mean is not None and std is not None:
            tensor = paddle.randn(shape, dtype="float32") * std * 0.2 + mean
            tensor = paddle.clip(tensor, min=min_val, max=max_val)
        else:
            tensor = paddle.uniform(
                shape=shape, dtype="float32", min=min_val, max=max_val
            )
    return tensor


def replay_tensor(info, use_numpy=True):
    device = info["info"]["device"]
    dtype = info["info"]["dtype"]
    shape = info["info"]["shape"]

    mean = info["info"]["mean"]
    std = info["info"]["std"]
    min_val = info["info"]["min_val"]
    max_val = info["info"]["max_val"]
    if None in shape:
        shape = list(map(lambda i: i if i is not None else 1, shape))
    if "data" in info and info["data"] is not None:
        return paddle.reshape(info["data"], shape).to(dtype).to(device)
    elif dtype in [paddle.int32, paddle.int64, paddle.bool]:
        init_dtype = "int32" if dtype == paddle.bool else "int64"
        if dtype == paddle.bool:
            min_val, max_val = 0, 1
        return (
            init_integer_tensor(init_dtype, shape, min_val, max_val, use_numpy)
            .to(dtype)
            .to(device)
        )
    else:
        tensor = init_float_tensor(shape, mean, std, min_val, max_val, use_numpy)
        return tensor.to(dtype).to(device)
