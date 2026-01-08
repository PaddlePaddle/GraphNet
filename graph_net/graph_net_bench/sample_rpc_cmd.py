#!/usr/bin/env python3

import os
import sys
import importlib.util
from pathlib import Path

import torch
import numpy as np


def load_model_and_weights(model_path: str):
    model_file = Path(model_path) / "model.py"
    if not model_file.exists():
        raise FileNotFoundError(f"model.py not found in {model_path}")

    spec = importlib.util.spec_from_file_location("remote_model_module", str(model_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'GraphModule'):
        raise ValueError("model.py must define 'GraphModule' class")

    weight_tensors = {}
    weight_meta_file = Path(model_path) / "weight_meta.py"
    if weight_meta_file.exists():
        spec = importlib.util.spec_from_file_location(
            "weight_meta_module", str(weight_meta_file)
        )
        weight_meta_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(weight_meta_module)

        tensor_classes = [
            getattr(weight_meta_module, name)
            for name in dir(weight_meta_module)
            if name.startswith("Program_weight_tensor_meta_")
        ]

        for tensor_cls in tensor_classes:
            name = tensor_cls.name
            shape = tensor_cls.shape
            dtype_name = tensor_cls.dtype.replace("torch.", "")
            dtype = getattr(torch, dtype_name)
            device = getattr(tensor_cls, 'device', 'cpu')

            if tensor_cls.data is not None:
                np_array = np.array(tensor_cls.data, dtype=np.dtype(dtype_name))
                np_array = np_array.reshape(shape)
                weight_tensors[name] = torch.from_numpy(np_array).to(device)
            else:
                if dtype == torch.bool:
                    weight_tensors[name] = torch.zeros(shape, dtype=dtype, device=device)
                else:
                    weight_tensors[name] = torch.randn(shape, dtype=dtype, device=device)

    model = module.GraphModule()
    for name, tensor in weight_tensors.items():
        param = getattr(model, name, None)
        if param is not None and isinstance(param, torch.Tensor):
            param.data.copy_(tensor)

    return model, weight_tensors


def get_forward_inputs(model_path: str, weight_tensors: dict):
    import inspect

    model_file = Path(model_path) / "model.py"
    spec = importlib.util.spec_from_file_location("remote_model_module", str(model_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    forward_params = inspect.signature(module.GraphModule.forward).parameters
    param_names = [
        name for name in forward_params.keys()
        if name != 'self'
    ]

    inputs = []
    for param_name in param_names:
        if param_name in weight_tensors:
            inputs.append(weight_tensors[param_name])
        else:
            raise ValueError(f"Missing weight tensor for parameter: {param_name}")

    return inputs


def save_outputs_as_npz(outputs, output_path: str):
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    np_arrays = {}
    for i, tensor in enumerate(outputs):
        key = f"output_{i}"
        np_arrays[key] = tensor.cpu().numpy()

    np.savez(output_path, **np_arrays)


def main():
    input_workspace = os.environ.get("INPUT_WORKSPACE")
    output_workspace = os.environ.get("OUTPUT_WORKSPACE")
    output_file_name = os.environ.get("OUTPUT_FILE_NAME")
    output_file_path = os.environ.get("OUTPUT_FILE_PATH")
    seed_str = os.environ.get("RANDOM_SEED")

    if not input_workspace:
        raise RuntimeError("INPUT_WORKSPACE environment variable not set")
    if not output_workspace:
        raise RuntimeError("OUTPUT_WORKSPACE environment variable not set")
    if not output_file_name:
        raise RuntimeError("OUTPUT_FILE_NAME environment variable not set")

    print(f"INPUT_WORKSPACE: {input_workspace}", file=sys.stderr)
    print(f"OUTPUT_WORKSPACE: {output_workspace}", file=sys.stderr)
    print(f"OUTPUT_FILE_NAME: {output_file_name}", file=sys.stderr)
    if output_file_path:
        print(f"OUTPUT_FILE_PATH: {output_file_path}", file=sys.stderr)

    if seed_str:
        seed = int(seed_str)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"RANDOM_SEED: {seed}", file=sys.stderr)

    print("Loading model and weights...", file=sys.stderr)
    model, weight_tensors = load_model_and_weights(input_workspace)
    print(f"Model loaded, {len(weight_tensors)} weight tensors", file=sys.stderr)

    print("Preparing inputs...", file=sys.stderr)
    inputs = get_forward_inputs(input_workspace, weight_tensors)
    print(f"Prepared {len(inputs)} inputs for forward()", file=sys.stderr)

    print("Running inference...", file=sys.stderr)
    model.eval()
    with torch.no_grad():
        outputs = model(*inputs)

    output_path = output_file_path or os.path.join(output_workspace, output_file_name)
    print(f"Saving outputs to {output_path}...", file=sys.stderr)
    save_outputs_as_npz(outputs, output_path)
    print("Outputs saved successfully!", file=sys.stderr)


if __name__ == "__main__":
    main()