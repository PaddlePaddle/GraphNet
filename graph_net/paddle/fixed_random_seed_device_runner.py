import paddle
import os
import time
import numpy as np
import random
import platform
import traceback
import argparse
import sys
from pathlib import Path
import json
from contextlib import redirect_stdout, redirect_stderr

from graph_net.paddle import utils
from graph_net import test_compiler_util
from graph_net import path_utils


def set_seed(random_seed):
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def get_hardware_name(device):
    if device == "cuda":
        return paddle.device.cuda.get_device_name(0)
    elif device == "cpu":
        return platform.processor()
    return "unknown"


def load_model(model_path):
    if os.path.isdir(model_path):
        model_dirs = []
        for root, dirs, files in os.walk(model_path):
            if "model.py" in files:
                model_dirs.append(root)

        if not model_dirs:
            raise FileNotFoundError(f"No valid model directories found in {model_path}")

        model_path = model_dirs[0]

    from importlib.util import spec_from_loader, module_from_spec

    model_file = f"{model_path}/model.py"
    with open(model_file, "r") as f:
        code = f.read()

    module_name = Path(model_file).stem
    spec = spec_from_loader(module_name, loader=None)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    exec(compile(f"import paddle\n{code}", model_file, "exec"), module.__dict__)

    return module.GraphModule()


def get_input_dict(model_path):
    if os.path.isdir(model_path):
        model_dirs = []
        for root, dirs, files in os.walk(model_path):
            if "model.py" in files:
                model_dirs.append(root)

        if not model_dirs:
            raise FileNotFoundError(f"No valid model directories found in {model_path}")

        model_path = model_dirs[0]

    inputs_params = utils.load_converted_from_text(model_path)
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]
    params.update(inputs)
    return {k: utils.replay_tensor(v) for k, v in params.items()}


def get_input_spec(model_path):
    if os.path.isdir(model_path):
        model_dirs = []
        for root, dirs, files in os.walk(model_path):
            if "model.py" in files:
                model_dirs.append(root)

        if not model_dirs:
            raise FileNotFoundError(f"No valid model directories found in {model_path}")

        model_path = model_dirs[0]

    inputs_params_list = utils.load_converted_list_from_text(model_path)
    input_spec = [None] * len(inputs_params_list)
    for i, v in enumerate(inputs_params_list):
        dtype = v["info"]["dtype"]
        shape = v["info"]["shape"]
        input_spec[i] = paddle.static.InputSpec(shape, dtype)
    return input_spec


def get_compiled_model(model, compiler, model_path):
    if compiler == "nope":
        return model
    input_spec = get_input_spec(model_path)
    build_strategy = paddle.static.BuildStrategy()
    compiled_model = paddle.jit.to_static(
        model,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )
    compiled_model.eval()
    return compiled_model


def measure_performance(model_call, synchronizer_func, warmup, trials):
    outputs = None
    for i in range(warmup):
        out_run = model_call()
        if i == 0:
            outputs = out_run

    synchronizer_func()

    e2e_times = []
    for i in range(trials):
        duration_box = test_compiler_util.DurationBox(-1)
        with test_compiler_util.naive_timer(duration_box, synchronizer_func):
            model_call()
        e2e_times.append(duration_box.value)
        # This print will be captured by the log redirect in main
        print(f"Trial {i + 1}: e2e={duration_box.value:.4f} ms", flush=True)

    return outputs, test_compiler_util.get_timing_stats(e2e_times)


def save_outputs_to_json(outputs, file_path):
    """
    Serializes paddle tensor outputs to a JSON file.
    """
    if isinstance(outputs, paddle.Tensor):
        outputs = [outputs]

    serializable_outputs = []
    if outputs is not None:
        for tensor in outputs:
            if tensor is not None:
                # Convert tensor to numpy array, then to nested Python list
                serializable_outputs.append(tensor.numpy().tolist())
            else:
                serializable_outputs.append(None)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_outputs, f, indent=4)


def test_single_model(args, model_path):
    set_seed(123)

    # These prints will be captured by the log redirect in main
    print(f"[Config] device: {args.device}", flush=True)
    print(f"[Config] compiler: {args.compiler}", flush=True)
    print(f"[Config] hardware: {get_hardware_name(args.device)}", flush=True)
    print(f"[Config] framework_version: {paddle.__version__}", flush=True)
    print(f"[Config] warmup: {args.warmup}", flush=True)
    print(f"[Config] trials: {args.trials}", flush=True)

    outputs = None
    success = False
    try:
        synchronizer_func = paddle.device.synchronize

        input_dict = get_input_dict(model_path)
        model = load_model(model_path)
        model.eval()

        print(f"Run model with compiler: {args.compiler}", flush=True)
        if args.compiler == "nope":
            compiled_model = model
        else:
            compiled_model = get_compiled_model(model, args.compiler, model_path)

        outputs, time_stats = measure_performance(
            lambda: compiled_model(**input_dict),
            synchronizer_func,
            args.warmup,
            args.trials,
        )
        success = True

        print(f"[Result] model_path: {model_path}", flush=True)
        print(f"[Result] compiler: {args.compiler}", flush=True)
        print(f"[Result] device: {args.device}", flush=True)
        print(f"[Result] e2e_mean: {time_stats['mean']:.5f}", flush=True)
        print(f"[Result] e2e_std: {time_stats['std']:.5f}", flush=True)

    except Exception as e:
        print(f"Run model failed: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        return False, None

    return success, outputs


def main():
    parser = argparse.ArgumentParser(
        description="Test device performance with fixed random seeds"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for testing (e.g., 'cpu', 'cuda', or 'dcu')",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        required=False,
        default="cinn",
        help="Compiler backend to use (cinn or nope)",
    )
    parser.add_argument(
        "--warmup", type=int, required=False, default=5, help="Number of warmup steps"
    )
    parser.add_argument(
        "--trials", type=int, required=False, default=10, help="Number of timing trials"
    )
    parser.add_argument(
        "--allow-list",
        type=str,
        required=False,
        default=None,
        help="Path to allow list file",
    )
    parser.add_argument(
        "--test-device-path",
        type=str,
        required=True,
        help="Path to save logs and output json files",
    )

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.test_device_path, exist_ok=True)

    test_samples = []
    if args.allow_list is not None:
        assert os.path.isfile(args.allow_list)
        graphnet_root = path_utils.get_graphnet_root()
        print(f"graphnet_root: {graphnet_root}")
        test_samples = []
        with open(args.allow_list, "r") as f:
            for line in f.readlines():
                test_samples.append(os.path.join(graphnet_root, line.strip()))

    sample_idx = 0
    failed_samples = []

    # --- MODIFICATION START ---
    # Store the root path object
    root_path_obj = Path(args.model_path)
    # --- MODIFICATION END ---

    for model_path in path_utils.get_recursively_model_path(args.model_path):
        if not test_samples or os.path.abspath(model_path) in test_samples:
            # --- MODIFICATION START ---
            # Get relative path from the root model_path
            relative_path = Path(model_path).relative_to(root_path_obj)
            # Replace path separators (like / or \) with _ to create a flat file name
            model_name = str(relative_path).replace(os.path.sep, "_")
            # --- MODIFICATION END ---

            log_file_path = os.path.join(args.test_device_path, f"{model_name}.log")
            json_file_path = os.path.join(
                args.test_device_path, f"{model_name}_outputs.json"
            )

            # --- Redirect stdout/stderr to log file ---
            with open(log_file_path, "w", encoding="utf-8") as log_f:
                with redirect_stdout(log_f), redirect_stderr(log_f):
                    print(
                        f"[{sample_idx}] fixed_random_seed_device_runner, model_path: {model_path}",
                        flush=True,
                    )

                    success, outputs = test_single_model(args, model_path)

                    if not success:
                        failed_samples.append(model_path)
                    else:
                        # Save outputs to JSON
                        try:
                            save_outputs_to_json(outputs, json_file_path)
                        except Exception as e:
                            print(
                                f"Failed to save outputs to JSON {json_file_path}: {str(e)}",
                                flush=True,
                            )
                            print(traceback.format_exc(), flush=True)
                            failed_samples.append(model_path)  # Count as failure

            sample_idx += 1

    # --- Print final summary to console ---
    print(
        f"Totally {sample_idx} verified samples, failed {len(failed_samples)} samples."
    )
    for model_path in failed_samples:
        print(f"- {model_path}")

    return 0 if len(failed_samples) == 0 else 1


if __name__ == "__main__":
    main()
