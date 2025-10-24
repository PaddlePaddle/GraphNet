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

from graph_net.paddle import utils
from graph_net import test_compiler_util


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
    for _ in range(warmup):
        model_call()
    synchronizer_func()

    e2e_times = []
    for i in range(trials):
        duration_box = test_compiler_util.DurationBox(-1)
        with test_compiler_util.naive_timer(duration_box, synchronizer_func):
            model_call()
        e2e_times.append(duration_box.value)
        print(f"Trial {i + 1}: e2e={duration_box.value:.4f} ms")

    return test_compiler_util.get_timing_stats(e2e_times)


def main():
    parser = argparse.ArgumentParser(description="Test device performance with fixed random seeds")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for testing (e.g., 'cpu', 'cuda', or 'dcu')"
    )
    parser.add_argument(
        "--compiler",
        type=str,
        required=False,
        default="cinn",
        help="Compiler backend to use (cinn or nope)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        required=False,
        default=5,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--trials",
        type=int,
        required=False,
        default=10,
        help="Number of timing trials"
    )
    
    args = parser.parse_args()
    
    set_seed(123)
    
    print(f"[Config] device: {args.device}")
    print(f"[Config] compiler: {args.compiler}")
    print(f"[Config] hardware: {get_hardware_name(args.device)}")
    print(f"[Config] framework_version: {paddle.__version__}")
    print(f"[Config] warmup: {args.warmup}")
    print(f"[Config] trials: {args.trials}")

    success = False
    try:
        synchronizer_func = paddle.device.synchronize
        
        input_dict = get_input_dict(args.model_path)
        model = load_model(args.model_path)
        model.eval()

        print(f"Run model with compiler: {args.compiler}")
        if args.compiler == "nope":
            compiled_model = model
        else:
            compiled_model = get_compiled_model(model, args.compiler, args.model_path)
        
        time_stats = measure_performance(
            lambda: compiled_model(**input_dict), 
            synchronizer_func, 
            args.warmup, 
            args.trials
        )
        success = True
        
        print(f"[Result] model_path: {args.model_path}")
        print(f"[Result] compiler: {args.compiler}")
        print(f"[Result] device: {args.device}")
        print(f"[Result] e2e_mean: {time_stats['mean']:.5f}")
        print(f"[Result] e2e_std: {time_stats['std']:.5f}")
        
    except Exception as e:
        print(f"Run model failed: {str(e)}")
        print(traceback.format_exc())
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    main()