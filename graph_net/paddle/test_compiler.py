from . import utils
import argparse
import importlib.util
import inspect
import paddle
from pathlib import Path
from typing import Type, Any
import sys
import os
import os.path
from dataclasses import dataclass
from contextlib import contextmanager
import time
import numpy as np


def load_class_from_file(file_path: str, class_name: str):
    file = Path(file_path).resolve()
    module_name = file.stem

    with open(file_path, "r", encoding="utf-8") as f:
        original_code = f.read()
    import_stmt = "import paddle"
    modified_code = f"{import_stmt}\n{original_code}"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    compiled_code = compile(modified_code, filename=file, mode="exec")
    exec(compiled_code, module.__dict__)

    model_class = getattr(module, class_name, None)
    return model_class


def get_synchronizer_func(args):
    assert args.compiler == "default"
    return paddle.device.synchronize


def get_model(args):
    model_class = load_class_from_file(
        f"{args.model_path}/model.py", class_name="GraphModule"
    )
    return model_class()


def get_input_dict(args):
    inputs_params = utils.load_converted_from_text(f"{args.model_path}")
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]

    params.update(inputs)
    state_dict = {k: utils.replay_tensor(v) for k, v in params.items()}
    return state_dict


@dataclass
class DurationBox:
    value: int


@contextmanager
def naive_timer(duration_box, get_synchronizer_func):
    get_synchronizer_func()
    start = time.time()
    yield
    get_synchronizer_func()
    end = time.time()
    duration_box.value = end - start


def get_input_spec(args):
    inputs_params_list = utils.load_converted_list_from_text(f"{args.model_path}")
    input_spec = [None] * len(inputs_params_list)
    for i, v in enumerate(inputs_params_list):
        dtype = v["info"]["dtype"]
        shape = v["info"]["shape"]
        input_spec[i] = paddle.static.InputSpec(shape, dtype)
    return input_spec


def test_single_model(args):
    synchronizer_func = get_synchronizer_func(args)
    input_dict = get_input_dict(args)
    model_dy = get_model(args)
    input_spec = get_input_spec(args)
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = False

    # eager
    model = paddle.jit.to_static(
        model_dy,
        full_graph=False,
    )
    model.eval()
    for _ in range(args.warmup if args.warmup > 0 else 0):
        model(**input_dict)
    eager_duration_box = DurationBox(-1)
    with naive_timer(eager_duration_box, synchronizer_func):
        expected_out = model(**input_dict)

    # compiled
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    compiled_model = paddle.jit.to_static(
        model_dy,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )
    compiled_model.eval()
    for _ in range(args.warmup if args.warmup > 0 else 0):
        compiled_model(**input_dict)
    compiled_duration_box = DurationBox(-1)
    with naive_timer(compiled_duration_box, synchronizer_func):
        compiled_out = compiled_model(**input_dict)
    if isinstance(expected_out, paddle.Tensor):
        expected_out = expected_out.numpy()
        compiled_out = compiled_out.numpy()
    elif isinstance(expected_out, list) or isinstance(expected_out, tuple):
        expected_out = expected_out[0].numpy()
        compiled_out = compiled_out[0].numpy()
    else:
        raise ValueError("Illegal return value.")

    def print_cmp(key, func, **kwargs):
        cmp_ret = func(expected_out, compiled_out, **kwargs)
        print(
            f"{args.log_prompt} {key} model_path:{args.model_path} {cmp_ret}",
            file=sys.stderr,
        )

    print_cmp("cmp.equal", get_cmp_equal)
    print_cmp("cmp.all_close_atol8_rtol8", get_cmp_all_close, atol=1e-8, rtol=1e-8)
    print_cmp("cmp.all_close_atol8_rtol5", get_cmp_all_close, atol=1e-8, rtol=1e-5)
    print_cmp("cmp.all_close_atol5_rtol5", get_cmp_all_close, atol=1e-5, rtol=1e-5)
    print_cmp("cmp.all_close_atol3_rtol2", get_cmp_all_close, atol=1e-3, rtol=1e-2)
    print_cmp("cmp.all_close_atol2_rtol1", get_cmp_all_close, atol=1e-2, rtol=1e-1)
    print_cmp("cmp.max_diff", get_cmp_max_diff)
    print_cmp("cmp.mean_diff", get_cmp_mean_diff)
    print_cmp("cmp.diff_count_atol8_rtol8", get_cmp_diff_count, atol=1e-8, rtol=1e-8)
    print_cmp("cmp.diff_count_atol8_rtol5", get_cmp_diff_count, atol=1e-8, rtol=1e-5)
    print_cmp("cmp.diff_count_atol5_rtol5", get_cmp_diff_count, atol=1e-5, rtol=1e-5)
    print_cmp("cmp.diff_count_atol3_rtol2", get_cmp_diff_count, atol=1e-3, rtol=1e-2)
    print_cmp("cmp.diff_count_atol2_rtol1", get_cmp_diff_count, atol=1e-2, rtol=1e-1)

    print(
        f"{args.log_prompt} duration model_path:{args.model_path} eager:{eager_duration_box.value} compiled:{compiled_duration_box.value}",
        file=sys.stderr,
    )


def get_cmp_equal(expected_out, compiled_out):
    return " ".join(
        str(int(np.sum(np.equal(a, b)))) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_all_close(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(int(np.allclose(a, b, atol=atol, rtol=rtol)))
        for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_max_diff(expected_out, compiled_out):
    return " ".join(
        str(np.max(np.abs(a - b)).item()) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_mean_diff(expected_out, compiled_out):
    return " ".join(
        str(np.mean(np.abs(a - b)).item()) for a, b in zip(expected_out, compiled_out)
    )


def get_cmp_diff_count(expected_out, compiled_out, atol, rtol):
    return " ".join(
        str(np.sum(~np.isclose(a, b, atol=atol, rtol=rtol)).item())
        for a, b in zip(expected_out, compiled_out)
    )


def test_multi_models(args):
    for model_path in get_recursively_model_path(args.model_path):
        cmd = "".join(
            [
                sys.executable,
                "-m graph_net.paddle.test_compiler",
                f"--model-path {model_path}",
                f"--compiler {args.compiler}",
                f"--warmup {args.warmup}",
                f"--log-prompt {args.log_prompt}",
            ]
        )
        cmd_ret = os.system(cmd)
        assert cmd_ret == 0, f"{cmd_ret=}, {cmd=}"


def get_recursively_model_path(root_dir):
    for sub_dir in get_immediate_subdirectory_paths(root_dir):
        if is_single_model_dir(sub_dir):
            yield sub_dir
        else:
            yield from get_recursively_model_path(sub_dir)


def get_immediate_subdirectory_paths(parent_dir):
    return [
        sub_dir
        for name in os.listdir(parent_dir)
        for sub_dir in [os.path.join(parent_dir, name)]
        if os.path.isdir(sub_dir)
    ]


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def main(args):
    assert os.path.isdir(args.model_path)
    if is_single_model_dir(args.model_path):
        test_single_model(args)
    else:
        test_multi_models(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compiler performance.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file(s), each subdirectory containing graph_net.json will be regarded as a model",
    )
    parser.add_argument(
        "--compiler",
        type=str,
        required=False,
        default="default",
        help="Path to customized compiler python file",
    )
    parser.add_argument(
        "--warmup", type=int, required=False, default=5, help="Number of warmup steps"
    )
    parser.add_argument(
        "--log-prompt",
        type=str,
        required=False,
        default="graph-net-test-compiler-log",
        help="Log prompt for performance log filtering.",
    )
    args = parser.parse_args()
    main(args=args)
