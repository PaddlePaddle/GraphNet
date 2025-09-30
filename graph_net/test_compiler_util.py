import os
import re
import sys
import json
import time
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class DurationBox:
    value: int


@contextmanager
def naive_timer(duration_box, synchronizer_func):
    synchronizer_func()
    start = time.time()
    yield
    synchronizer_func()
    end = time.time()
    duration_box.value = (end - start) * 1000  # Store in milliseconds


def get_timing_stats(elapsed_times):
    stats = {
        "mean": float(f"{np.mean(elapsed_times):.6g}"),
        "std": float(f"{np.std(elapsed_times):.6g}"),
        "min": float(f"{np.min(elapsed_times):.6g}"),
        "max": float(f"{np.max(elapsed_times):.6g}"),
    }
    return stats


def get_model_name(model_path):
    model_name = None
    with open(os.path.join(model_path, "graph_net.json"), "r") as f:
        data = json.load(f)
        model_name = data.get("model_name", None)

    if model_name is not None:
        fields = model_path.split(os.sep)
        pattern = rf"^subgraph(_\d+)?$"
        model_name = fields[-2] if re.match(pattern, fields[-1]) else fields[-1]
    return model_name


def get_subgraph_tag(model_path):
    fields = model_path.split(os.sep)
    pattern = rf"^subgraph(_\d+)?$"
    return fields[-1] if re.match(pattern, fields[-1]) else ""


def print_with_log_prompt(key, value, log_prompt):
    print(f"{log_prompt} {key} {value}", file=sys.stderr, flush=True)


def print_basic_config(args, hardware_name, compile_framework_version):
    model_path = os.path.normpath(args.model_path)
    print_with_log_prompt("[Processing]", model_path, args.log_prompt)

    model_name = get_model_name(model_path)
    print_with_log_prompt("[Config] model:", model_name, args.log_prompt)

    print_with_log_prompt("[Config] device:", args.device, args.log_prompt)
    print_with_log_prompt("[Config] hardware:", hardware_name, args.log_prompt)
    print_with_log_prompt("[Config] compiler:", args.compiler, args.log_prompt)
    print_with_log_prompt("[Config] warmup:", args.warmup, args.log_prompt)
    print_with_log_prompt("[Config] trials:", args.trials, args.log_prompt)
    print_with_log_prompt(
        "[Config] compile_framework_version:",
        compile_framework_version,
        args.log_prompt,
    )


def print_running_status(args, eager_success, compiled_success):
    def convert_to_str(b):
        return "success" if b else "failed"

    print_with_log_prompt(
        "[Result][status]",
        f"eager:{convert_to_str(eager_success)} compiled:{convert_to_str(compiled_success)}",
        args.log_prompt,
    )


def print_times_and_speedup(args, eager_stats, compiled_stats):
    print_with_log_prompt(
        "[Performance][eager]:", json.dumps(eager_stats), args.log_prompt
    )
    print_with_log_prompt(
        "[Performance][compiled]:", json.dumps(compiled_stats), args.log_prompt
    )

    e2e_speedup = 0
    gpu_speedup = 0

    eager_e2e_time_ms = eager_stats.get("e2e", {}).get("mean", 0)
    compiled_e2e_time_ms = compiled_stats.get("e2e", {}).get("mean", 0)

    if eager_e2e_time_ms > 0 and compiled_e2e_time_ms > 0:
        e2e_speedup = eager_e2e_time_ms / compiled_e2e_time_ms

    if "cuda" in args.device:
        eager_gpu_time_ms = eager_stats.get("gpu", {}).get("mean", 0)
        compiled_gpu_time_ms = compiled_stats.get("gpu", {}).get("mean", 0)

        if eager_gpu_time_ms > 0 and compiled_gpu_time_ms > 0:
            gpu_speedup = eager_gpu_time_ms / compiled_gpu_time_ms

    if e2e_speedup > 0:
        print_with_log_prompt("[Speedup][e2e]:", f"{e2e_speedup:.5f}", args.log_prompt)

    if "cuda" in args.device and gpu_speedup > 0:
        print_with_log_prompt("[Speedup][gpu]:", f"{gpu_speedup:.5f}", args.log_prompt)


def check_type_match(eager_dtypes, compiled_dtypes):
    if len(eager_dtypes) != len(compiled_dtypes):
        type_match = False
    else:
        type_match = all(
            eager == compiled for eager, compiled in zip(eager_dtypes, compiled_dtypes)
        )
    return type_match


def check_output_datatype(args, eager_dtypes, compiled_dtypes):
    print_with_log_prompt("[Datatype][eager]:", " ".join(eager_dtypes), args.log_prompt)
    print_with_log_prompt(
        "[Datatype][compiled]:", " ".join(compiled_dtypes), args.log_prompt
    )

    # datatype check
    type_match = check_type_match(eager_dtypes, compiled_dtypes)
    print_with_log_prompt(
        "[DataType]",
        f"eager:{eager_dtypes} compiled:{compiled_dtypes} match:{type_match}",
        args.log_prompt,
    )
    return type_match


def print_and_store_cmp(key, cmp_func, args, expected_out, compiled_out, **kwargs):
    cmp_ret = cmp_func(expected_out, compiled_out, **kwargs)
    print_with_log_prompt(f"[Correctness]{key}:", cmp_ret, args.log_prompt)
    return cmp_ret


def check_equal(args, expected_out, compiled_out, cmp_equal_func):
    print_and_store_cmp(
        key="[equal]",
        cmp_func=cmp_equal_func,
        args=args,
        expected_out=expected_out,
        compiled_out=compiled_out,
    )


def tolerance_generator(t):
    # for float16
    yield 10 ** (t * 3 / 5), 10**t
    # for bfloat16
    yield 10 ** (t * 1.796 / 5), 10**t
    # yield float32
    yield 10 ** (t * 5.886 / 5), 10**t
    # yield float64
    yield 10 ** (t * 7 / 5), 10 ** (t * 7 / 5)


def calculate_tolerance_pair(begin, end):
    tolerance_pair_list = []
    for t in range(begin, end + 1):
        for rtol, atol in tolerance_generator(t):
            effective_atol = float(f"{atol:.3g}")
            effective_rtol = float(f"{rtol:.3g}")
            tolerance_pair_list.append(
                {
                    "atol": effective_atol,
                    "rtol": effective_rtol,
                }
            )
    return tolerance_pair_list


def generate_allclose_configs(cmp_all_close_func):
    tolerance_pair_list = calculate_tolerance_pair(-10, 5)

    cmp_configs = []
    for pair in tolerance_pair_list:
        atol, rtol = pair["atol"], pair["rtol"]
        cmp_configs.append(
            (f"[all_close_atol_{atol:.2E}_rtol_{rtol:.2E}]", cmp_all_close_func, pair)
        )
    return cmp_configs


def check_allclose(
    args,
    expected_out,
    compiled_out,
    cmp_all_close_func,
    cmp_max_diff_func,
    cmp_mean_diff_func,
    cmp_max_relative_diff_func,
    cmp_mean_relative_diff_func,
):
    cmp_configs = generate_allclose_configs(cmp_all_close_func)
    cmp_configs.append(("[max_diff]", cmp_max_diff_func, {}))
    cmp_configs.append(("[mean_diff]", cmp_mean_diff_func, {}))
    cmp_configs.append(("[max_relative_diff]", cmp_max_relative_diff_func, {}))
    cmp_configs.append(("[mean_relative_diff]", cmp_mean_relative_diff_func, {}))

    for key, func, kwargs in cmp_configs:
        print_and_store_cmp(
            key=key,
            cmp_func=func,
            args=args,
            expected_out=expected_out,
            compiled_out=compiled_out,
            **kwargs,
        )
