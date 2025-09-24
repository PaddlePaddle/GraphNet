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


def check_correctness(
    args,
    expected_out,
    compiled_out,
    cmp_equal_func,
    cmp_all_close_func,
    cmp_max_diff_func,
    cmp_mean_diff_func,
    cmp_diff_count_func,
):
    cmp_configs = [
        ("[equal]", cmp_equal_func, {}),
        ("[all_close_atol_1.00E-06_rtol_1.00E-10]", cmp_all_close_func, {"atol": 1.00E-06, "rtol": 1.00E-10}),
        ("[all_close_atol_2.56E-04_rtol_1.00E-10]", cmp_all_close_func, {"atol": 2.56E-04, "rtol": 1.00E-10}),
        ("[all_close_atol_1.69E-12_rtol_1.00E-10]", cmp_all_close_func, {"atol": 1.69E-12, "rtol": 1.00E-10}),
        ("[all_close_atol_1.00E-14_rtol_1.00E-10]", cmp_all_close_func, {"atol": 1.00E-14, "rtol": 1.00E-10}),

        ("[all_close_atol_3.98E-06_rtol_1.00E-09]", cmp_all_close_func, {"atol": 3.98E-06, "rtol": 1.00E-09}),
        ("[all_close_atol_5.85E-04_rtol_1.00E-09]", cmp_all_close_func, {"atol": 5.85E-04, "rtol": 1.00E-09}),
        ("[all_close_atol_2.54E-11_rtol_1.00E-09]", cmp_all_close_func, {"atol": 2.54E-11, "rtol": 1.00E-09}),
        ("[all_close_atol_2.51E-13_rtol_1.00E-09]", cmp_all_close_func, {"atol": 2.51E-13, "rtol": 1.00E-09}),

        ("[all_close_atol_1.58E-05_rtol_1.00E-08]", cmp_all_close_func, {"atol": 1.58E-05, "rtol": 1.00E-08}),
        ("[all_close_atol_1.34E-03_rtol_1.00E-08]", cmp_all_close_func, {"atol": 1.34E-03, "rtol": 1.00E-08}),
        ("[all_close_atol_3.82E-10_rtol_1.00E-08]", cmp_all_close_func, {"atol": 3.82E-10, "rtol": 1.00E-08}),
        ("[all_close_atol_6.31E-12_rtol_1.00E-08]", cmp_all_close_func, {"atol": 6.31E-12, "rtol": 1.00E-08}),

        ("[all_close_atol_6.31E-05_rtol_1.00E-07]", cmp_all_close_func, {"atol": 6.31E-05, "rtol": 1.00E-07}),
        ("[all_close_atol_3.06E-03_rtol_1.00E-07]", cmp_all_close_func, {"atol": 3.06E-03, "rtol": 1.00E-07}),
        ("[all_close_atol_5.75E-09_rtol_1.00E-07]", cmp_all_close_func, {"atol": 5.75E-09, "rtol": 1.00E-07}),
        ("[all_close_atol_1.58E-10_rtol_1.00E-07]", cmp_all_close_func, {"atol": 1.58E-10, "rtol": 1.00E-07}),

        ("[all_close_atol_2.51E-04_rtol_1.00E-06]", cmp_all_close_func, {"atol": 2.51E-04, "rtol": 1.00E-06}),
        ("[all_close_atol_7.00E-03_rtol_1.00E-06]", cmp_all_close_func, {"atol": 7.00E-03, "rtol": 1.00E-06}),
        ("[all_close_atol_8.65E-08_rtol_1.00E-06]", cmp_all_close_func, {"atol": 8.65E-08, "rtol": 1.00E-06}),
        ("[all_close_atol_3.98E-09_rtol_1.00E-06]", cmp_all_close_func, {"atol": 3.98E-09, "rtol": 1.00E-06}),

        ("[all_close_atol_1.00E-03_rtol_1.00E-05]", cmp_all_close_func, {"atol": 1.00E-03, "rtol": 1.00E-05}),
        ("[all_close_atol_1.60E-02_rtol_1.00E-05]", cmp_all_close_func, {"atol": 1.60E-02, "rtol": 1.00E-05}),
        ("[all_close_atol_1.30E-06_rtol_1.00E-05]", cmp_all_close_func, {"atol": 1.30E-06, "rtol": 1.00E-05}),
        ("[all_close_atol_1.00E-07_rtol_1.00E-05]", cmp_all_close_func, {"atol": 1.00E-07, "rtol": 1.00E-05}),

        ("[all_close_atol_3.98E-03_rtol_1.00E-04]", cmp_all_close_func, {"atol": 3.98E-03, "rtol": 1.00E-04}),
        ("[all_close_atol_3.66E-02_rtol_1.00E-04]", cmp_all_close_func, {"atol": 3.66E-02, "rtol": 1.00E-04}),
        ("[all_close_atol_1.96E-05_rtol_1.00E-04]", cmp_all_close_func, {"atol": 1.96E-05, "rtol": 1.00E-04}),
        ("[all_close_atol_2.51E-06_rtol_1.00E-04]", cmp_all_close_func, {"atol": 2.51E-06, "rtol": 1.00E-04}),

        ("[all_close_atol_1.58E-02_rtol_1.00E-03]", cmp_all_close_func, {"atol": 1.58E-02, "rtol": 1.00E-03}),
        ("[all_close_atol_8.36E-02_rtol_1.00E-03]", cmp_all_close_func, {"atol": 8.36E-02, "rtol": 1.00E-03}),
        ("[all_close_atol_2.94E-04_rtol_1.00E-03]", cmp_all_close_func, {"atol": 2.94E-04, "rtol": 1.00E-03}),
        ("[all_close_atol_6.31E-05_rtol_1.00E-03]", cmp_all_close_func, {"atol": 6.31E-05, "rtol": 1.00E-03}),

        ("[all_close_atol_6.31E-02_rtol_1.00E-02]", cmp_all_close_func, {"atol": 6.31E-02, "rtol": 1.00E-02}),
        ("[all_close_atol_1.91E-01_rtol_1.00E-02]", cmp_all_close_func, {"atol": 1.91E-01, "rtol": 1.00E-02}),
        ("[all_close_atol_4.42E-03_rtol_1.00E-02]", cmp_all_close_func, {"atol": 4.42E-03, "rtol": 1.00E-02}),
        ("[all_close_atol_1.58E-03_rtol_1.00E-02]", cmp_all_close_func, {"atol": 1.58E-03, "rtol": 1.00E-02}),

        ("[all_close_atol_2.51E-01_rtol_1.00E-01]", cmp_all_close_func, {"atol": 2.51E-01, "rtol": 1.00E-01}),
        ("[all_close_atol_4.37E-01_rtol_1.00E-01]", cmp_all_close_func, {"atol": 4.37E-01, "rtol": 1.00E-01}),
        ("[all_close_atol_6.65E-02_rtol_1.00E-01]", cmp_all_close_func, {"atol": 6.65E-02, "rtol": 1.00E-01}),
        ("[all_close_atol_3.98E-02_rtol_1.00E-01]", cmp_all_close_func, {"atol": 3.98E-02, "rtol": 1.00E-01}),

        ("[all_close_atol_1.00E+00_rtol_1.00E+00]", cmp_all_close_func, {"atol": 1.00E+00, "rtol": 1.00E+00}),
        ("[all_close_atol_1.00E+00_rtol_1.00E+00]", cmp_all_close_func, {"atol": 1.00E+00, "rtol": 1.00E+00}),
        ("[all_close_atol_1.00E+00_rtol_1.00E+00]", cmp_all_close_func, {"atol": 1.00E+00, "rtol": 1.00E+00}),
        ("[all_close_atol_1.00E+00_rtol_1.00E+00]", cmp_all_close_func, {"atol": 1.00E+00, "rtol": 1.00E+00}),

        ("[all_close_atol_3.98E+00_rtol_1.00E+01]", cmp_all_close_func, {"atol": 3.98E+00, "rtol": 1.00E+01}),
        ("[all_close_atol_2.29E+00_rtol_1.00E+01]", cmp_all_close_func, {"atol": 2.29E+00, "rtol": 1.00E+01}),
        ("[all_close_atol_1.50E+01_rtol_1.00E+01]", cmp_all_close_func, {"atol": 1.50E+01, "rtol": 1.00E+01}),
        ("[all_close_atol_2.51E+01_rtol_1.00E+01]", cmp_all_close_func, {"atol": 2.51E+01, "rtol": 1.00E+01}),

        ("[all_close_atol_1.58E+01_rtol_1.00E+02]", cmp_all_close_func, {"atol": 1.58E+01, "rtol": 1.00E+02}),
        ("[all_close_atol_5.23E+00_rtol_1.00E+02]", cmp_all_close_func, {"atol": 5.23E+00, "rtol": 1.00E+02}),
        ("[all_close_atol_2.26E+02_rtol_1.00E+02]", cmp_all_close_func, {"atol": 2.26E+02, "rtol": 1.00E+02}),
        ("[all_close_atol_6.31E+02_rtol_1.00E+02]", cmp_all_close_func, {"atol": 6.31E+02, "rtol": 1.00E+02}),

        ("[all_close_atol_6.31E+01_rtol_1.00E+03]", cmp_all_close_func, {"atol": 6.31E+01, "rtol": 1.00E+03}),
        ("[all_close_atol_1.20E+01_rtol_1.00E+03]", cmp_all_close_func, {"atol": 1.20E+01, "rtol": 1.00E+03}),
        ("[all_close_atol_3.40E+03_rtol_1.00E+03]", cmp_all_close_func, {"atol": 3.40E+03, "rtol": 1.00E+03}),
        ("[all_close_atol_1.58E+04_rtol_1.00E+03]", cmp_all_close_func, {"atol": 1.58E+04, "rtol": 1.00E+03}),

        ("[all_close_atol_2.51E+02_rtol_1.00E+04]", cmp_all_close_func, {"atol": 2.51E+02, "rtol": 1.00E+04}),
        ("[all_close_atol_2.73E+01_rtol_1.00E+04]", cmp_all_close_func, {"atol": 2.73E+01, "rtol": 1.00E+04}),
        ("[all_close_atol_5.11E+04_rtol_1.00E+04]", cmp_all_close_func, {"atol": 5.11E+04, "rtol": 1.00E+04}),
        ("[all_close_atol_3.98E+05_rtol_1.00E+04]", cmp_all_close_func, {"atol": 3.98E+05, "rtol": 1.00E+04}),

        ("[all_close_atol_1.00E+03_rtol_1.00E+05]", cmp_all_close_func, {"atol": 1.00E+03, "rtol": 1.00E+05}),
        ("[all_close_atol_6.25E+01_rtol_1.00E+05]", cmp_all_close_func, {"atol": 6.25E+01, "rtol": 1.00E+05}),
        ("[all_close_atol_7.69E+05_rtol_1.00E+05]", cmp_all_close_func, {"atol": 7.69E+05, "rtol": 1.00E+05}),
        ("[all_close_atol_1.00E+07_rtol_1.00E+05]", cmp_all_close_func, {"atol": 1.00E+07, "rtol": 1.00E+05}),
        ("[max_diff]", cmp_max_diff_func, {}),
        ("[mean_diff]", cmp_mean_diff_func, {}),
    ]

    for key, func, kwargs in cmp_configs:
        print_and_store_cmp(
            key=key,
            cmp_func=func,
            args=args,
            expected_out=expected_out,
            compiled_out=compiled_out,
            **kwargs,
        )
