#!/usr/bin/env python3
"""
dim_probe_retry.py — 针对性优化重试

处理以下场景:
1. 超时模型 (status=timeout): 强制使用 FakeTensorMode 重试
2. OOM 模型 (status=error, CUDA OOM): 强制使用 FakeTensorMode 重试
3. 低变体模型: 用更密探测值补充 (1.5x, 3x, 6x 等)
4. 旧模型补 joint combo probing

用法:
    # 重试超时+OOM模型
    python -m graph_net.agent.dimension_generalizer.dim_probe_retry --mode retry-failed \
        --probe-output /work/full_probe_output \
        --base-dir /work/agent_extrator_model \
        --model-list /work/agent_extrator_model/full_model_list.txt

    # 补充低变体模型的探测值
    python -m graph_net.agent.dimension_generalizer.dim_probe_retry --mode enrich \
        --probe-output /work/full_probe_output \
        --base-dir /work/agent_extrator_model \
        --model-list /work/agent_extrator_model/full_model_list.txt

    # 补跑 joint combo probing
    python -m graph_net.agent.dimension_generalizer.dim_probe_retry --mode joint-combo \
        --probe-output /work/full_probe_output \
        --base-dir /work/agent_extrator_model \
        --model-list /work/agent_extrator_model/full_model_list.txt
"""

import os
import sys
import json
import time
import argparse
import subprocess
import resource
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
_GRAPHNET_ROOT = str(SCRIPT_DIR.parent.parent.parent)
if _GRAPHNET_ROOT not in sys.path:
    sys.path.insert(0, _GRAPHNET_ROOT)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 重试时用更长超时（FakeTensor 模式下一般很快，但加载模型可能慢）
RETRY_TIMEOUT = 180  # 3分钟
MEM_LIMIT_GB = 30
NUM_WORKERS = 4


def _set_num_workers(n):
    global NUM_WORKERS
    NUM_WORKERS = n


# 补充探测值：更密的倍数
ENRICH_MULTIPLIERS = [1.5, 3, 6, 0.25, 0.75]
ENRICH_ABSOLUTE = [3, 6, 12, 24, 48, 96, 192, 384, 768]


def log_msg(log_file, lock, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        with lock:
            with open(log_file, "a") as f:
                f.write(line + "\n")


# ============================================================
# Mode 1: retry-failed — 重试超时和 OOM 模型
# ============================================================


def find_failed_models(probe_output, model_list):
    """找出 status=timeout 或 status=error(OOM) 的模型"""
    failed = []
    for rel_path in model_list:
        done_path = os.path.join(probe_output, rel_path, "done.txt")
        if not os.path.exists(done_path):
            continue
        with open(done_path) as f:
            content = f.read()
        if "status=timeout" in content:
            failed.append((rel_path, "timeout"))
        elif "status=error" in content:
            # 检查是否 OOM
            if (
                "ALLOC" in content.upper()
                or "OOM" in content.upper()
                or "memory" in content.lower()
            ):
                failed.append((rel_path, "oom"))
            else:
                failed.append((rel_path, "error"))
    return failed


def retry_single_model(rel_path, base_dir, probe_output):
    """用 FakeTensorMode 重试单个模型"""
    model_dir = os.path.join(base_dir, rel_path)
    out_dir = os.path.join(probe_output, rel_path)
    worker_script = os.path.join(str(SCRIPT_DIR), "dim_probe.py")

    # 先删掉旧 done.txt 让 dim_probe 重新处理
    done_path = os.path.join(out_dir, "done.txt")
    if os.path.exists(done_path):
        os.remove(done_path)

    # 设置环境变量强制使用 FakeTensorMode
    env = os.environ.copy()
    env["DIM_PROBE_FORCE_FAKE"] = "1"

    cmd = [
        sys.executable,
        worker_script,
        "--model-dir",
        model_dir,
        "--output-dir",
        out_dir,
        "--quiet",
    ]

    def _limit_mem():
        mem_bytes = MEM_LIMIT_GB * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            timeout=RETRY_TIMEOUT,
            capture_output=True,
            text=True,
            preexec_fn=_limit_mem,
            env=env,
        )
        elapsed = time.time() - t0

        if proc.returncode == 0:
            # 检查结果
            n_dyn = 0
            probe_json = os.path.join(out_dir, "probe_result.json")
            if os.path.exists(probe_json):
                try:
                    with open(probe_json) as f:
                        pr = json.load(f)
                    n_dyn = len(pr.get("dynamic_dims", {}))
                except Exception:
                    pass
            return (rel_path, "ok", f"dynamic={n_dyn} ({elapsed:.0f}s)")
        else:
            err = (proc.stderr or "")[-200:]
            # 写 done.txt
            os.makedirs(out_dir, exist_ok=True)
            with open(done_path, "w") as f:
                f.write(f"status=error\nerror=retry_exit_{proc.returncode}\n")
            return (rel_path, "error", f"exit={proc.returncode} ({elapsed:.0f}s) {err}")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        os.makedirs(out_dir, exist_ok=True)
        with open(done_path, "w") as f:
            f.write("status=timeout\nerror=retry_timeout\n")
        return (rel_path, "timeout", f"still timeout ({elapsed:.0f}s)")
    except Exception as e:
        os.makedirs(out_dir, exist_ok=True)
        with open(done_path, "w") as f:
            f.write(f"status=error\nerror={e}\n")
        return (rel_path, "error", str(e))


def run_retry_failed(probe_output, base_dir, model_list, log_file, lock):
    """重试所有超时和 OOM 模型"""
    failed_models = find_failed_models(probe_output, model_list)
    timeout_models = [(r, t) for r, t in failed_models if t == "timeout"]
    oom_models = [(r, t) for r, t in failed_models if t == "oom"]
    other_errors = [(r, t) for r, t in failed_models if t == "error"]

    # 只重试 timeout 和 oom
    to_retry = timeout_models + oom_models
    log_msg(
        log_file,
        lock,
        f"retry-failed: timeout={len(timeout_models)} oom={len(oom_models)} "
        f"other_error={len(other_errors)} → 重试 {len(to_retry)} 个",
    )

    if not to_retry:
        log_msg(log_file, lock, "无需重试")
        return

    succeeded = 0
    still_failed = 0
    done_count = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for rel_path, fail_type in to_retry:
            fut = executor.submit(retry_single_model, rel_path, base_dir, probe_output)
            futures[fut] = (rel_path, fail_type)

        for fut in as_completed(futures):
            rel_path, status, detail = fut.result()
            done_count += 1

            if status == "ok":
                succeeded += 1
                log_msg(log_file, lock, f"[RETRY OK] {rel_path} {detail}")
            else:
                still_failed += 1
                log_msg(log_file, lock, f"[RETRY FAIL] {rel_path}: {detail}")

            if done_count % 20 == 0:
                log_msg(
                    log_file,
                    lock,
                    f"[Progress] {done_count}/{len(to_retry)} "
                    f"ok={succeeded} fail={still_failed}",
                )

    log_msg(
        log_file,
        lock,
        f"retry-failed 完成: 重试 {len(to_retry)}，成功 {succeeded}，仍失败 {still_failed}",
    )


# ============================================================
# Mode 2: enrich — 补充低变体模型的探测值
# ============================================================


def find_low_variant_models(probe_output, model_list, threshold=8):
    """找出有动态维度但成功值少于 threshold 的模型"""
    low_models = []
    for rel_path in model_list:
        probe_json = os.path.join(probe_output, rel_path, "probe_result.json")
        if not os.path.exists(probe_json):
            continue
        try:
            with open(probe_json) as f:
                data = json.load(f)
            dynamic_dims = data.get("dynamic_dims", {})
            if not dynamic_dims:
                continue
            # 检查任一维度的 successful_values 数量
            min_ok = min(len(d["successful_values"]) for d in dynamic_dims.values())
            if min_ok < threshold:
                low_models.append((rel_path, min_ok))
        except Exception:
            continue
    return low_models


def enrich_single_model(rel_path, base_dir, probe_output):
    """对单个模型补充更密的探测值"""
    model_dir = os.path.join(base_dir, rel_path)
    out_dir = os.path.join(probe_output, rel_path)
    probe_json = os.path.join(out_dir, "probe_result.json")

    if not os.path.exists(probe_json):
        return (rel_path, "skip", "no probe_result.json")

    try:
        with open(probe_json) as f:
            data = json.load(f)
    except Exception as e:
        return (rel_path, "error", str(e))

    dynamic_dims = data.get("dynamic_dims", {})
    if not dynamic_dims:
        return (rel_path, "skip", "no dynamic dims")

    # 加载模型
    model_py = os.path.join(model_dir, "model.py")
    weight_meta_py = os.path.join(model_dir, "weight_meta.py")
    if not os.path.exists(model_py) or not os.path.exists(weight_meta_py):
        return (rel_path, "skip", "missing files")

    # 在子进程中执行以隔离 CUDA 状态
    # 构造一个临时 Python 脚本内联执行
    enrich_script = f"""
import sys, json, os
sys.path.insert(0, '{_GRAPHNET_ROOT}')
sys.path.insert(0, '{str(SCRIPT_DIR)}')
from dim_probe import (load_weight_metas, classify_params, try_forward,
                        _resolve_dtype, build_tensor, ParamMeta,
                        save_probe_result_json)
import importlib.util, inspect, torch

model_dir = '{model_dir}'
out_dir = '{out_dir}'
probe_json = '{probe_json}'

with open(probe_json) as f:
    data = json.load(f)

dynamic_dims = data.get("dynamic_dims", {{}})

# 加载模型
metas = load_weight_metas(os.path.join(model_dir, "weight_meta.py"))
metas_dict = {{m.name: m for m in metas}}
symint_params, data_input_params, weight_params = classify_params(
    os.path.join(model_dir, "model.py"))

spec = importlib.util.spec_from_file_location("m", os.path.join(model_dir, "model.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sig = inspect.signature(getattr(mod, 'GraphModule').forward)
param_order = [p for p in sig.parameters if p != 'self']
model = getattr(mod, 'GraphModule')()
model.eval()

use_meta = len(weight_params) > 50
weight_cache = {{}}

added = 0
for sym_name, info in dynamic_dims.items():
    orig_val = info["original_value"]
    already_ok = set(info["successful_values"])
    already_fail = set(info.get("failed_values", []))

    # 生成补充探测值
    new_vals = set()
    for mult in {ENRICH_MULTIPLIERS}:
        v = int(orig_val * mult)
        if v > 0 and v not in already_ok and v not in already_fail:
            new_vals.add(v)
    for v in {ENRICH_ABSOLUTE}:
        if v > 0 and v not in already_ok and v not in already_fail:
            new_vals.add(v)

    for test_val in sorted(new_vals):
        if len(already_ok) >= 10:
            break
        # 构造 overrides
        shape_overrides = {{}}
        symint_overrides = {{}}
        for dp in data_input_params:
            dp_meta = metas_dict.get(dp)
            if dp_meta is None or not dp_meta.shape:
                continue
            new_shape = list(dp_meta.shape)
            changed = False
            for axis, dv in enumerate(new_shape):
                if dv == orig_val:
                    new_shape[axis] = test_val
                    changed = True
            if changed:
                shape_overrides[dp] = new_shape
        for sp in symint_params:
            sp_meta = metas_dict.get(sp)
            if sp_meta and sp_meta.data and sp_meta.data[0] == orig_val:
                symint_overrides[sp] = test_val

        ok, err = try_forward(model, metas_dict, param_order,
                              symint_params, data_input_params, weight_params,
                              shape_overrides=shape_overrides,
                              symint_overrides=symint_overrides,
                              weight_cache=weight_cache,
                              use_meta_device=use_meta)
        if ok:
            already_ok.add(test_val)
            added += 1
        else:
            already_fail.add(test_val)

    info["successful_values"] = sorted(already_ok)
    info["failed_values"] = sorted(already_fail)

with open(probe_json, 'w') as f:
    json.dump(data, f, indent=2)

print(f"added={{added}}")
"""

    def _limit_mem():
        mem_bytes = MEM_LIMIT_GB * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", enrich_script],
            timeout=RETRY_TIMEOUT,
            capture_output=True,
            text=True,
            preexec_fn=_limit_mem,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            added = 0
            for line in proc.stdout.strip().split("\n"):
                if line.startswith("added="):
                    added = int(line.split("=")[1])
            return (rel_path, "ok", f"+{added} values ({elapsed:.0f}s)")
        else:
            err = (proc.stderr or "")[-200:]
            return (rel_path, "error", f"exit={proc.returncode} ({elapsed:.0f}s) {err}")
    except subprocess.TimeoutExpired:
        return (rel_path, "timeout", "enrich timeout")
    except Exception as e:
        return (rel_path, "error", str(e))


def run_enrich(probe_output, base_dir, model_list, log_file, lock):
    """补充低变体模型的探测值"""
    low_models = find_low_variant_models(probe_output, model_list, threshold=8)
    log_msg(log_file, lock, f"enrich: 找到 {len(low_models)} 个低变体模型")

    if not low_models:
        log_msg(log_file, lock, "无需补充")
        return

    succeeded = 0
    total_added = 0
    done_count = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for rel_path, min_ok in low_models:
            fut = executor.submit(enrich_single_model, rel_path, base_dir, probe_output)
            futures[fut] = rel_path

        for fut in as_completed(futures):
            rel_path, status, detail = fut.result()
            done_count += 1

            if status == "ok":
                succeeded += 1
                # 解析 added 数量
                if "+" in detail:
                    try:
                        added = int(detail.split("+")[1].split(" ")[0])
                        total_added += added
                    except Exception:
                        pass
                log_msg(log_file, lock, f"[ENRICH OK] {rel_path} {detail}")
            else:
                log_msg(log_file, lock, f"[ENRICH FAIL] {rel_path}: {detail}")

            if done_count % 20 == 0:
                log_msg(
                    log_file,
                    lock,
                    f"[Progress] {done_count}/{len(low_models)} "
                    f"ok={succeeded} added={total_added}",
                )

    log_msg(
        log_file,
        lock,
        f"enrich 完成: {len(low_models)} 模型，成功 {succeeded}，新增 {total_added} 个值",
    )


# ============================================================
# Mode 3: joint-combo — 补跑联合组合探测
# ============================================================


def find_models_without_combos(probe_output, model_list):
    """找出有动态维度但没有 verified_combos 的模型"""
    models = []
    for rel_path in model_list:
        probe_json = os.path.join(probe_output, rel_path, "probe_result.json")
        if not os.path.exists(probe_json):
            continue
        try:
            with open(probe_json) as f:
                data = json.load(f)
            if not data.get("dynamic_dims"):
                continue
            if data.get("verified_combos"):
                continue  # 已有
            models.append(rel_path)
        except Exception:
            continue
    return models


def joint_combo_single_model(rel_path, base_dir, probe_output):
    """对单个模型补跑 joint combo probing"""
    model_dir = os.path.join(base_dir, rel_path)
    out_dir = os.path.join(probe_output, rel_path)

    model_py = os.path.join(model_dir, "model.py")
    weight_meta_py = os.path.join(model_dir, "weight_meta.py")
    if not os.path.exists(model_py) or not os.path.exists(weight_meta_py):
        return (rel_path, "skip", "missing files")

    combo_script = f"""
import sys, json, os
sys.path.insert(0, '{_GRAPHNET_ROOT}')
sys.path.insert(0, '{str(SCRIPT_DIR)}')
from dim_probe import probe_joint_combos

# probe_joint_combos 需要 model_dir 和 output_dir
# model_dir 用来加载 model.py + weight_meta.py
# output_dir 用来读写 probe_result.json
probe_joint_combos('{model_dir}', {{}}, '{out_dir}',
                   max_combos=10, verbose=False)

# 读回结果
probe_json = os.path.join('{out_dir}', 'probe_result.json')
with open(probe_json) as f:
    data = json.load(f)
n_combos = len(data.get('verified_combos', []))
print(f'combos={{n_combos}}')
"""

    def _limit_mem():
        mem_bytes = MEM_LIMIT_GB * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", combo_script],
            timeout=RETRY_TIMEOUT,
            capture_output=True,
            text=True,
            preexec_fn=_limit_mem,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            n_combos = 0
            for line in proc.stdout.strip().split("\n"):
                if line.startswith("combos="):
                    n_combos = int(line.split("=")[1])
            return (rel_path, "ok", f"combos={n_combos} ({elapsed:.0f}s)")
        else:
            err = (proc.stderr or "")[-200:]
            return (rel_path, "error", f"exit={proc.returncode} ({elapsed:.0f}s) {err}")
    except subprocess.TimeoutExpired:
        return (rel_path, "timeout", "combo timeout")
    except Exception as e:
        return (rel_path, "error", str(e))


def run_joint_combo(probe_output, base_dir, model_list, log_file, lock):
    """补跑 joint combo probing"""
    models = find_models_without_combos(probe_output, model_list)
    log_msg(log_file, lock, f"joint-combo: 找到 {len(models)} 个缺少 verified_combos 的模型")

    if not models:
        log_msg(log_file, lock, "无需补跑")
        return

    succeeded = 0
    total_combos = 0
    done_count = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for rel_path in models:
            fut = executor.submit(
                joint_combo_single_model, rel_path, base_dir, probe_output
            )
            futures[fut] = rel_path

        for fut in as_completed(futures):
            rel_path, status, detail = fut.result()
            done_count += 1

            if status == "ok":
                succeeded += 1
                if "combos=" in detail:
                    try:
                        n = int(detail.split("combos=")[1].split(" ")[0])
                        total_combos += n
                    except Exception:
                        pass
                log_msg(log_file, lock, f"[COMBO OK] {rel_path} {detail}")
            else:
                log_msg(log_file, lock, f"[COMBO FAIL] {rel_path}: {detail}")

            if done_count % 20 == 0:
                log_msg(
                    log_file,
                    lock,
                    f"[Progress] {done_count}/{len(models)} "
                    f"ok={succeeded} combos={total_combos}",
                )

    log_msg(
        log_file,
        lock,
        f"joint-combo 完成: {len(models)} 模型，成功 {succeeded}，总 combos={total_combos}",
    )


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="dim_probe 针对性优化重试")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["retry-failed", "enrich", "joint-combo", "all"],
        help="运行模式",
    )
    parser.add_argument(
        "--probe-output", type=str, required=True, help="dim_probe 输出目录"
    )
    parser.add_argument("--base-dir", type=str, required=True, help="源模型目录")
    parser.add_argument("--model-list", type=str, required=True, help="模型列表文件")
    parser.add_argument("--workers", type=int, default=4, help="并行 worker 数 (默认 4)")
    args = parser.parse_args()

    _set_num_workers(args.workers)

    with open(args.model_list) as f:
        model_list = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    log_file = os.path.join(args.probe_output, "retry_log.txt")
    lock = threading.Lock()

    log_msg(log_file, lock, f"=== dim_probe_retry mode={args.mode} ===")
    log_msg(log_file, lock, f"models={len(model_list)} workers={args.workers}")

    t0 = time.time()

    if args.mode in ("retry-failed", "all"):
        run_retry_failed(args.probe_output, args.base_dir, model_list, log_file, lock)

    if args.mode in ("enrich", "all"):
        run_enrich(args.probe_output, args.base_dir, model_list, log_file, lock)

    if args.mode in ("joint-combo", "all"):
        run_joint_combo(args.probe_output, args.base_dir, model_list, log_file, lock)

    elapsed = time.time() - t0
    log_msg(log_file, lock, f"=== 完成 mode={args.mode} elapsed={elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
