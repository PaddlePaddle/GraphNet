#!/usr/bin/env python3
"""
run_trialrun_pipeline.py — 纯 trial-run 维度泛化

与 run_pipeline.py 不同，本脚本不走 FX 图重写 / reifier 匹配，
直接用 dim_probe 已验证成功的维度值生成模型变体。

工作流程:
    1. 读取 dim_probe 输出的 probe_result.json（每个 Symbol 的成功值列表）
    2. 读取 input_tensor_constraints.py（符号维度定义）
    3. 为每组维度值生成变体：拷贝模型目录 + 替换 metadata 文件 + 替换 model.py 硬编码
    4. 可选验证每个变体的 forward pass

用法:
    python -m graph_net.agent.dimension_generalizer.run_trialrun_pipeline \
        --probe-output /path/to/probe_output \
        --model-list /path/to/model_list.txt \
        --output-dir /path/to/final_output \
        [--verbose] [--resume]
"""

import os
import sys
import json
import copy
import shutil
import logging
import argparse
import traceback
import subprocess
from pathlib import Path
from collections import OrderedDict
from itertools import product

import sympy

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
)

# 确保 GraphNet 根目录在 sys.path 中
_GRAPHNET_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _GRAPHNET_ROOT not in sys.path:
    sys.path.insert(0, _GRAPHNET_ROOT)

from graph_net.dynamic_dim_constraints import DynamicDimConstraints  # noqa: E402
from graph_net.tensor_meta import TensorMeta  # noqa: E402


# ─────────────────────────────────────────────
# 变体值生成策略
# ─────────────────────────────────────────────

# 每个模型最多生成的变体数
MAX_VARIANTS = 10

# 经典的 (batch, seq_len) 组合 — 工业界最常用的配置
COMMON_BATCH_SEQ_COMBOS = [
    (1, 128),  # 在线推理：单条短文本
    (1, 256),  # 在线推理：单条中等文本
    (1, 512),  # 在线推理：单条长文本
    (1, 1024),  # 在线推理：单条超长文本
    (4, 128),  # 小批量推理
    (8, 128),  # 中批量推理
    (16, 64),  # 大批量短文本
    (32, 64),  # 大批量短文本
    (4, 256),  # 小批量中等文本
    (8, 64),  # 中批量短文本
]

# 经典单维度值（batch_size 或 seq_len 或空间分辨率）
COMMON_SINGLE_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def generate_single_dim_variants(dynamic_dims, sym_names, symbols):
    """单维度变化策略：每次只改一个维度，其余保持原值。
    适用于多维度组合有交叉约束的模型。
    """
    orig_values = {name: dynamic_dims[name]["original_value"] for name in sym_names}
    variants = []
    seen = set()

    for name, sym in zip(sym_names, symbols):
        info = dynamic_dims[name]
        ok_vals = set(info["successful_values"])
        ok_vals.add(info["original_value"])
        # 优先用常见值
        common = sorted(v for v in ok_vals if v in COMMON_SINGLE_VALUES)
        others = sorted(v for v in ok_vals if v not in COMMON_SINGLE_VALUES)
        for val in common + others:
            if val == orig_values[name]:
                continue  # 跳过原始值（和原始组合一样）
            od = OrderedDict()
            for n2, s2 in zip(sym_names, symbols):
                od[s2] = val if n2 == name else orig_values[n2]
            key = tuple(od.values())
            if key not in seen:
                seen.add(key)
                variants.append(od)
            if len(variants) >= MAX_VARIANTS:
                return variants

    return variants


def generate_variant_values(probe_detail, verbose=False):
    """根据 probe_result.json 生成变体维度组合列表。

    优先使用 verified_combos（dim_probe 联合探测验证通过的组合），
    如果没有则 fallback 到笛卡尔积。

    Args:
        probe_detail: probe_result.json 内容

    Returns:
        list of OrderedDict: 每个 dict 是 {sympy.Symbol: int_value} 的映射
    """
    dynamic_dims = probe_detail.get("dynamic_dims", {})
    if not dynamic_dims:
        return []

    sym_names = sorted(dynamic_dims.keys(), key=lambda s: int(s[1:]))
    symbols = [sympy.Symbol(name) for name in sym_names]

    # 优先使用 verified_combos
    verified_combos = probe_detail.get("verified_combos", [])
    if verified_combos:
        variants = []
        for combo_dict in verified_combos:
            od = OrderedDict()
            for name, sym in zip(sym_names, symbols):
                od[sym] = combo_dict.get(name, dynamic_dims[name]["original_value"])
            variants.append(od)
        variants = variants[:MAX_VARIANTS]
        if verbose:
            print(f"  [variants] 使用 verified_combos: {len(variants)} 个")
        return variants

    # Fallback: 笛卡尔积
    per_sym_vals = []
    for name in sym_names:
        info = dynamic_dims[name]
        ok = set(info["successful_values"])
        orig = info["original_value"]
        ok.add(orig)
        common = sorted(v for v in ok if v in COMMON_SINGLE_VALUES)
        others = sorted(v for v in ok if v not in COMMON_SINGLE_VALUES)
        per_sym_vals.append(common + others)

    variants = []
    for combo in product(*per_sym_vals):
        variants.append(OrderedDict(zip(symbols, combo)))
        if len(variants) >= MAX_VARIANTS:
            break

    orig_combo = OrderedDict(
        [
            (sympy.Symbol(name), dynamic_dims[name]["original_value"])
            for name in sym_names
        ]
    )
    if orig_combo not in variants:
        variants.insert(0, orig_combo)
        variants = variants[:MAX_VARIANTS]

    if verbose:
        print(f"  [variants] 生成 {len(variants)} 个维度组合（笛卡尔积）")
    return variants


# ─────────────────────────────────────────────
# 变体生成
# ─────────────────────────────────────────────


def _rewrite_model_py(model_py_code, orig_values, new_values):
    """替换 model.py 中与动态维度关联的硬编码常量。

    只替换安全的模式：
    - torch.arange(N, ...) / torch.zeros(N, ...) / torch.ones(N, ...)
    - .expand(..., N, ...) / .reshape(..., N, ...) / .view(..., N, ...)
    - 函数调用中的独立数字参数
    - 特殊处理：batch 维度（S0）即使原值为 1 也要替换 reshape/view/expand 的第一个参数

    Args:
        model_py_code: model.py 源码
        orig_values: {symbol_name: original_int_value}  e.g. {"S0": 1, "S1": 40960}
        new_values:  {symbol_name: new_int_value}        e.g. {"S0": 4, "S1": 128}
    Returns:
        修改后的源码
    """
    import re

    # 只替换 orig > 1 的值（值为 1 的太常见，替换容易误伤）
    # 但 S0 (batch 维度) 即使是 1 也要替换
    replacements = {}
    batch_original = orig_values.get("S0")
    batch_new = new_values.get("S0")

    for sym in orig_values:
        orig = orig_values[sym]
        new = new_values[sym]
        if orig != new and (orig > 1 or sym == "S0"):
            replacements[orig] = new

    if not replacements:
        return model_py_code

    code = model_py_code
    for orig_val, new_val in replacements.items():
        orig_s = str(orig_val)
        new_s = str(new_val)

        # 模式1: torch.arange(N  /  torch.zeros(N  /  torch.ones(N  /  torch.full(N
        code = re.sub(
            r"(torch\.(?:arange|zeros|ones|full|empty)\()" + orig_s + r"(\b)",
            r"\g<1>" + new_s + r"\2",
            code,
        )

        # 特殊处理：batch 维度（S0）替换 reshape/view/expand 的第一个参数
        if orig_val == batch_original and new_val == batch_new:
            # 替换 .reshape(1, ...) / .view(1, ...) / .expand(1, ...) 中的第一个 1
            # 只替换 reshape/view/expand 后面跟着的 1，而且是第一个参数
            pattern = (
                r"(\.(?:reshape|view|expand|repeat)\s*\(\s*)" + orig_s + r"(?=[,\)])"
            )
            code = re.sub(pattern, r"\g<1>" + new_s, code)

        # 对于非 batch 维度（orig > 1），使用原来的模式替换
        if orig_val > 1:
            # 模式2: .expand(... N ...) / .reshape(... N ...) / .view(... N ...)
            code = re.sub(
                r"(\.(expand|reshape|view|repeat|narrow)\([^)]*?)(?<=[,(  ])"
                + orig_s
                + r"(?=[,) ])",
                r"\g<1>" + new_s,
                code,
            )
            # 模式3: 独立数字参数 — 如 slice(None, N, None)、new_ones(N, ...)
            code = re.sub(r"\(" + orig_s + r",", "(" + new_s + ",", code)
            code = re.sub(r", " + orig_s + r"\)", ", " + new_s + ")", code)
            code = re.sub(r", " + orig_s + r",", ", " + new_s + ",", code)

    return code


def generate_variants(
    model_path,
    dyn_dim_cstrs,
    variant_values_list,
    output_dir,
    rel_model_path,
    verbose=False,
    src_model_path=None,
):
    """为每组维度值生成一个模型变体目录。

    拷贝模型目录，更新 input_tensor_constraints.py、weight_meta.py，
    并替换 model.py 中与动态维度关联的硬编码常量。

    Args:
        model_path: probe 输出目录（含 input_tensor_constraints.py 等）
        src_model_path: 源模型目录（含 model.py），如果为 None 则使用 model_path
    """
    from graph_net.constraint_util import update_tensor_metas_by_dyn_dim_cstr

    # 确定拷贝源：优先用 src_model_path（有 model.py），否则用 model_path
    copy_src = (
        src_model_path
        if src_model_path and os.path.isdir(src_model_path)
        else model_path
    )

    # 从 copy_src 加载 tensor_metas（probe dir 可能没有 weight_meta.py）
    meta_src = copy_src
    tensor_metas = []
    input_meta_path = os.path.join(meta_src, "input_meta.py")
    if os.path.exists(input_meta_path):
        tensor_metas.extend(TensorMeta.unserialize_from_py_file(input_meta_path))
    weight_meta_path = os.path.join(meta_src, "weight_meta.py")
    if os.path.exists(weight_meta_path):
        tensor_metas.extend(TensorMeta.unserialize_from_py_file(weight_meta_path))

    generated_count = 0
    for idx, symbol2value in enumerate(variant_values_list):
        cur_dyn_dim_cstrs = copy.deepcopy(dyn_dim_cstrs)
        cur_tensor_metas = copy.deepcopy(tensor_metas)

        # 检查约束
        if not cur_dyn_dim_cstrs.check_delta_symbol2example_value(symbol2value):
            if verbose:
                vals_str = ", ".join(f"{s}={v}" for s, v in symbol2value.items())
                print(f"  [skip] variant {idx} ({vals_str}) 不满足约束")
            continue

        cur_dyn_dim_cstrs.update_symbol2example_value(symbol2value)
        update_tensor_metas_by_dyn_dim_cstr(cur_tensor_metas, cur_dyn_dim_cstrs)

        # 输出路径: output_dir/<index>/<rel_model_path>
        to_model_path = Path(output_dir) / str(idx) / rel_model_path
        to_model_path.mkdir(parents=True, exist_ok=True)

        # 拷贝原始模型目录
        shutil.copytree(Path(copy_src), to_model_path, dirs_exist_ok=True)

        # 替换 model.py 中的硬编码常量
        model_py_path = to_model_path / "model.py"
        if model_py_path.exists():
            orig_code = model_py_path.read_text()
            orig_vals = {
                str(s): dyn_dim_cstrs.symbol2example_value[s]
                for s in dyn_dim_cstrs.symbols
            }
            new_vals = {str(s): v for s, v in symbol2value.items()}
            new_code = _rewrite_model_py(orig_code, orig_vals, new_vals)
            if new_code != orig_code:
                model_py_path.write_text(new_code)

        # 写入更新后的 input_tensor_constraints.py
        cstr_code = cur_dyn_dim_cstrs.serialize_to_py_str()
        (to_model_path / "input_tensor_constraints.py").write_text(cstr_code)

        # 写入更新后的 weight_meta.py
        weight_meta_code = "\n".join(
            tm.serialize_to_py_str() for tm in cur_tensor_metas
        )
        (to_model_path / "weight_meta.py").write_text(weight_meta_code)

        generated_count += 1
        if verbose:
            vals_str = ", ".join(f"{s}={v}" for s, v in symbol2value.items())
            print(f"  [variant {idx}] {vals_str} → {to_model_path}")

    return generated_count


_VERIFY_SCRIPT = """\
import sys, importlib.util, torch
sys.path.insert(0, "{graphnet_root}")
from graph_net.torch import utils

model_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("m", model_path + "/model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.GraphModule()

inputs_params = utils.load_converted_from_text(model_path)
state_dict = {{k: utils.get_dummy_tensor(v) for k, v in inputs_params["weight_info"].items()}}

with torch.no_grad():
    model(**state_dict)
"""


def verify_variant(variant_path, verbose=False):
    """用子进程验证变体能否正确执行 forward，避免 CUDA assert 级联。"""
    model_py = os.path.join(variant_path, "model.py")
    if not os.path.exists(model_py):
        return False, "model.py not found"

    script = _VERIFY_SCRIPT.format(graphnet_root=_GRAPHNET_ROOT)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(variant_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return True, None
        else:
            err = result.stderr.strip()[-500:] if result.stderr else "unknown error"
            return False, err
    except subprocess.TimeoutExpired:
        return False, "timeout (120s)"
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# 单模型处理
# ─────────────────────────────────────────────


def process_single_model(
    probe_model_path,
    rel_model_path,
    output_dir,
    verbose=False,
    src_model_path=None,
    skip_verify=False,
):
    """处理单个模型：读取 probe 结果 → 直接生成变体"""
    print(f"\n{'='*60}")
    print(f"处理: {rel_model_path}")
    print(f"{'='*60}")

    # 检查 probe_result.json
    probe_json_path = os.path.join(probe_model_path, "probe_result.json")
    cstr_path = os.path.join(probe_model_path, "input_tensor_constraints.py")

    if not os.path.exists(cstr_path):
        print("  [跳过] 无 input_tensor_constraints.py（无动态维度）")
        return {"status": "skip", "reason": "no constraints"}

    # 读取约束
    dyn_dim_cstrs = DynamicDimConstraints.unserialize_from_py_file(cstr_path)
    if not dyn_dim_cstrs.symbols:
        print("  [跳过] 无符号维度")
        return {"status": "skip", "reason": "no symbols"}

    sym_shapes_str = dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str()
    print(f"  符号维度: {dyn_dim_cstrs.symbols}")
    print(f"  示例值: {dyn_dim_cstrs.symbol2example_value}")
    print(f"  符号 shapes: {sym_shapes_str}")

    # 读取 probe_result.json（如果有的话）
    if os.path.exists(probe_json_path):
        with open(probe_json_path) as f:
            probe_detail = json.load(f)
        print(
            f"  probe_result.json: {json.dumps(probe_detail['dynamic_dims'], indent=4)}"
        )
    else:
        # 没有 probe_result.json，用约束中的信息构造默认值
        print("  [注意] 无 probe_result.json，使用默认值")
        probe_detail = {"dynamic_dims": {}}
        for sym in dyn_dim_cstrs.symbols:
            name = str(sym)
            orig_val = dyn_dim_cstrs.symbol2example_value[sym]
            probe_detail["dynamic_dims"][name] = {
                "original_value": orig_val,
                "successful_values": COMMON_SINGLE_VALUES,  # 假设常见值都能用
                "failed_values": [],
            }

    # 生成变体值组合
    variant_values = generate_variant_values(probe_detail, verbose)
    if not variant_values:
        print("  [跳过] 无法生成变体值组合")
        return {"status": "skip", "reason": "no variant values"}

    print(f"  将生成 {len(variant_values)} 个变体")

    # 生成变体
    count = generate_variants(
        probe_model_path,
        dyn_dim_cstrs,
        variant_values,
        output_dir,
        rel_model_path,
        verbose,
        src_model_path=src_model_path,
    )

    if skip_verify:
        # 跳过验证，直接返回生成的变体数
        print(f"  生成 {count} 个变体（跳过验证）")
        return {
            "status": "ok",
            "variants": count,
            "total_combos": len(variant_values),
            "failed_variants": [],
        }

    # 验证每个变体（并行子进程）
    from concurrent.futures import ThreadPoolExecutor, as_completed

    verified = 0
    failed_variants = []
    variant_paths = []
    for idx in range(len(variant_values)):
        variant_path = os.path.join(output_dir, str(idx), rel_model_path)
        if os.path.isdir(variant_path):
            variant_paths.append((idx, variant_path))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(verify_variant, vp, verbose): (idx, vp)
            for idx, vp in variant_paths
        }
        for future in as_completed(futures):
            idx, vp = futures[future]
            ok, err = future.result()
            if ok:
                verified += 1
                if verbose:
                    print(f"  [verify OK] variant {idx}")
            else:
                failed_variants.append(idx)
                shutil.rmtree(vp)
                if verbose:
                    print(f"  [verify FAIL] variant {idx}: {err}")

    # 如果验证全失败，fallback 到单维度变化策略
    if verified == 0 and count > 0:
        print(f"  [fallback] 笛卡尔积 {count} 个变体全部验证失败，尝试单维度策略...")
        probe_json_path = os.path.join(probe_model_path, "probe_result.json")
        if os.path.exists(probe_json_path):
            with open(probe_json_path) as f:
                probe_detail = json.load(f)
            dynamic_dims = probe_detail.get("dynamic_dims", {})
            sym_names = sorted(dynamic_dims.keys(), key=lambda s: int(s[1:]))
            symbols = [sympy.Symbol(name) for name in sym_names]
            single_variants = generate_single_dim_variants(
                dynamic_dims, sym_names, symbols
            )
            if single_variants:
                single_count = generate_variants(
                    probe_model_path,
                    dyn_dim_cstrs,
                    single_variants,
                    output_dir,
                    rel_model_path,
                    verbose,
                    src_model_path=src_model_path,
                )
                fb_paths = []
                for idx in range(len(single_variants)):
                    variant_path = os.path.join(output_dir, str(idx), rel_model_path)
                    if os.path.isdir(variant_path):
                        fb_paths.append((idx, variant_path))
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(verify_variant, vp, verbose): (idx, vp)
                        for idx, vp in fb_paths
                    }
                    for future in as_completed(futures):
                        idx, vp = futures[future]
                        ok, err = future.result()
                        if ok:
                            verified += 1
                            if verbose:
                                print(f"  [verify OK] single-dim variant {idx}")
                        else:
                            shutil.rmtree(vp)
                            if verbose:
                                print(
                                    f"  [verify FAIL] single-dim variant {idx}: {err}"
                                )
                print(f"  [fallback] 单维度策略: 生成 {single_count}，验证通过 {verified}")

    print(f"  生成变体，验证通过 {verified} 个")
    return {
        "status": "ok",
        "variants": verified,
        "total_combos": len(variant_values),
        "failed_variants": failed_variants,
    }


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="纯 trial-run 维度泛化（跳过 FX 重写）")
    parser.add_argument(
        "--probe-output", type=str, required=True, help="dim_probe 生成的输出目录"
    )
    parser.add_argument(
        "--base-dir", type=str, default=None, help="源模型目录（含 model.py），默认同 probe-output"
    )
    parser.add_argument("--model-list", type=str, required=True, help="模型列表文件")
    parser.add_argument("--output-dir", type=str, required=True, help="最终输出目录")
    parser.add_argument("--verbose", action="store_true", default=False, help="详细输出")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已有输出的模型")
    parser.add_argument(
        "--retry-zero", action="store_true", help="只重新处理 0 变体的模型（使用单维度 fallback 策略）"
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="跳过变体验证（快速生成，后续单独验证）"
    )
    args = parser.parse_args()

    import time

    probe_output = Path(args.probe_output)
    base_dir = Path(args.base_dir) if args.base_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.model_list) as f:
        model_list = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    total = len(model_list)
    print(f"共 {total} 个模型待处理")
    print(f"probe_output: {probe_output}")
    print(f"output_dir: {output_dir}")

    results = {}
    ok_count = 0
    skip_count = 0
    error_count = 0
    total_variants = 0
    start_time = time.time()

    for i, rel_path in enumerate(model_list):
        probe_model_path = str(probe_output / rel_path)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"\n[Progress] {i+1}/{total} ({(i+1)/total*100:.1f}%) "
                f"ok={ok_count} skip={skip_count} fail={error_count} "
                f"variants={total_variants} "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
            )

        if not os.path.isdir(probe_model_path):
            skip_count += 1
            results[rel_path] = {"status": "skip", "reason": "dir not found"}
            continue

        if args.resume:
            variant_0_path = output_dir / "0" / rel_path
            if variant_0_path.exists() and (variant_0_path / "model.py").exists():
                skip_count += 1
                results[rel_path] = {"status": "skip", "reason": "already done"}
                continue

        if args.retry_zero:
            # 只处理有 input_tensor_constraints.py 但 variant_0 不存在的模型
            cstr_path = os.path.join(probe_model_path, "input_tensor_constraints.py")
            variant_0_path = output_dir / "0" / rel_path
            has_variant = (
                variant_0_path.exists() and (variant_0_path / "model.py").exists()
            )
            has_dynamic = os.path.exists(cstr_path)
            if has_variant or not has_dynamic:
                skip_count += 1
                results[rel_path] = {
                    "status": "skip",
                    "reason": "not zero-variant target",
                }
                continue

        try:
            src_model_path = str(base_dir / rel_path) if base_dir else None
            result = process_single_model(
                probe_model_path,
                rel_path,
                str(output_dir),
                args.verbose,
                src_model_path=src_model_path,
                skip_verify=getattr(args, "skip_verify", False),
            )
            results[rel_path] = result
            if result["status"] == "ok":
                ok_count += 1
                total_variants += result.get("variants", 0)
            else:
                skip_count += 1
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — 保存已有结果...")
            break
        except Exception as e:
            error_count += 1
            results[rel_path] = {"status": "error", "error": str(e)}
            if args.verbose:
                traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n\n{'='*60}")
    print(f"Trial-run Pipeline 完成: {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"总计: {len(results)} 个模型")
    print(f"  成功: {ok_count}")
    print(f"  跳过: {skip_count}")
    print(f"  失败: {error_count}")
    print(f"  共生成 {total_variants} 个变体")

    result_path = output_dir / "pipeline_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
