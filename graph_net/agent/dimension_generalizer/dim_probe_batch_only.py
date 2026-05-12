"""
Batch-only 维度探测工具

只探测 batch 维度（axis=0），跳过其他维度，避免因 seq_len/H/W 与权重 shape 耦合导致的失败。

用法：
    python -m graph_net.agent.dimension_generalizer.dim_probe_batch_only --model-dir /path/to/model
    python -m graph_net.agent.dimension_generalizer.dim_probe_batch_only --base-dir /path/to/models --model-list model_list.txt
"""

import argparse
import importlib.util
import inspect
import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict
import tempfile

# 确保 GraphNet 根目录在 sys.path 中
_GRAPHNET_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _GRAPHNET_ROOT not in sys.path:
    sys.path.insert(0, _GRAPHNET_ROOT)

# 导入 dim_probe 的核心函数
from graph_net.agent.dimension_generalizer.dim_probe import (  # noqa: E402
    load_weight_metas,
    classify_params,
    try_forward,
)
from graph_net.agent.dimension_generalizer.batch_hardcode_preprocess import (  # noqa: E402
    preprocess_model_py_for_batch,
)

# 探测参数
PROBE_MULTIPLIERS = [2, 4, 0.5]
PROBE_ABSOLUTE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MODEL_TIMEOUT = 120  # 秒


def probe_batch_only_single_model(model_dir: str, verbose: bool = True) -> Dict:
    """
    只探测 batch 维度（axis=0）

    Args:
        model_dir: 模型目录，包含 model.py 和 weight_meta.py
        verbose: 是否打印详细信息

    Returns:
        探测结果字典
    """
    model_py = os.path.join(model_dir, "model.py")
    weight_meta_py = os.path.join(model_dir, "weight_meta.py")

    if not os.path.exists(model_py) or not os.path.exists(weight_meta_py):
        if verbose:
            print(f"  [SKIP] {model_dir}: missing model.py or weight_meta.py")
        return {"model_dir": model_dir, "params": {}, "error": "missing files"}

    # 预处理：替换 batch 硬编码
    temp_model_py = None
    try:
        # 使用固定的临时目录，避免 /tmp 空间问题
        temp_dir = "/work/tmp_dim_probe"
        os.makedirs(temp_dir, exist_ok=True)
        temp_model_py = os.path.join(
            temp_dir, f"temp_model_{os.path.basename(model_dir)}.py"
        )

        _, replacements = preprocess_model_py_for_batch(
            model_py, temp_model_py, verbose=verbose
        )

        if replacements:
            model_py = temp_model_py
        else:
            # 没有替换，删除空临时文件
            if os.path.exists(temp_model_py):
                os.unlink(temp_model_py)
                temp_model_py = None

    except Exception as e:
        # 预处理失败，清理并继续
        if temp_model_py and os.path.exists(temp_model_py):
            os.unlink(temp_model_py)
        temp_model_py = None
        if verbose:
            print(f"  [WARN] Preprocess failed: {e}")

    # 加载元信息
    metas = load_weight_metas(weight_meta_py)
    metas_dict = {m.name: m for m in metas}

    # 分析 forward 签名
    symint_params, data_input_params, weight_params = classify_params(model_py)

    # forward 参数顺序
    spec = importlib.util.spec_from_file_location("model_ord", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    graph_module_cls = getattr(mod, "GraphModule")
    sig = inspect.signature(graph_module_cls.forward)
    param_order = [p for p in sig.parameters if p != "self"]

    # 加载模型
    model = graph_module_cls()
    model.eval()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Model: {model_dir}")
        print(f"  Data input params: {data_input_params}")
        print(f"{'='*60}")

    # 权重参数很多时，用 meta device 避免 OOM
    use_meta = len(weight_params) > 50
    if use_meta and verbose:
        print(
            f"  [INFO] Using meta device for shape-only probing ({len(weight_params)} weight params)"
        )

    # 先验证原始 shape 能跑通
    weight_cache = {}
    ok, err = try_forward(
        model,
        metas_dict,
        param_order,
        symint_params,
        data_input_params,
        weight_params,
        weight_cache=weight_cache,
        use_meta_device=use_meta,
    )
    if not ok:
        if verbose:
            print(f"  [WARN] Original shape fails: {err}")
        return {"model_dir": model_dir, "params": {}, "error": f"original fails: {err}"}

    if verbose:
        print("  [OK] Original shape runs successfully")

    result = {"model_dir": model_dir, "params": {}}
    batch_dynamic = False
    batch_original_value = None
    batch_successful_values = []

    # 只探测 axis=0 (batch 维度)
    for pname in data_input_params:
        meta = metas_dict.get(pname)
        if meta is None or not meta.shape or len(meta.shape) < 1:
            continue

        orig_val = meta.shape[0]  # axis=0
        if orig_val <= 0:
            continue

        # 记录第一个遇到的 batch 维度值
        if batch_original_value is None:
            batch_original_value = orig_val

        param_result = {
            "original_shape": meta.shape,
            "type": "data_input",
            "dims": {},
        }

        # 探测 axis=0
        dim_result = {
            "original_value": orig_val,
            "is_dynamic": False,
            "tested_values": {},
            "successful_values": [],
            "failed_values": [],
        }

        # 生成测试值
        test_values = set()
        for mult in PROBE_MULTIPLIERS:
            v = int(orig_val * mult)
            if v > 0 and v != orig_val:
                test_values.add(v)
        for v in PROBE_ABSOLUTE:
            if v != orig_val and v > 0:
                test_values.add(v)
        test_values = sorted(test_values)

        # 逐个测试
        for test_val in test_values[:10]:  # 限制测试数量
            new_shape = list(meta.shape)
            new_shape[0] = test_val
            shape_overrides = {pname: new_shape}

            ok_test, err_test = try_forward(
                model,
                metas_dict,
                param_order,
                symint_params,
                data_input_params,
                weight_params,
                shape_overrides=shape_overrides,
                use_meta_device=use_meta,
                weight_cache=weight_cache,
            )

            dim_result["tested_values"][test_val] = ok_test
            if ok_test:
                dim_result["successful_values"].append(test_val)
                batch_successful_values.append(test_val)
            else:
                dim_result["failed_values"].append(test_val)

        # 判断是否为动态
        if len(dim_result["successful_values"]) >= 2:
            dim_result["is_dynamic"] = True
            batch_dynamic = True

        param_result["dims"]["0"] = dim_result
        result["params"][pname] = param_result

    # 如果找到 batch 动态维度，保存到结果中
    if batch_dynamic:
        result["batch_dimension"] = {
            "original_value": batch_original_value,
            "successful_values": sorted(list(set(batch_successful_values))),
            "is_dynamic": True,
        }
        if verbose:
            print("  [SUCCESS] Batch dimension is DYNAMIC!")
            print(f"    Original: {batch_original_value}")
            print(
                f"    Successful values: {result['batch_dimension']['successful_values'][:10]}"
            )
    else:
        if verbose:
            print("  [INFO] Batch dimension is static (no successful values)")

    # 清理临时文件
    if temp_model_py and os.path.exists(temp_model_py):
        try:
            os.unlink(temp_model_py)
        except Exception:
            pass

    return result


def generate_batch_only_constraints(
    probe_result: Dict, model_dir: str, output_dir: str
):
    """只生成 batch 维度的 constraints"""
    batch_dim = probe_result.get("batch_dimension")
    if not batch_dim or not batch_dim["is_dynamic"]:
        print("  No dynamic batch dimension found, skipping constraint generation")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        constraint_file = os.path.join(tmpdir, "input_tensor_constraints.py")

        with open(constraint_file, "w") as f:
            f.write(
                """from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol('S0')

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: %d}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
"""
                % batch_dim["original_value"]
            )

            # 添加所有数据输入的 shape（batch 用 S0，其他维度保持原值）
            params = probe_result.get("params", {})
            for pname, pinfo in params.items():
                if pinfo["type"] == "data_input" and "0" in pinfo["dims"]:
                    orig_shape = pinfo["original_shape"]
                    # 第一个维度用 S0，其他保持原值
                    shape_str = "[S0"
                    for dim in orig_shape[1:]:
                        shape_str += f", {dim}"
                    shape_str += "]"
                    f.write(f"    ({shape_str}, '{pname}'),\n")

            f.write(
                """]
"""
            )

        # 写到输出目录
        # output_dir 可能已经是完整路径了（批量模式），也可能需要加上 basename（单模型模式）
        # 检查 output_dir 是否已经包含模型名
        if os.path.basename(output_dir) == os.path.basename(model_dir):
            # output_dir 已经是模型目录了
            out_dir = output_dir
        else:
            # 需要加上模型名
            out_dir = os.path.join(output_dir, os.path.basename(model_dir))

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "input_tensor_constraints.py")
        shutil.copy(constraint_file, out_file)

        print(f"  [GENERATED] {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-only Dimension Probe: 只探测 batch 维度"
    )
    parser.add_argument("--model-dir", type=str, help="单个 subgraph 目录")
    parser.add_argument("--base-dir", type=str, help="模型根目录 (配合 --model-list)")
    parser.add_argument("--model-list", type=str, help="model_list.txt 路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.model_dir:
        # 单个模型
        print(f"探测单个模型: {args.model_dir}")
        result = probe_batch_only_single_model(args.model_dir, verbose=args.verbose)

        if not result.get("error") and result.get("batch_dimension"):
            # 计算相对于某个根目录的相对路径
            # 简化处理：使用 basename，用户需要确保 base_dir 和 output_dir 结构一致
            # 或者添加一个参数来指定根目录
            out_dir = os.path.join(args.output_dir, os.path.basename(args.model_dir))
            os.makedirs(out_dir, exist_ok=True)

            generate_batch_only_constraints(result, args.model_dir, out_dir)

            # 构建 pipeline 期望的格式
            batch_dim = result["batch_dimension"]
            # dynamic_dims 的 key 应该是符号名 S0, S1...
            # batch 维度总是 S0
            dynamic_dims = {}
            if batch_dim["is_dynamic"]:
                dynamic_dims["S0"] = {
                    "original_value": batch_dim["original_value"],
                    "successful_values": batch_dim["successful_values"],
                    "failed_values": [],  # batch 维度应该没有失败的
                }

            probe_data = {
                "dynamic_dims": dynamic_dims,
                "batch_dimension": batch_dim,  # 额外保留
            }

            with open(os.path.join(out_dir, "probe_result.json"), "w") as f:
                json.dump(probe_data, f, indent=2)
            # 写 done.txt
            with open(os.path.join(out_dir, "done.txt"), "w") as f:
                f.write("status=ok\n")
            print("  [OK] Batch dimension is dynamic, constraints generated")
        else:
            print("  [INFO] No dynamic batch dimension found")

    elif args.base_dir and args.model_list:
        # 批量处理
        with open(args.model_list) as f:
            models = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        total = len(models)
        success = 0
        failed = 0

        print(f"开始探测 {total} 个模型的 batch 维度...")

        for i, rel_path in enumerate(models):
            model_dir = os.path.join(args.base_dir, rel_path)
            print(f"\n[{i+1}/{total}] {rel_path}")

            try:
                result = probe_batch_only_single_model(model_dir, verbose=False)
            except Exception as e:
                print(f"  [ERROR] Exception during probe: {e}")
                failed += 1
                continue

            if not result.get("error") and result.get("batch_dimension"):
                # 输出目录
                out_dir = os.path.join(args.output_dir, rel_path)
                os.makedirs(out_dir, exist_ok=True)

                # 生成 constraints（传递 out_dir）
                generate_batch_only_constraints(result, model_dir, out_dir)

                # 构建 pipeline 期望的格式
                batch_dim = result["batch_dimension"]
                # dynamic_dims 的 key 应该是符号名 S0...
                dynamic_dims = {}
                if batch_dim["is_dynamic"]:
                    dynamic_dims["S0"] = {
                        "original_value": batch_dim["original_value"],
                        "successful_values": batch_dim["successful_values"],
                        "failed_values": [],
                    }

                probe_data = {
                    "dynamic_dims": dynamic_dims,
                    "batch_dimension": batch_dim,
                }

                with open(os.path.join(out_dir, "probe_result.json"), "w") as f:
                    json.dump(probe_data, f, indent=2)
                with open(os.path.join(out_dir, "done.txt"), "w") as f:
                    f.write("status=ok\n")
                success += 1
                print("  [OK] Batch dimension is dynamic")
            else:
                failed += 1
                print("  [INFO] No dynamic batch dimension")

        print(f"\n完成: {total} 个模型")
        print(f"  Batch 动态: {success}")
        print(f"  Batch 静态: {failed}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
