"""
维度探测工具 (Dimension Probe)

方法：对已 extract 的 subgraph，逐个维度尝试不同值，通过"能否跑通"来判断该维度是否可泛化。

思路：
1. 从 weight_meta.py 解析每个 forward() 参数的 shape/dtype/data
2. 区分"数据输入"(torch.Tensor) 和"模型权重"(nn.parameter.Parameter) —— 权重 shape 不动
3. 对每个数据输入的每个维度，逐一尝试不同值 (2x, 0.5x 等)
4. 构造新 tensor，调用 forward()，能跑通 → 该维度可泛化
5. 输出报告 + 自动生成泛化后的 input_tensor_constraints.py

用法：
    python -m graph_net.agent.dimension_generalizer.dim_probe --model-dir /path/to/subgraph
    python -m graph_net.agent.dimension_generalizer.dim_probe --base-dir /path/to/try_models --model-list model_list.txt
"""

import argparse
import importlib.util
import inspect
import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 确保 GraphNet 根目录在 sys.path 中
_GRAPHNET_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _GRAPHNET_ROOT not in sys.path:
    sys.path.insert(0, _GRAPHNET_ROOT)

# 导入 batch 硬编码预处理工具
from graph_net.agent.dimension_generalizer.batch_hardcode_preprocess import (  # noqa: E402
    preprocess_model_py_for_batch,
)


# ============================================================
# 1. 解析 weight_meta.py 得到每个参数的元信息
# ============================================================


class ParamMeta:
    """存储一个 forward() 参数的元信息"""

    def __init__(
        self,
        name: str,
        shape: list,
        dtype: str,
        device: str,
        mean: Optional[float],
        std: Optional[float],
        data: Optional[list],
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.mean = mean
        self.std = std
        self.data = data

    def __repr__(self):
        return f"ParamMeta(name={self.name}, shape={self.shape}, dtype={self.dtype})"


def load_weight_metas(weight_meta_path: str) -> List[ParamMeta]:
    """从 weight_meta.py 中加载所有参数元信息"""
    spec = importlib.util.spec_from_file_location("weight_meta", weight_meta_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    metas = []
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if isinstance(obj, type) and attr_name.startswith(
            "Program_weight_tensor_meta_"
        ):
            meta = ParamMeta(
                name=getattr(obj, "name"),
                shape=list(getattr(obj, "shape")),
                dtype=getattr(obj, "dtype"),
                device=getattr(obj, "device", "cpu"),
                mean=getattr(obj, "mean", None),
                std=getattr(obj, "std", None),
                data=getattr(obj, "data", None),
            )
            metas.append(meta)
    return metas


# ============================================================
# 2. 加载 model.py 中的 GraphModule，分析 forward 签名
# ============================================================


def load_graph_module(model_py_path: str):
    """动态加载 model.py 并返回 GraphModule 实例"""

    spec = importlib.util.spec_from_file_location("model", model_py_path)
    mod = importlib.util.module_from_spec(spec)
    # model.py 可能 import torch 的子模块，确保 torch 已在 sys.modules
    spec.loader.exec_module(mod)
    graph_module_cls = getattr(mod, "GraphModule")
    return graph_module_cls()


def classify_params(model_py_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    分析 forward() 签名，把参数分为三类：
    - symint_params: torch.SymInt (如 s0)
    - data_input_params: torch.Tensor (数据输入，如 input_ids)
    - weight_params: torch.nn.parameter.Parameter (模型权重)
    返回各类参数名的列表
    """
    spec = importlib.util.spec_from_file_location("model_sig", model_py_path)
    mod = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(mod)
    graph_module_cls = getattr(mod, "GraphModule")
    sig = inspect.signature(graph_module_cls.forward)

    symint_params = []
    data_input_params = []
    weight_params = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        ann = param.annotation
        if ann == inspect.Parameter.empty:
            # 无标注，按名字猜测
            if pname.startswith("s") and pname[1:].isdigit():
                symint_params.append(pname)
            else:
                data_input_params.append(pname)
        else:
            ann_str = str(ann)
            if "SymInt" in ann_str:
                symint_params.append(pname)
            elif "Parameter" in ann_str:
                weight_params.append(pname)
            else:
                data_input_params.append(pname)

    return symint_params, data_input_params, weight_params


# ============================================================
# 3. 构造 tensor 的工具函数
# ============================================================


def _resolve_dtype(dtype_str: str):
    """将字符串 dtype 转换为 torch.dtype"""
    import torch

    dtype_map = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.bool": torch.bool,
    }
    return dtype_map.get(dtype_str, torch.float32)


def build_tensor(meta: ParamMeta, override_shape: Optional[list] = None):
    """根据 ParamMeta 构造一个 tensor"""
    import torch

    shape = override_shape if override_shape is not None else meta.shape
    dtype = _resolve_dtype(meta.dtype)
    device = meta.device

    if not shape:
        # 标量 (如 s0)
        if meta.data is not None and len(meta.data) > 0:
            return meta.data[0]  # 直接返回 python int
        return 0

    if meta.data is not None and override_shape is None:
        # 有 literal data 且没有 override shape，直接用
        import math

        total = math.prod(shape)
        if len(meta.data) == total:
            t = torch.tensor(meta.data, dtype=dtype, device=device).reshape(shape)
            return t

    # 否则随机生成
    if dtype in (torch.int64, torch.int32):
        import math

        total = math.prod(shape)
        name_lower = (meta.name or "").lower()
        # 位置型索引 tensor: 生成递增序列 [0, 1, ..., n-1]
        if "position" in name_lower or "cache_position" in name_lower:
            t = torch.arange(total, dtype=dtype, device=device).reshape(shape)
        elif meta.data is not None and len(meta.data) > 0:
            # 有原始 data，用同范围的随机值
            max_val = max(meta.data) + 1
            t = torch.randint(0, max(max_val, 2), shape, dtype=dtype, device=device)
        else:
            t = torch.randint(0, 1000, shape, dtype=dtype, device=device)
    elif dtype == torch.bool:
        t = torch.randint(0, 2, shape, dtype=torch.int32, device=device).bool()
    else:
        if meta.mean is not None and meta.std is not None:
            t = (
                torch.randn(shape, dtype=torch.float32, device=device) * meta.std
                + meta.mean
            )
            t = t.to(dtype)
        else:
            t = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    return t


# ============================================================
# 4. 核心：试跑 forward
# ============================================================


def try_forward(
    model,
    metas_dict: Dict[str, ParamMeta],
    param_order: list,
    symint_params: list,
    data_input_params: list,
    weight_params: list,
    shape_overrides: Dict[str, list] = None,
    symint_overrides: Dict[str, int] = None,
    weight_cache: Dict = None,
    use_meta_device: bool = False,
) -> Tuple[bool, str]:
    """
    尝试用给定 shape 跑一次 forward。
    shape_overrides: {param_name: new_shape}
    symint_overrides: {param_name: new_int_value}
    weight_cache: 缓存已创建的权重 tensor，避免重复分配
    use_meta_device: True 时使用 FakeTensorMode 做零内存 shape 推断
    返回 (success, error_msg)
    """
    import torch

    if shape_overrides is None:
        shape_overrides = {}
    if symint_overrides is None:
        symint_overrides = {}
    if weight_cache is None:
        weight_cache = {}

    if use_meta_device:
        # 使用 FakeTensorMode: 不分配真实内存，只做 shape 推断
        try:
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode(allow_non_fake_inputs=True):
                args = []
                for pname in param_order:
                    if pname not in metas_dict:
                        # 参数不在 metas_dict 中，说明有默认值，跳过（不传）
                        continue
                    if pname in symint_params:
                        if pname in symint_overrides:
                            args.append(symint_overrides[pname])
                        else:
                            meta = metas_dict[pname]
                            args.append(meta.data[0] if meta.data else 0)
                    else:
                        meta = metas_dict[pname]
                        override = shape_overrides.get(pname)
                        shape = override if override is not None else meta.shape
                        if not shape:
                            if meta.data is not None and len(meta.data) > 0:
                                args.append(meta.data[0])
                            else:
                                args.append(0)
                        else:
                            dtype = _resolve_dtype(meta.dtype)
                            # 用模型原始 device，FakeTensorMode 下不分配真实 GPU 内存
                            device = getattr(meta, "device", "cpu") or "cpu"
                            fake_t = torch.empty(shape, dtype=dtype, device=device)
                            args.append(fake_t)
                with torch.no_grad():
                    model(*args)
                return True, ""
        except Exception as e:
            return False, str(e)
    else:
        args = []
        for pname in param_order:
            if pname not in metas_dict:
                # 参数不在 metas_dict 中，说明有默认值，跳过（不传）
                continue
            if pname in symint_params:
                if pname in symint_overrides:
                    args.append(symint_overrides[pname])
                else:
                    meta = metas_dict[pname]
                    args.append(meta.data[0] if meta.data else 0)
            else:
                meta = metas_dict[pname]
                override = shape_overrides.get(pname)
                is_weight = pname in weight_params
                if is_weight and pname in weight_cache and override is None:
                    args.append(weight_cache[pname])
                else:
                    t = build_tensor(meta, override_shape=override)
                    if is_weight and override is None:
                        weight_cache[pname] = t
                    args.append(t)

    try:
        with torch.no_grad():
            model(*args)
        return True, ""
    except Exception as e:
        return False, str(e)


# ============================================================
# 5. 维度探测逻辑
# ============================================================

# 常见的探测维度值
PROBE_MULTIPLIERS = [2, 4, 0.5]  # 相对于原始值的倍数
PROBE_ABSOLUTE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # 绝对值
MODEL_TIMEOUT = 120  # 单模型超时 2 分钟


def _auto_fix_symint(
    model,
    metas_dict,
    param_order,
    symint_params,
    data_input_params,
    weight_params,
    verbose,
    weight_cache=None,
    use_meta_device=False,
):
    """
    当原始 shape 跑不通时，尝试自动修复 s0 的值。
    常见原因：s0 的 data=[4] 但实际应该等于某个数据输入的维度值 (如 seq_len=32)。
    策略：收集数据输入中所有出现的维度值，逐一尝试替换 s0。
    返回 (fixed, symint_fixes: dict) — symint_fixes = {s0: correct_value}
    """
    # 收集数据输入中所有出现的维度值
    candidate_values = set()
    for dp in data_input_params:
        meta = metas_dict.get(dp)
        if meta and meta.shape:
            for v in meta.shape:
                if v > 0:
                    candidate_values.add(v)

    for sp in symint_params:
        sp_meta = metas_dict.get(sp)
        if sp_meta is None or sp_meta.data is None:
            continue
        orig_val = sp_meta.data[0]

        for cand in sorted(candidate_values):
            if cand == orig_val:
                continue
            symint_overrides = {sp: cand}
            ok, err = try_forward(
                model,
                metas_dict,
                param_order,
                symint_params,
                data_input_params,
                weight_params,
                symint_overrides=symint_overrides,
                weight_cache=weight_cache or {},
                use_meta_device=use_meta_device,
            )
            if ok:
                if verbose:
                    print(
                        f"  [AUTO-FIX] {sp}: {orig_val} -> {cand} (makes model runnable)"
                    )
                # 更新 metas_dict 中 s0 的 data
                sp_meta.data = [cand]
                return True, {sp: cand}

    return False, {}


def probe_single_model(model_dir: str, verbose: bool = True) -> Dict:
    """
    对一个 subgraph 目录进行维度探测。

    返回格式:
    {
        "model_dir": str,
        "params": {
            param_name: {
                "original_shape": [int, ...],
                "type": "data_input" | "weight" | "symint",
                "dims": {
                    axis_index: {
                        "original_value": int,
                        "is_dynamic": bool,
                        "tested_values": {value: success_bool, ...}
                    }
                }
            }
        }
    }
    """

    model_dir = str(Path(model_dir).resolve())
    model_py = os.path.join(model_dir, "model.py")
    weight_meta_py = os.path.join(model_dir, "weight_meta.py")

    if not os.path.exists(model_py) or not os.path.exists(weight_meta_py):
        print(f"  [SKIP] {model_dir}: missing model.py or weight_meta.py")
        return {"model_dir": model_dir, "params": {}, "error": "missing files"}

    # 预处理：替换 batch 硬编码，以便 dim_probe 能正确探测 batch 维度
    # 将 model.py 拷贝到临时文件进行预处理
    import tempfile

    temp_model_py = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            temp_model_py = tmp.name

        _, replacements = preprocess_model_py_for_batch(
            model_py, temp_model_py, verbose=verbose
        )

        # 如果有替换，使用预处理后的文件
        if replacements:
            model_py = temp_model_py

    finally:
        # 注意：不要立即删除 temp_model_py，因为模型加载需要它
        # 会在函数结束时被 GC 清理
        pass

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
        print(f"  SymInt params: {symint_params}")
        print(f"  Data input params: {data_input_params}")
        print(f"  Weight params: {len(weight_params)} parameters")
        print(f"{'='*60}")

    # 权重参数很多时，用 meta device 避免 OOM
    use_meta = len(weight_params) > 50
    # 环境变量强制使用 FakeTensorMode（用于重试超时/OOM模型）
    if os.environ.get("DIM_PROBE_FORCE_FAKE") == "1":
        use_meta = True
    if use_meta and verbose:
        print(
            f"  [INFO] Using meta device for shape-only probing ({len(weight_params)} weight params)"
        )

    # 先验证原始 shape 能跑通
    weight_cache = {}  # 缓存权重 tensor，避免多次分配
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
            print("  [INFO] Trying auto-fix for SymInt values...")

        # 尝试自动修复 symint 值
        fixed, fixes = _auto_fix_symint(
            model,
            metas_dict,
            param_order,
            symint_params,
            data_input_params,
            weight_params,
            verbose,
            weight_cache=weight_cache,
            use_meta_device=use_meta,
        )
        if not fixed:
            print(f"  [ERROR] Cannot fix, skipping: {err}")
            return {
                "model_dir": model_dir,
                "params": {},
                "error": f"original fails: {err}",
            }

    if verbose:
        print("  [OK] Original shape runs successfully")

    result = {"model_dir": model_dir, "params": {}}

    # 对每个数据输入参数的每个维度进行探测
    for pname in data_input_params:
        meta = metas_dict.get(pname)
        if meta is None:
            continue
        if not meta.shape:
            continue  # 标量，跳过

        param_result = {"original_shape": meta.shape, "type": "data_input", "dims": {}}

        for axis, orig_val in enumerate(meta.shape):
            if orig_val <= 0:
                continue

            dim_result = {
                "original_value": orig_val,
                "is_dynamic": False,
                "tested_values": {},
            }

            # 生成要测试的值
            test_values = set()
            for mult in PROBE_MULTIPLIERS:
                v = int(orig_val * mult)
                if v > 0 and v != orig_val:
                    test_values.add(v)
            for v in PROBE_ABSOLUTE:
                if v != orig_val and v > 0:
                    test_values.add(v)
            test_values = sorted(test_values)

            any_success = False
            success_count = 0
            for test_val in test_values:
                # 早停：已有 2 个成功值就够确认是动态维度
                if success_count >= 6:
                    break
                # 构造新 shape
                new_shape = list(meta.shape)
                new_shape[axis] = test_val
                shape_overrides = {pname: new_shape}

                # 同步修改 s0 的值（如果存在的话）
                # s0 通常表示 seq_len，对应数据输入中某个维度
                # 在 subgraph_1 的 model.py 中, s0 用于 arange(s0)
                # 我们需要把 s0 也同步改成匹配的值
                symint_overrides = {}

                # 对于有多个数据输入的情况，需要联动调整所有包含该维度的输入
                # 例如 subgraph_1: position_ids=[1,32], cache_position=[32]
                # 如果 seq_len=32 变了，两个输入都要变
                linked_overrides = {}
                for other_pname in data_input_params:
                    if other_pname == pname:
                        continue
                    other_meta = metas_dict.get(other_pname)
                    if other_meta is None or not other_meta.shape:
                        continue
                    other_shape = list(other_meta.shape)
                    changed = False
                    for other_axis, other_val in enumerate(other_shape):
                        if other_val == orig_val:
                            other_shape[other_axis] = test_val
                            changed = True
                    if changed:
                        linked_overrides[other_pname] = other_shape

                all_overrides = {**shape_overrides, **linked_overrides}

                # 同步 symint
                for sp in symint_params:
                    sp_meta = metas_dict.get(sp)
                    if sp_meta and sp_meta.data:
                        sp_val = sp_meta.data[0]
                        # 如果 s0 的值等于某个被改变的维度的原始值
                        if sp_val == orig_val:
                            symint_overrides[sp] = test_val
                        # 常见情况：s0 是 seq_len 的某种表示
                        # 如果 s0 不等于 orig_val，检查是否需要更新
                        # (暂不处理复杂关联)

                ok, err = try_forward(
                    model,
                    metas_dict,
                    param_order,
                    symint_params,
                    data_input_params,
                    weight_params,
                    shape_overrides=all_overrides,
                    symint_overrides=symint_overrides,
                    weight_cache=weight_cache,
                    use_meta_device=use_meta,
                )
                dim_result["tested_values"][test_val] = ok
                if ok:
                    any_success = True
                    success_count += 1

            dim_result["is_dynamic"] = any_success
            param_result["dims"][axis] = dim_result

            if verbose:
                status = "DYNAMIC" if any_success else "STATIC"
                succeeded = [v for v, ok in dim_result["tested_values"].items() if ok]
                failed = [v for v, ok in dim_result["tested_values"].items() if not ok]
                print(
                    f"  {pname}[axis={axis}] orig={orig_val}: [{status}]"
                    f"  ok={succeeded[:5]}{'...' if len(succeeded)>5 else ''}"
                    f"  fail={failed[:3]}{'...' if len(failed)>3 else ''}"
                )

        result["params"][pname] = param_result

    # 也探测 symint 参数
    for pname in symint_params:
        meta = metas_dict.get(pname)
        if meta is None or meta.data is None:
            continue
        orig_val = meta.data[0]
        if orig_val <= 0:
            continue

        param_result = {
            "original_shape": [],
            "original_value": orig_val,
            "type": "symint",
            "dims": {},
        }

        dim_result = {
            "original_value": orig_val,
            "is_dynamic": False,
            "tested_values": {},
        }

        test_values = set()
        for v in PROBE_ABSOLUTE:
            if v != orig_val and v > 0:
                test_values.add(v)
        test_values = sorted(test_values)

        any_success = False
        success_count = 0
        for test_val in test_values:
            if success_count >= 2:
                break
            symint_overrides = {pname: test_val}

            # 同步修改所有数据输入中与 orig_val 匹配的维度
            all_shape_overrides = {}
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
                    all_shape_overrides[dp] = new_shape

            ok, err = try_forward(
                model,
                metas_dict,
                param_order,
                symint_params,
                data_input_params,
                weight_params,
                shape_overrides=all_shape_overrides,
                symint_overrides=symint_overrides,
                weight_cache=weight_cache,
                use_meta_device=use_meta,
            )
            dim_result["tested_values"][test_val] = ok
            if ok:
                any_success = True
                success_count += 1

        dim_result["is_dynamic"] = any_success
        param_result["dims"][0] = dim_result

        if verbose:
            status = "DYNAMIC" if any_success else "STATIC"
            succeeded = [v for v, ok in dim_result["tested_values"].items() if ok]
            print(
                f"  {pname} (SymInt) orig={orig_val}: [{status}]"
                f"  ok={succeeded[:5]}{'...' if len(succeeded)>5 else ''}"
            )

        result["params"][pname] = param_result

    return result


# ============================================================
# 6. 生成泛化后的 input_tensor_constraints.py
# ============================================================


def generate_constraints(probe_result: Dict, model_dir: str, output_dir: str):
    """根据探测结果，生成带 sympy Symbol 的 input_tensor_constraints.py"""

    params = probe_result.get("params", {})
    if not params:
        return

    # 找出所有被标记为 dynamic 的 (param_name, axis) 对
    # 以及它们的原始值
    dynamic_dims = []  # [(param_name, axis, orig_val)]
    for pname, pinfo in params.items():
        if pinfo["type"] == "symint":
            for axis_str, dinfo in pinfo["dims"].items():
                if dinfo["is_dynamic"]:
                    dynamic_dims.append((pname, int(axis_str), dinfo["original_value"]))
        elif pinfo["type"] == "data_input":
            for axis_str, dinfo in pinfo["dims"].items():
                if dinfo["is_dynamic"]:
                    dynamic_dims.append((pname, int(axis_str), dinfo["original_value"]))

    if not dynamic_dims:
        print("  No dynamic dims found, skipping constraint generation")
        return

    # 按原始值分组，同一原始值共享同一个 Symbol
    val_to_symbol = {}
    symbol_idx = 0
    for pname, axis, orig_val in dynamic_dims:
        if orig_val not in val_to_symbol:
            val_to_symbol[orig_val] = f"S{symbol_idx}"
            symbol_idx += 1

    # 加载原始 weight_meta 获取所有参数的 shape
    weight_meta_path = os.path.join(model_dir, "weight_meta.py")
    metas = load_weight_metas(weight_meta_path)
    metas_dict = {m.name: m for m in metas}

    # 分析 forward 签名获取参数顺序和类型
    model_py_path = os.path.join(model_dir, "model.py")
    symint_params, data_input_params, weight_params = classify_params(model_py_path)

    spec = importlib.util.spec_from_file_location("model_gen", model_py_path)
    mod = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(mod)
    sig = inspect.signature(getattr(mod, "GraphModule").forward)
    param_order = [p for p in sig.parameters if p != "self"]

    # 建立 (pname, axis) -> symbol_name 的映射
    dynamic_map = {}  # (pname, axis) -> symbol_name
    for pname, axis, orig_val in dynamic_dims:
        dynamic_map[(pname, axis)] = val_to_symbol[orig_val]

    # 生成代码
    symbols = sorted(set(val_to_symbol.values()), key=lambda s: int(s[1:]))
    symbol2example = {}
    for orig_val, sym_name in val_to_symbol.items():
        symbol2example[sym_name] = orig_val

    lines = []
    lines.append("")
    lines.append("from sympy import Symbol, Expr, Rel, Eq")
    lines.append("")

    for s in symbols:
        lines.append(f"{s} = Symbol('{s}')")
    lines.append("")

    lines.append(f"dynamic_dim_constraint_symbols = [{', '.join(symbols)}]")
    lines.append("")

    example_parts = [f"{s}: {symbol2example[s]}" for s in symbols]
    lines.append(
        f"dynamic_dim_constraint_symbol2example_value = {{{', '.join(example_parts)}}}"
    )
    lines.append("")

    lines.append("dynamic_dim_constraint_relations = []")
    lines.append("")

    # 构建 input_shapes
    shape_entries = []
    for pname in param_order:
        meta = metas_dict.get(pname)
        if meta is None:
            continue

        if not meta.shape:
            # 标量 (SymInt like s0)
            shape_entries.append(f"([], '{pname}')")
        else:
            shape_parts = []
            for axis, dim_val in enumerate(meta.shape):
                sym_name = dynamic_map.get((pname, axis))
                if sym_name is not None:
                    shape_parts.append(sym_name)
                else:
                    # 也检查是否与某个 dynamic dim 的原始值相同
                    # 且该参数是数据输入（非权重）
                    if pname in data_input_params and dim_val in val_to_symbol:
                        # 检查这个参数的这个 axis 是否也探测通过
                        if (pname, axis) in dynamic_map:
                            shape_parts.append(dynamic_map[(pname, axis)])
                        else:
                            shape_parts.append(str(dim_val))
                    else:
                        shape_parts.append(str(dim_val))
            shape_str = f"[{', '.join(shape_parts)}]"
            shape_entries.append(f"({shape_str}, '{pname}')")

    lines.append(
        "dynamic_dim_constraint_input_shapes = [" + ", ".join(shape_entries) + "]"
    )
    lines.append("")

    # 输出到文件
    out_path = os.path.join(output_dir, "input_tensor_constraints.py")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  [GENERATED] {out_path}")
    print(f"    Symbols: {symbols}")
    print(f"    Example values: {symbol2example}")


def save_probe_result_json(probe_result: Dict, output_dir: str):
    """将 probe 详情保存为 probe_result.json，记录每个 Symbol 的成功/失败值。

    格式:
    {
      "dynamic_dims": {
        "S0": {"original_value": 32, "successful_values": [1,2,4,...], "failed_values": [...]},
        "S1": {"original_value": 128, "successful_values": [...], "failed_values": [...]}
      }
    }
    """
    params = probe_result.get("params", {})
    if not params:
        return

    # 收集所有动态维度的 (orig_val -> tested_values)
    # 同一 orig_val 合并（因为它们共享同一个 Symbol）
    val_to_tested = {}  # orig_val -> {test_val: bool, ...}
    for pname, pinfo in params.items():
        for axis_str, dinfo in pinfo.get("dims", {}).items():
            if not dinfo.get("is_dynamic"):
                continue
            orig_val = dinfo["original_value"]
            if orig_val not in val_to_tested:
                val_to_tested[orig_val] = {}
            # 合并 tested_values（取 OR：任一参数成功即算成功）
            for tv, ok in dinfo.get("tested_values", {}).items():
                if ok or tv not in val_to_tested[orig_val]:
                    val_to_tested[orig_val][tv] = ok

    if not val_to_tested:
        return

    # 按 orig_val 排序分配 Symbol 名
    symbol_idx = 0
    dynamic_dims = {}
    for orig_val in sorted(val_to_tested.keys()):
        sym_name = f"S{symbol_idx}"
        symbol_idx += 1
        tested = val_to_tested[orig_val]
        dynamic_dims[sym_name] = {
            "original_value": orig_val,
            "successful_values": sorted([int(v) for v, ok in tested.items() if ok]),
            "failed_values": sorted([int(v) for v, ok in tested.items() if not ok]),
        }

    out_path = os.path.join(output_dir, "probe_result.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"dynamic_dims": dynamic_dims}, f, indent=2)


def probe_joint_combos(
    model_dir: str,
    probe_result: Dict,
    output_dir: str,
    max_combos: int = 10,
    verbose: bool = False,
):
    """联合探测：用已发现的成功值做组合，直接试跑验证。

    在 dim_probe 阶段完成，把验证通过的组合写入 probe_result.json 的 verified_combos 字段。
    pipeline 直接用这些组合，保证 100% 验证通过。
    """
    from itertools import product as iterproduct

    probe_json_path = os.path.join(output_dir, "probe_result.json")
    if not os.path.exists(probe_json_path):
        return

    with open(probe_json_path) as f:
        probe_data = json.load(f)

    dynamic_dims = probe_data.get("dynamic_dims", {})
    if not dynamic_dims:
        return

    # 加载模型（和 probe_single_model 一样的流程）
    model_py = os.path.join(model_dir, "model.py")
    weight_meta_py = os.path.join(model_dir, "weight_meta.py")
    if not os.path.exists(model_py) or not os.path.exists(weight_meta_py):
        return

    from graph_net.tensor_meta import TensorMeta

    metas = TensorMeta.unserialize_from_py_file(weight_meta_py)
    metas_dict = {m.name: m for m in metas}
    param_order = [m.name for m in metas]

    spec = importlib.util.spec_from_file_location("model_module", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model_class = getattr(mod, "GraphModule")
    model = model_class()

    symint_params = [n for n in param_order if n.startswith("s")]
    weight_params = [
        n for n in param_order if not n.startswith("L_") and not n.startswith("s")
    ]
    data_input_params = [n for n in param_order if n.startswith("L_")]

    n_weights = len(weight_params)
    use_meta = n_weights > 50
    weight_cache = {}

    # 收集每个 Symbol 的候选值（成功值 + 原始值）
    sym_names = sorted(dynamic_dims.keys(), key=lambda s: int(s[1:]))
    per_sym_vals = []
    for name in sym_names:
        info = dynamic_dims[name]
        vals = list(set(info["successful_values"] + [info["original_value"]]))
        vals.sort()
        per_sym_vals.append(vals)

    # 建立 Symbol 原始值到参数维度的映射
    # 需要知道每个 Symbol 对应哪些 (param, axis)
    sym_orig_vals = [dynamic_dims[name]["original_value"] for name in sym_names]

    # 笛卡尔积生成候选组合，逐个试跑
    verified_combos = []
    tested = 0
    for combo in iterproduct(*per_sym_vals):
        if len(verified_combos) >= max_combos:
            break
        tested += 1
        if tested > max_combos * 5:  # 最多试 50 个组合
            break

        # 构造 shape_overrides 和 symint_overrides
        shape_overrides = {}
        symint_overrides = {}

        # 对每个数据输入参数，替换对应维度
        for dp in data_input_params:
            dp_meta = metas_dict.get(dp)
            if dp_meta is None or not dp_meta.shape:
                continue
            new_shape = list(dp_meta.shape)
            changed = False
            for axis, dim_val in enumerate(new_shape):
                for si, orig_val in enumerate(sym_orig_vals):
                    if dim_val == orig_val:
                        new_shape[axis] = combo[si]
                        changed = True
                        break
            if changed:
                shape_overrides[dp] = new_shape

        # 对每个 symint 参数，替换值
        for sp in symint_params:
            sp_meta = metas_dict.get(sp)
            if sp_meta and sp_meta.data:
                sp_val = sp_meta.data[0]
                for si, orig_val in enumerate(sym_orig_vals):
                    if sp_val == orig_val:
                        symint_overrides[sp] = combo[si]
                        break

        ok, err = try_forward(
            model,
            metas_dict,
            param_order,
            symint_params,
            data_input_params,
            weight_params,
            shape_overrides=shape_overrides,
            symint_overrides=symint_overrides,
            weight_cache=weight_cache,
            use_meta_device=use_meta,
        )
        if ok:
            combo_dict = {sym_names[i]: combo[i] for i in range(len(sym_names))}
            verified_combos.append(combo_dict)
            if verbose:
                print(f"  [COMBO OK] {combo_dict}")
        elif verbose:
            combo_dict = {sym_names[i]: combo[i] for i in range(len(sym_names))}
            print(f"  [COMBO FAIL] {combo_dict}: {err}")

    # 写回 probe_result.json
    probe_data["verified_combos"] = verified_combos
    with open(probe_json_path, "w") as f:
        json.dump(probe_data, f, indent=2)

    if verbose:
        print(f"  [JOINT] 测试了 {tested} 个组合，验证通过 {len(verified_combos)} 个")


# ============================================================
# 7. 批量处理 + 输出完整报告
# ============================================================


def print_report(all_results: List[Dict]):
    """打印汇总报告"""
    print("\n" + "=" * 70)
    print("  DIMENSION PROBE REPORT")
    print("=" * 70)

    for r in all_results:
        model_dir = r["model_dir"]
        print(f"\n  Model: {Path(model_dir).name}")

        if r.get("error"):
            print(f"    Error: {r['error']}")
            continue

        params = r.get("params", {})
        if not params:
            print("    No data-input parameters found")
            continue

        for pname, pinfo in params.items():
            ptype = pinfo["type"]
            dims = pinfo.get("dims", {})

            for axis_str, dinfo in dims.items():
                status = "DYNAMIC" if dinfo["is_dynamic"] else "STATIC"
                orig = dinfo["original_value"]
                succeeded = sorted(
                    [v for v, ok in dinfo["tested_values"].items() if ok]
                )

                if ptype == "symint":
                    print(f"    {pname} (SymInt): orig={orig} -> [{status}]", end="")
                else:
                    print(
                        f"    {pname}[axis={axis_str}]: orig={orig} -> [{status}]",
                        end="",
                    )

                if succeeded:
                    print(
                        f"  works_with={succeeded[:8]}{'...' if len(succeeded)>8 else ''}"
                    )
                else:
                    print()

    print("\n" + "=" * 70)


NUM_WORKERS = 4  # 并行 worker 数


def _run_single_model_worker(args_tuple):
    """在线程池中执行单个模型的子进程探测，返回 (rel_path, status, detail)"""
    import subprocess
    import resource
    import time

    rel_path, base_dir, output_dir, verbose = args_tuple
    model_dir = os.path.join(base_dir, rel_path)
    out_model_dir = os.path.join(output_dir, rel_path) if output_dir else None
    worker_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dim_probe.py"
    )

    MEM_LIMIT_GB = 30
    cmd = [sys.executable, worker_script, "--model-dir", model_dir]
    if output_dir:
        cmd += ["--output-dir", out_model_dir]
    if not verbose:
        cmd += ["--quiet"]

    def _limit_mem():
        mem_bytes = MEM_LIMIT_GB * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    model_start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            timeout=MODEL_TIMEOUT,
            capture_output=True,
            text=True,
            preexec_fn=_limit_mem,
        )
        elapsed = time.time() - model_start

        if proc.returncode == 0:
            n_dyn = 0
            if out_model_dir and os.path.exists(
                os.path.join(out_model_dir, "probe_result.json")
            ):
                try:
                    with open(os.path.join(out_model_dir, "probe_result.json")) as pf:
                        pr = json.load(pf)
                    n_dyn = len(pr.get("dynamic_dims", {}))
                except Exception:
                    pass
            # 写 done.txt
            if out_model_dir:
                os.makedirs(out_model_dir, exist_ok=True)
                with open(os.path.join(out_model_dir, "done.txt"), "w") as f:
                    f.write("status=ok\n")
            if n_dyn > 0:
                return (rel_path, "ok", f"dynamic={n_dyn} ({elapsed:.0f}s)")
            else:
                return (rel_path, "ok", f"(all static) ({elapsed:.0f}s)")
        else:
            err_msg = (proc.stderr or proc.stdout or "unknown")[-200:]
            if out_model_dir:
                os.makedirs(out_model_dir, exist_ok=True)
                with open(os.path.join(out_model_dir, "done.txt"), "w") as f:
                    f.write(f"status=error\nerror=exit_{proc.returncode}\n")
            return (
                rel_path,
                "error",
                f"exit={proc.returncode} ({elapsed:.0f}s) {err_msg}",
            )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - model_start
        # 杀死超时子进程
        try:
            proc.kill()
        except Exception:
            pass
        if out_model_dir:
            os.makedirs(out_model_dir, exist_ok=True)
            with open(os.path.join(out_model_dir, "done.txt"), "w") as f:
                f.write("status=timeout\n")
        return (rel_path, "timeout", f"exceeded {MODEL_TIMEOUT}s ({elapsed:.0f}s)")
    except Exception as e:
        if out_model_dir:
            os.makedirs(out_model_dir, exist_ok=True)
            with open(os.path.join(out_model_dir, "done.txt"), "w") as f:
                f.write(f"status=error\nerror={e}\n")
        return (rel_path, "error", str(e))


def process_model_list(
    base_dir: str,
    model_list_path: str,
    output_dir: str,
    verbose: bool,
    resume: bool = False,
):
    """处理 model_list.txt 中的所有模型（并行版，NUM_WORKERS 个 worker）

    Args:
        resume: 如果为 True，跳过已有 done.txt 标记的模型
    """
    import time
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open(model_list_path) as f:
        models = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    # 写时间戳日志（加锁保证线程安全）
    import threading

    log_lock = threading.Lock()
    log_file = os.path.join(output_dir, "run_log.txt") if output_dir else None

    def log(msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if log_file:
            with log_lock:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "a") as f:
                    f.write(line + "\n")

    total = len(models)

    # 过滤：跳过不存在的目录和已完成的模型
    todo_models = []
    skipped = 0
    for rel_path in models:
        model_dir = os.path.join(base_dir, rel_path)
        if not os.path.isdir(model_dir):
            skipped += 1
            continue
        if resume and output_dir:
            done_marker = os.path.join(output_dir, rel_path, "done.txt")
            if os.path.exists(done_marker):
                skipped += 1
                continue
        todo_models.append(rel_path)

    log(
        f"dim_probe 开始: {total} models, todo={len(todo_models)}, "
        f"skip={skipped}, resume={resume}, workers={NUM_WORKERS}"
    )

    succeeded = 0
    failed = 0
    start_time = time.time()
    done_count = 0

    # 并行执行
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for rel_path in todo_models:
            fut = executor.submit(
                _run_single_model_worker, (rel_path, base_dir, output_dir, verbose)
            )
            futures[fut] = rel_path

        try:
            for fut in as_completed(futures):
                rel_path, status, detail = fut.result()
                done_count += 1

                if status == "ok":
                    succeeded += 1
                    log(f"[OK] {rel_path} {detail}")
                elif status == "timeout":
                    failed += 1
                    log(f"[TIMEOUT] {rel_path}: {detail}")
                else:
                    failed += 1
                    log(f"[ERROR] {rel_path}: {detail}")

                # 定期进度报告
                if done_count % 50 == 0 or done_count == 1:
                    elapsed = time.time() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (len(todo_models) - done_count) / rate if rate > 0 else 0
                    log(
                        f"[Progress] {done_count}/{len(todo_models)} "
                        f"({done_count/len(todo_models)*100:.1f}%) "
                        f"ok={succeeded} fail={failed} "
                        f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
                    )
        except KeyboardInterrupt:
            log("[INTERRUPT] KeyboardInterrupt, cancelling remaining tasks...")
            executor.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - start_time
    log(f"dim_probe 完成: {total} models, {elapsed:.0f}s")
    log(f"  成功: {succeeded}, 失败: {failed}, 跳过: {skipped}")
    return []


# ============================================================
# main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Dimension Probe: 通过试跑探测动态维度")
    parser.add_argument("--model-dir", type=str, help="单个 subgraph 目录")
    parser.add_argument("--base-dir", type=str, help="模型根目录 (配合 --model-list)")
    parser.add_argument("--model-list", type=str, help="model_list.txt 路径")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录，保存泛化后的 input_tensor_constraints.py",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已有输出的模型")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.model_dir:
        result = probe_single_model(args.model_dir, verbose=verbose)
        if args.output_dir and not result.get("error"):
            generate_constraints(result, args.model_dir, args.output_dir)
            save_probe_result_json(result, args.output_dir)
            # 联合探测
            probe_joint_combos(
                args.model_dir, result, args.output_dir, max_combos=10, verbose=verbose
            )
            # 拷贝原始文件
            for fname in [
                "model.py",
                "weight_meta.py",
                "graph_net.json",
                "input_meta.py",
            ]:
                src = os.path.join(args.model_dir, fname)
                dst = os.path.join(args.output_dir, fname)
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
        if verbose:
            print_report([result])

    elif args.base_dir and args.model_list:
        from datetime import datetime

        output_dir = args.output_dir or (args.base_dir + "_probe_output")
        # 非 resume 时，输出目录名加时间戳
        if not args.resume:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir.rstrip("/") + f"_{ts}"
        process_model_list(
            args.base_dir,
            args.model_list,
            output_dir,
            verbose=verbose,
            resume=args.resume,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
