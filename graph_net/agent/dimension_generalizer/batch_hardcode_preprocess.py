"""
Batch 维度硬编码预处理工具

用于 dim_probe 前预处理 model.py，将明显的 batch 硬编码替换为动态取值。

核心思路：
1. 识别 reshape/view/expand/reshape_as/repeat 等操作中第一个参数为 1 的情况
2. 判断这个 1 是否代表 batch 维度（通常是第一个维度）
3. 替换为动态取值（如 tensor.size(0)）

注意：
- 只替换"明显"的 batch 硬编码，避免误伤
- 替换后 dim_probe 才能正确探测 batch 维度的动态性
"""

import re
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class Replacement:
    """记录一次替换"""

    line_num: int
    original: str
    modified: str
    reason: str


def _find_tensor_name_for_line(line: str, prev_lines: List[str]) -> str:
    """找到当前操作对应的 tensor 变量名"""
    line = line.strip()

    # 模式1: x = tensor.reshape(1, ...)  -> 找 tensor
    match = re.match(
        r"(\w+)\s*=\s*([^\s\(]+)\.(?:reshape|view|expand|repeat)\s*\(", line
    )
    if match:
        source = match.group(2)
        # 如果 source 是简单的变量名，直接返回
        if re.match(r"^[a-zA-Z_]\w*$", source):
            return source

    # 模式2: 变量在右侧，如 linear = torch.nn.functional.linear(...) 然后下一行 reshape = linear.reshape(...)
    # 这种情况需要看上一行
    # 找上一行的赋值，看哪个变量在这个 reshape 之前定义
    for prev_line in reversed(prev_lines[-5:]):
        prev_line = prev_line.strip()
        # 匹配: variable = ... (分号结尾)
        match = re.match(r"(\w+)\s*=", prev_line)
        if match:
            var_name = match.group(1)
            # 检查这个变量是否在当前行被使用
            if var_name + "." in line:
                return var_name

    # 默认：尝试从赋值左侧推断
    # 比如: reshape = linear.reshape(...)  -> 返回 linear
    match = re.match(r"(\w+)\s*=\s*([^\s\(]+)\.(?:reshape|view|expand|repeat)", line)
    if match:
        source = match.group(2)
        if re.match(r"^[a-zA-Z_]\w*$", source):
            return source

    return None


def _is_likely_batch_hardcoded(shape_str: str) -> bool:
    """
    判断 shape 开头的 1 是否是 batch 维度的硬编码

    启发式规则：
    1. shape 以 1, 开头（允许前面有括号）
    2. 后续维度不是 1（避免误伤全 1 的 shape）
    3. shape 长度 >= 2（至少 2D）
    """
    shape_str = shape_str.strip()
    # 移除开头的括号
    if shape_str.startswith("(") or shape_str.startswith("["):
        shape_str = shape_str[1:]

    if not shape_str.startswith("1,"):
        return False

    # 解析 shape
    try:
        # 移除剩余的括号，分割
        shape_str = shape_str.strip("()[]")
        parts = [p.strip() for p in shape_str.split(",")]

        # 至少 2 维
        if len(parts) < 2:
            return False

        # 第二维不能是 -1 或 1（避免 [1, -1] 这种推断形状）
        if parts[1] in ["-1", "1"]:
            # 但如果第三维及以上有非 1 值，仍可能是 batch
            if any(p not in ["1", "-1"] for p in parts[2:]):
                return True
            return False

        return True

    except Exception:
        return False


def _replace_batch_in_reshape_view(
    line: str, op: str, tensor_name: str
) -> Tuple[str, bool, str]:
    """
    替换 reshape/view 中的 batch 硬编码

    原理：
    - reshape(1, 576, 1536) 中的 1 改成 tensor.size(0)
    - 需要知道操作前的 tensor 名

    返回: (修改后行, 是否修改, 原因)
    """
    # 找到操作调用
    pattern = rf"\.({op})\s*\(([^)]+)\)"
    match = re.search(pattern, line)
    if not match:
        return line, False, ""

    args_str = match.group(2)
    args = [a.strip() for a in args_str.split(",")]

    # 检查第一个参数是否是 1
    if len(args) < 2 or args[0] != "1":
        return line, False, ""

    # 检查是否是明显的 batch 硬编码
    shape_str = "(" + ", ".join(args) + ")"
    if not _is_likely_batch_hardcoded(shape_str):
        return line, False, ""

    # 如果有 tensor 名，替换第一个 1 为 size(0)
    if tensor_name and re.match(r"^[a-zA-Z_]\w*$", tensor_name):
        new_args = [f"{tensor_name}.size(0)"] + args[1:]
        new_args_str = ", ".join(new_args)
        new_line = re.sub(rf"\.{op}\s*\([^)]+\)", f".{op}({new_args_str})", line)
        return (
            new_line,
            True,
            f"Replace hardcoded batch=1 in {op} with {tensor_name}.size(0)",
        )

    return line, False, ""


def _replace_batch_in_expand(
    line: str, op: str, tensor_name: str
) -> Tuple[str, bool, str]:
    """
    替换 expand 中的 batch 硬编码

    原理：
    - expand(1, -1) 中的 1 改成 tensor.size(0)
    - expand(1, 514) 中的 1 改成 tensor.size(0)
    """
    pattern = rf"\.({op})\s*\(([^)]+)\)"
    match = re.search(pattern, line)
    if not match:
        return line, False, ""

    args_str = match.group(2)
    args = [a.strip() for a in args_str.split(",")]

    # 检查第一个参数是否是 1
    if len(args) < 1 or args[0] != "1":
        return line, False, ""

    # 如果有 tensor 名，替换第一个 1
    if tensor_name and re.match(r"^[a-zA-Z_]\w*$", tensor_name):
        new_args = [f"{tensor_name}.size(0)"] + args[1:]
        new_args_str = ", ".join(new_args)
        new_line = re.sub(rf"\.{op}\s*\([^)]+\)", f".{op}({new_args_str})", line)
        return (
            new_line,
            True,
            f"Replace hardcoded batch=1 in {op} with {tensor_name}.size(0)",
        )

    return line, False, ""


def _replace_batch_in_repeat(line: str, tensor_name: str) -> Tuple[str, bool, str]:
    """替换 repeat 中的 batch 硬编码"""
    return _replace_batch_in_expand(line, "repeat", tensor_name)


def preprocess_model_py_for_batch(
    model_py_path: str, output_path: str = None, verbose: bool = True
) -> Tuple[str, List[Replacement]]:
    """
    预处理 model.py，替换明显的 batch 硬编码

    Args:
        model_py_path: 原始 model.py 路径
        output_path: 输出路径，如果为 None 则覆盖原文件
        verbose: 是否打印替换详情

    Returns:
        (处理后内容, 替换列表)
    """
    with open(model_py_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    replacements = []

    for i, line in enumerate(lines):
        line_num = i + 1
        modified = False
        original_line = line

        # 找到当前行操作的 tensor 名
        prev_lines = new_lines if new_lines else lines[:i]
        tensor_name = _find_tensor_name_for_line(line, prev_lines)

        # 尝试各种操作
        # 1. reshape
        if ".reshape(" in line:
            new_line, changed, reason = _replace_batch_in_reshape_view(
                line, "reshape", tensor_name
            )
            if changed:
                line = new_line
                modified = True
                replacements.append(
                    Replacement(line_num, original_line.strip(), line.strip(), reason)
                )

        # 2. view
        if not modified and ".view(" in line:
            new_line, changed, reason = _replace_batch_in_reshape_view(
                line, "view", tensor_name
            )
            if changed:
                line = new_line
                modified = True
                replacements.append(
                    Replacement(line_num, original_line.strip(), line.strip(), reason)
                )

        # 3. expand
        if not modified and ".expand(" in line:
            new_line, changed, reason = _replace_batch_in_expand(
                line, "expand", tensor_name
            )
            if changed:
                line = new_line
                modified = True
                replacements.append(
                    Replacement(line_num, original_line.strip(), line.strip(), reason)
                )

        # 4. repeat
        if not modified and ".repeat(" in line:
            new_line, changed, reason = _replace_batch_in_repeat(line, tensor_name)
            if changed:
                line = new_line
                modified = True
                replacements.append(
                    Replacement(line_num, original_line.strip(), line.strip(), reason)
                )

        new_lines.append(line)

    result = "".join(new_lines)

    if verbose and replacements:
        print(
            f"  [BATCH PREPROCESS] Replaced {len(replacements)} hardcoded batch dimensions"
        )
        for r in replacements[:10]:  # 只显示前 10 个
            print(f"    L{r.line_num}: {r.reason}")
        if len(replacements) > 10:
            print(f"    ... and {len(replacements) - 10} more")

    # 写入文件
    if output_path is None:
        output_path = model_py_path

    with open(output_path, "w") as f:
        f.write(result)

    return result, replacements


if __name__ == "__main__":
    # 测试
    import sys

    if len(sys.argv) < 2:
        print("Usage: python batch_hardcode_preprocess.py <model.py> [output.py]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path + ".preprocessed"

    content, replacements = preprocess_model_py_for_batch(
        input_path, output_path, verbose=True
    )
    print(f"\nOutput written to: {output_path}")
    print(f"Total replacements: {len(replacements)}")
