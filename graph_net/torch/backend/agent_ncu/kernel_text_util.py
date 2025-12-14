import os
import re
import glob
import subprocess

import textwrap
from typing import Tuple
from unittest.mock import patch

from graph_net.torch.backend.agent_template.cuda_eval_template import (
    TEST_CUDA_CODE_TEMPLATE,
    RUN_NCU_TEMPLATE,
    NCU_BASH_SCRIPT_TEMPLATE,
    NCU_EXTRACT_PY_CONTENT,
)


def _write_file(path: str, content: str | bytes, mode: str = "w"):
    with open(path, mode) as f:
        f.write(content)


def remove_pybind_module(cuda_code: str) -> str:
    pattern = re.compile(r"PYBIND11_MODULE\s*\([^)]*\)\s*\{[^{}]*\}", re.DOTALL)
    cleaned_code = re.sub(pattern, "", cuda_code)
    return cleaned_code.strip()


def extract_cuda_code(text: str):
    codeblock_seps = ["python", ""]
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    codeblock_start = f"```(?:{languages_pattern})"
    pattern = re.compile(
        codeblock_start + r"\s*\n(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL | re.IGNORECASE
    )

    matches = list(pattern.finditer(text))
    if matches:
        last_match = matches[-1]
        code_content = last_match.group(1).rstrip()
        return code_content
    return text


def save_log_info(work_dir: str, filename: str, content: str):
    with open(os.path.join(work_dir, filename), "a") as fout:
        fout.write(content)


def format_with_kernelbench_style(gm, model_inputs) -> str:
    """CodeTemplate: Format the given GraphModule into KernelBench input style."""

    if gm is None or not model_inputs:
        raise ValueError("GraphModule and model_inputs must be provided.")

    # fetch raw forward code
    raw_forward_code = gm.code
    raw_forward_code = raw_forward_code.replace("\n\n\ndef forward", "def forward")

    # fetch input shapes, format like: 'xi = torch.randn(<shape>)'
    return_line = ""
    tensor_lines = []
    input_shapes = (
        [inp.shape for inp in model_inputs] if model_inputs is not None else []
    )
    for i, shape in enumerate(input_shapes):
        tmp_varname = f"x{i+1}"
        tmp_tensor_line = f"    {tmp_varname}=torch.randn{tuple(shape)}\n"
        tensor_lines.append(tmp_tensor_line)
        return_line += f"{tmp_varname}, "
    return_line = f"    return [{return_line[:-2]}]"

    torch_model_code = []
    torch_model_code.append("import torch")
    torch_model_code.append("import torch.nn as nn")
    torch_model_code.append("")
    torch_model_code.append("class Model(nn.Module):")
    torch_model_code.append("    def __init__(self): # no args")
    torch_model_code.append("        super().__init__()")
    torch_model_code.append("")
    torch_model_code.append(textwrap.indent(raw_forward_code, "    "))
    torch_model_code.append("")
    torch_model_code.append("def get_init_inputs():")
    torch_model_code.append("    return []")
    torch_model_code.append("")
    torch_model_code.append("def get_inputs():")
    torch_model_code.extend(tensor_lines)
    torch_model_code.append(return_line)

    return "\n".join(torch_model_code)


def compile_kernel(cuda_code: str = None, work_dir: str = None) -> bool:
    is_compilable = False
    compile_info = {
        "exec_filename": None,
        "exec_content": None,
        "msg": None,
    }

    # save model_new.py before compile
    with open(os.path.join(work_dir, "model_new.py"), "w") as fout:
        fout.write(cuda_code)

    try:
        test_compile_cmd = "python model_new.py"
        with patch.dict(
            os.environ,
            {
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "TORCH_EXTENSIONS_DIR": "build",
                "MAX_JOBS": "64",
            },
        ):
            compile_result = subprocess.run(
                test_compile_cmd,
                timeout=180,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                cwd=work_dir,
            )
        compile_log = compile_result.stdout.decode()
        so_files = glob.glob(f"{work_dir}/build/**/*.so", recursive=True)
        if not len(so_files) == 1:
            print(f"Expected 1 .so file, found {len(so_files)}: {so_files}")
        with open(so_files[0], "rb") as fin:
            bin_content = fin.read()
        compile_info["exec_filename"] = os.path.basename(so_files[0])
        compile_info["exec_content"] = bin_content
        compile_info["msg"] = "[Compile] state: success"
        is_compilable = True
    except subprocess.TimeoutExpired:
        compile_info["msg"] = "[Compile] state: failed\n"
        compile_info["msg"] += "[Failed due to timeout in compilation] "
    except Exception as e:
        compile_info["msg"] = "[Compile] state: failed\n"
        compile_info[
            "msg"
        ] += f"[Failed due to compilation error: {e}, log: {compile_log}]"
    finally:
        return is_compilable, compile_info


def exec_eval_cuda(
    exec_filename: str, exec_content: bytes, pytorch_module: str, work_dir: str
) -> Tuple[bool, str]:
    # save files first
    with open(os.path.join(work_dir, exec_filename), "wb") as fout:
        fout.write(exec_content)
    with open(os.path.join(work_dir, "model.py"), "w") as fout:
        fout.write(pytorch_module)
    with open(os.path.join(work_dir, "eval.py"), "w") as fout:
        fout.write(TEST_CUDA_CODE_TEMPLATE)

    test_log = ""
    try:
        test_cmd = "python eval.py"
        test_result = subprocess.run(
            test_cmd,
            timeout=60,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            cwd=work_dir,
        )
        test_log = test_result.stdout.decode()
        if test_result.returncode != 0:
            return False, f"failed: test error: [{test_log}]"

    except subprocess.TimeoutExpired:
        return False, "failed: test timed out"
    except Exception as e:
        return False, f"failed: test error: [{e}] log: [{test_log}]"

    with open(os.path.join(work_dir, "log.log"), "a") as fout:
        fout.write(test_log)

    return ("[Performance] [speedup]" in test_log), test_log


def exec_eval_cuda_with_ncu(
    ext_filename: str, ext_content: bytes, work_dir: str
) -> Tuple[bool, str]:
    # prepare workspace files
    _write_file(os.path.join(work_dir, ext_filename), ext_content, "wb")
    _write_file(os.path.join(work_dir, "test.py"), RUN_NCU_TEMPLATE)
    _write_file(os.path.join(work_dir, "run.sh"), NCU_BASH_SCRIPT_TEMPLATE)
    _write_file(os.path.join(work_dir, "extract.py"), NCU_EXTRACT_PY_CONTENT)

    # run ncu script
    test_log = ""
    try:
        test_cmd = "bash run.sh"
        test_result = subprocess.run(
            test_cmd,
            timeout=60,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            cwd=work_dir,
        )
        test_log = test_result.stdout.decode()
    except subprocess.TimeoutExpired:
        return False, "failed: test timed out"
    except Exception as e:
        return False, f"failed: test error: [{e}] log: [{test_log}]"

    with open(os.path.join(work_dir, "log.log"), "a") as fout:
        fout.write(test_log)

    return True, test_log
