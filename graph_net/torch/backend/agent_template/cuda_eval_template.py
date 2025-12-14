NCU_METRICS = (
    "sm__cycles_active.avg,"
    "sm__warps_active.avg.pct_of_peak_sustained_active,"
    "launch__occupancy_limit_blocks,"
    "launch__occupancy_limit_registers,"
    "launch__occupancy_limit_shared_mem,"
    "launch__registers_per_thread,"
    "sm__inst_executed.sum,"
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active,"
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,"
    "dram__bytes_read.sum,"
    "dram__bytes_write.sum,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__bytes.sum.per_second,"
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,"
    "l1tex__t_sector_hit_rate.pct,"
    "l1tex__throughput.avg.pct_of_peak_sustained_active,"
    "lts__t_sector_hit_rate.pct,"
    "lts__throughput.avg.pct_of_peak_sustained_active,"
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct,"
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,"
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,"
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct,"
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct,"
    "smsp__sass_average_branch_targets_threads_uniform.pct"
)

NCU_BASH_SCRIPT_TEMPLATE = r"""
METRICS="{NCU_METRICS}"

ncu \\
--metrics $METRICS \\
--target-processes all \\
-o uname_report \\
python test.py 

ncu --import uname_report.ncu-rep --csv > record.txt

python extract.py
"""

NCU_EXTRACT_PY_CONTENT = r"""
import csv

def extract_metrics(csv_path, target_run="4"):
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] != target_run:
                continue
            metric_name = row[12]
            metric_unit = row[13]
            metric_value = row[14]
            results.append((metric_name, metric_unit, metric_value))
    return results

if __name__ == "__main__":
    path = "record.txt" 
    metrics = extract_metrics(path)
    for name, unit, val in metrics:
        print(f"{name}: {val} {unit}")
"""

TEST_CUDA_CODE_TEMPLATE = r'''
import ast
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from contextlib import contextmanager


def rewrite_cuda_model_code(src_path, dst_path):
    """Replace "op = load_inline" with "import op" to separate compilation and execution"""

    model_src = Path(src_path).read_text()
    tree = ast.parse(model_src)

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and isinstance(call := node.value, ast.Call) and \
            ((isinstance(call.func, ast.Attribute) and call.func.attr == 'load_inline') or (isinstance(call.func, ast.Name) and call.func.id == 'load_inline')):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            ext_alias = node.targets[0].id
            for kw in call.keywords:
                if kw.arg == 'name':
                    assert isinstance(kw.value, ast.Constant)
                    ext_name = kw.value.value
                    break
            else:
                raise RuntimeError("Cannot find extension name from model_new.py")
            tree.body[i] = ast.parse(f'import {ext_name} as {ext_alias}').body[0]

    model_src = ast.unparse(tree)
    Path(dst_path).write_text(model_src)

rewrite_cuda_model_code(src_path='model_new.py', dst_path='model_new_patch.py')


from model import Model, get_inputs, get_init_inputs
from model_new_patch import ModelNew

def transform_tensors(tensors, fn):
    if not isinstance(tensors, (list, tuple)):
        return tensors
    outputs = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = fn(tensor)
        elif isinstance(tensor, (list, tuple)):
            tensor = transform_tensors(tensor, fn)
        elif isinstance(tensor, dict):
            tensor = {k:transform_tensors(v, fn) for k, v in tensor.items()}

        outputs.append(tensor)
    return outputs


def check_equal(actual, expected):
    assert isinstance(actual, (list, tuple)) == isinstance(expected, (list, tuple))
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
        expected = [expected]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-2, rtol=1e-2)


@contextmanager
def block_torch_functional(excludes=None):
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith('_') and name not in excludes:
            originals[name] = attr
            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function {F.__name__}.{__name} is not allowed in this context."
                )
            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)

def benchmark_model(model, inputs, num_runs=20):
    for _ in range(3):
        _ = model(*inputs)
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(*inputs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return output, avg_time

init_inputs = get_init_inputs()
if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]
torch_model = Model(*init_inputs).cuda()
cuda_model = ModelNew(*init_inputs).cuda()
cuda_model.load_state_dict(torch_model.state_dict())
torch_inputs = get_inputs()
if not isinstance(torch_inputs, (list, tuple)):
    torch_inputs = [torch_inputs]
torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

torch.cuda.synchronize()

cuda_outputs, cuda_time = benchmark_model(cuda_model,cuda_inputs)

torch.cuda.synchronize()
for _ in range(3):
    torch_outputs = torch_model(*torch_inputs)
torch.cuda.synchronize()

start_time = time.perf_counter()
for _ in range(20):
    torch.cuda.synchronize()  
    torch_outputs = torch_model(*torch_inputs)
    torch.cuda.synchronize()
torch_time =(time.perf_counter() - start_time)/20
check_equal(cuda_outputs, torch_outputs)

print(f"[Performance] [cuda]: {cuda_time * 1000:.2f}ms")
print(f"[Performance] [eager]: {torch_time * 1000:.2f}ms")
print(f"[Performance] [speedup]: {torch_time / cuda_time:.2f}x")
'''

RUN_NCU_TEMPLATE = r'''
import torch
import torch.nn.functional as F
import ast
from pathlib import Path
import sys
from contextlib import contextmanager
import time

def rewrite_cuda_model_code(src_path, dst_path):
    """Replace "op = load_inline" with "import op" to separate compilation and execution"""

    model_src = Path(src_path).read_text()
    tree = ast.parse(model_src)

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and isinstance(call := node.value, ast.Call) and \
            ((isinstance(call.func, ast.Attribute) and call.func.attr == 'load_inline') or (isinstance(call.func, ast.Name) and call.func.id == 'load_inline')):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            ext_alias = node.targets[0].id
            for kw in call.keywords:
                if kw.arg == 'name':
                    assert isinstance(kw.value, ast.Constant)
                    ext_name = kw.value.value
                    break
            else:
                raise RuntimeError("Cannot find extension name from model_new.py")
            tree.body[i] = ast.parse(f'import {ext_name} as {ext_alias}').body[0]

    model_src = ast.unparse(tree)
    Path(dst_path).write_text(model_src)

rewrite_cuda_model_code(src_path='model_new.py', dst_path='model_new_patch.py')


from model import Model, get_inputs, get_init_inputs
from model_new_patch import ModelNew

def transform_tensors(tensors, fn):
    if not isinstance(tensors, (list, tuple)):
        return tensors
    outputs = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = fn(tensor)
        elif isinstance(tensor, (list, tuple)):
            tensor = transform_tensors(tensor, fn)
        elif isinstance(tensor, dict):
            tensor = {k:transform_tensors(v, fn) for k, v in tensor.items()}

        outputs.append(tensor)
    return outputs


def check_equal(actual, expected):
    assert isinstance(actual, (list, tuple)) == isinstance(expected, (list, tuple))
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
        expected = [expected]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-1, rtol=1e-1)


@contextmanager
def block_torch_functional(excludes=None):
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith('_') and name not in excludes:
            originals[name] = attr
            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function {F.__name__}.{__name} is not allowed in this context."
                )
            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


init_inputs = get_init_inputs()
if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]
torch_model = Model(*init_inputs).cuda()
cuda_model = ModelNew(*init_inputs).cuda()
cuda_model.load_state_dict(torch_model.state_dict())

torch_inputs = get_inputs()
if not isinstance(torch_inputs, (list, tuple)):
    torch_inputs = [torch_inputs]
torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

for _ in range(5):
    cuda_outputs = cuda_model(*cuda_inputs)
'''
