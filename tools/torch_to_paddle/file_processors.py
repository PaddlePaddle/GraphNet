import os
import re
import json
import shutil
import torch
import gc

from pathlib import Path
from graph_net.torch import utils
from torch.fx.passes.shape_prop import ShapeProp
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module
from graph_net.torch.fx_graph_serialize_util import serialize_graph_module_to_str
from tools.torch_to_paddle.utils import save_unconverted_api, execute_paconvert
from graph_net_bench.torch.backend.unstable_to_stable_backend import (
    UnstableToStableBackend,
)


API_UNSTABLE_TO_STABLE = {
    "torch._C._nn.one_hot": "_impl_unstable_to_stable_one_hot",
    "torch._C._linalg.linalg_vector_norm": "_impl_unstable_to_stable_linalg_vector_norm",
    "torch._C._fft.fft_irfft": "_impl_unstable_to_stable_irfft",
    "torch._C._special.special_logit": "_impl_unstable_to_stable_special_logit",
    "torch._C._fft.fft_rfft": "_impl_unstable_to_stable_rfft",
    "torch._C._nn.pad": "_impl_unstable_to_stable_pad",
    "torch._C._nn.gelu": "_impl_unstable_to_stable_gelu",
    "torch._C._nn.softplus": "_impl_unstable_to_stable_softplus",
    "torch._C._nn.scaled_dot_product_attention": "_impl_unstable_to_stable_sdpa",
    "torch._C._linalg.linalg_norm": "_impl_unstable_to_stable_linalg_norm",
    "torch._C._nn.linear": "_impl_unstable_to_stable_linear_to_functional_linear",
    "torch._C._set_grad_enabled": "_impl_unstable_to_stable_set_grad_enabled",
    "torch._C._nn.avg_pool2d": "_impl_unstable_to_stable_avg_pool2d",
    "torch._C._fft.fft_fftn": "_impl_unstable_to_stable_fftn",
}


def get_gm_from_model_path(model_path):
    # Parse the computation graph
    module, inputs = get_torch_module_and_inputs(model_path)
    model = parse_sole_graph_module(module, inputs)

    with torch.no_grad():
        ShapeProp(model).propagate(*inputs)
    return model


def read_unconverted_api(rel_model_path, summary_file):
    # Read unconverted api of sample.
    try:
        with open(summary_file, "r", encoding="utf-8") as json_f:
            all_data = json.load(json_f)
    except FileNotFoundError:
        print(f"Not found: {summary_file}")

    return all_data[rel_model_path]["api_unsupported_list"]


def gm_unstable_to_stable(gm, unconverted_api):
    # Convert api of gm from unstable to stable.
    converter = UnstableToStableBackend({})
    new_unconverted_api = []
    for api in unconverted_api:
        if api in API_UNSTABLE_TO_STABLE:
            converter_method_name = API_UNSTABLE_TO_STABLE[api]
            converter_method = getattr(converter, converter_method_name)
            gm = converter_method(gm)
        else:
            new_unconverted_api.append(api)
    return gm, new_unconverted_api


def save_gm_to_model_py(gm, output_dir):
    # Save new gm to model.py.
    model_code = serialize_graph_module_to_str(gm)
    write_code = utils.apply_templates(model_code)

    output_model_py = os.path.join(output_dir, "model_unstable_to_stable.py")
    Path(output_model_py).parent.mkdir(parents=True, exist_ok=True)
    with open(output_model_py, "w") as f:
        f.write(write_code)


def convert_api_from_unstable_to_stable(
    rel_model_path, model_path, output_dir, summary_dir
):
    # Convert model.py from unstable to stable.
    summary_file = Path(os.path.join(summary_dir, "summary.json"))
    unconverted_api = read_unconverted_api(rel_model_path, summary_file)
    gm = get_gm_from_model_path(model_path)
    gm_modified, new_unconverted_api = gm_unstable_to_stable(gm, unconverted_api)
    save_gm_to_model_py(gm_modified, output_dir)

    del gm
    del gm_modified
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_unconverted_api(rel_model_path, new_unconverted_api, summary_file)


def remove_string_from_model(input_file, target_string):
    # Delete a fixed string from model.py.
    if not os.path.exists(input_file):
        print(f"Error: Not found {input_file}")
        return

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        if target_string not in content:
            return

        new_content = content.replace(target_string, "")

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(new_content)

    except Exception as e:
        print(f"Error: {e}")


def convert_model_py(model_path, output_dir, log_dir):
    # Convert model.py from torch to paddle.
    input_model_py = Path(os.path.join(model_path, "model_unstable_to_stable.py"))
    output_model_py = Path(os.path.join(output_dir, "model.py"))
    output_log = os.path.join(log_dir, "conversion.log")
    Path(output_log).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "paconvert",
        "-i",
        input_model_py,
        "-o",
        output_model_py,
        "--log_dir",
        output_log,
        "--show_unsupport_api",
    ]
    execute_paconvert(cmd)
    remove_string_from_model(output_model_py, ">>>>>>")


def convert_weight_meta_py(model_path, output_dir):
    # Convert weight_meta.py from torch to paddle.
    input_file = os.path.join(model_path, "weight_meta.py")
    output_file = os.path.join(output_dir, "weight_meta.py")

    if not os.path.exists(input_file):
        print(f"[Error] Not found: {input_file}")
        return

    pattern = r"(dtype\s*=\s*['\"])torch(?=.*['\"])"
    replacement = r"\1paddle"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = re.sub(pattern, replacement, content)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(new_content)

    except Exception as e:
        print(f"Error: {e}")


def convert_graph_net_json(model_path, output_dir):
    # Convert graph_net.json from torch to paddle.
    input_file = os.path.join(model_path, "graph_net.json")
    output_file = os.path.join(output_dir, "graph_net.json")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("framework") == "torch":
        data["framework"] = "paddle"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def copy_sample_files(model_path, output_dir, files_copied):
    # Copy files of sample.
    for fname in files_copied:
        input_file = os.path.join(model_path, fname)
        output_file = os.path.join(output_dir, fname)
        shutil.copy(input_file, output_file)


def convert_other_files(model_path, output_dir):
    # Convert other files.
    files_copied = ["input_meta.py", "input_tensor_constraints.py", "graph_hash.txt"]
    copy_sample_files(model_path, output_dir, files_copied)


def convert_sample_from_torch_to_paddle(model_path, output_dir, log_dir):
    # Convert a sample from torch to paddle.
    convert_model_py(output_dir, output_dir, log_dir)
    convert_weight_meta_py(model_path, output_dir)
    convert_graph_net_json(model_path, output_dir)
    convert_other_files(model_path, output_dir)
