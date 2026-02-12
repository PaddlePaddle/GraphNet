import argparse
import subprocess
import os
import re
import json
import shutil
from pathlib import Path


def _get_model_path_list(args):
    # Get a list of model path from args.
    assert args.model_path_list is not None
    with open(args.model_path_list) as f:
        yield from (
            clean_line
            for line in f
            for clean_line in [line.strip()]
            if len(clean_line) > 0
            if not clean_line.startswith("#")
        )

def remove_string_from_model(input_file, target_string):
    # Delete a fixed string from model.py.
    if not os.path.exists(input_file):
        print(f"Error: Not found {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if target_string not in content:
            return

        new_content = content.replace(target_string, "")

        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

    except Exception as e:
        print(f"Error: {e}")


def run_paconvert(input_file, output_file, output_log):
    # Run padconvet to convert model.py from torch to paddle.
    Path(output_log).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "paconvert",
        "-i", input_file,
        "-o", output_file,
        "--log_dir", output_log, 
        "--show_unsupport_api"
    ]

    try:
        subprocess.run(command, check=True)
        print("Convert successfully")
    except subprocess.CalledProcessError as e:
        print(f"Convert failed: {e}")
    except FileNotFoundError:
        print("Error: The paconvert command could not be found. Please ensure that the tool is installed.")
    

def convert_model_py(model_path, output_dir): 
    # Convert model.py from torch to paddle.
    input_model_py = os.path.join(model_path, "model.py")
    output_model_py = os.path.join(output_dir, "model.py")
    output_log = os.path.join(output_dir, "log.log")
    run_paconvert(input_model_py, output_model_py, output_log)
    remove_string_from_model(output_model_py, ">>>>>>")

def convert_weight_meta_py(model_path, output_dir):
    # Convert weight_meta.py from torch to paddle.
    input_file = os.path.join(model_path, 'weight_meta.py')
    output_file = os.path.join(output_dir, 'weight_meta.py')
    
    if not os.path.exists(input_file):
        print(f"[Error] Not found: {input_file}")
        return
    
    pattern = r"(dtype\s*=\s*['\"])torch(?=.*['\"])"
    replacement = r"\1paddle"

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = re.sub(pattern, replacement, content)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

    except Exception as e:
        print(f"Error: {e}")


def convert_graph_net_json(model_path, output_dir):
    # Convert graph_net.json from torch to paddle.
    input_file = os.path.join(model_path, 'graph_net.json')
    output_file = os.path.join(output_dir, 'graph_net.json')

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data.get("framework") == "torch":
        data["framework"] = "paddle"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    

def copy_sample_files(model_path, output_dir, files_copied):
    # Copy files of sample.
    for fname in files_copied:
        input_file = os.path.join(model_path, fname)
        output_file = os.path.join(output_dir, fname)
        shutil.copy(input_file, output_file)


def convert_sample_from_torch_to_paddle(model_path, output_dir):
    # Convert a sample from torch to paddle.
    files_copied = ["input_meta.py", "input_tensor_constraints.py", "graph_hash.txt"]
    convert_model_py(model_path, output_dir)
    convert_weight_meta_py(model_path, output_dir)
    convert_graph_net_json(model_path, output_dir)
    copy_sample_files(model_path, output_dir, files_copied)

def get_api_convert_rate(log_path):
    # Get api convert rate of sample.
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'Convert Rate is:\s*(\d+\.?\d*)%', line)
                if match:
                    rate = match.group(1) 
                    return rate
                
    except FileNotFoundError:
        print(f"Not found: {log_path}")

def get_api_unsupported(log_path):
    # Get a list of api unsupported.
    api_unsupported_list = []
    try: 
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("torch."):
                    parts = line.split() 
                    if len(parts) >= 2:
                        name = parts[0]
                        api_unsupported_list.append((name))

    except FileNotFoundError:
        print(f"Not found: {log_path}")

    return api_unsupported_list

def save_result_to_json(rel_model_path, result, result_file):
    # Save result of sample to result.json.
    all_data = {}
    try:
        with open(result_file, 'r', encoding='utf-8') as json_f:
            all_data = json.load(json_f)

    except (json.JSONDecodeError, ValueError):
        all_data = {}

    all_data[rel_model_path] = {
        "api_convert_rate": result[0],
        "api_unsupported_list": result[1] 
    }

    with open(result_file, 'w', encoding='utf-8') as json_f:
        json.dump(all_data, json_f, indent=4, ensure_ascii=False)
    

def convert_log_process(rel_model_path, output_dir, result_file):
    # Get api convert rate and api unsupported from log.
    log_path = os.path.join(output_dir, "log.log")
    api_convert_rate = get_api_convert_rate(log_path)
    api_unsupported_list = get_api_unsupported(log_path)
    result = [api_convert_rate, api_unsupported_list]

    save_result_to_json(rel_model_path, result, result_file)


def main(args):
    # Convert samples from torch to paddle.
    model_path_prefix = args.model_path_prefix
    model_path_list = list(_get_model_path_list(args))
    output_dir = args.output_dir
    result_file = args.result_file

    for model_path in model_path_list:
        abs_model_path = os.path.join(model_path_prefix, model_path)
        abs_output_dir = os.path.join(output_dir, model_path.split("samples/", 1)[-1])
        convert_sample_from_torch_to_paddle(abs_model_path, abs_output_dir)
        convert_log_process(model_path, abs_output_dir, result_file)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compiler performance.")
    parser.add_argument(
        "--model-path-list",
        type=str,
        required=False,
        default=None,
        help="Path of file containing model paths.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=None,
        help="Output directory of samples from torch to paddle.",
    )

    parser.add_argument(
        "--model-path-prefix",
        type=str,
        required=False,
        default=None,
        help="Path prefix of samples in list of model path.",
    )

    parser.add_argument(
        "--result-file",
        type=str,
        required=False,
        default=None,
        help="Result of convert samples from torch to paddle.",
    )

    args = parser.parse_args()
    main(args=args)
