import os
import json
import subprocess
import re
from pathlib import Path


def get_model_path_list(model_path_list):
    # Get a list of model path from args.
    assert model_path_list is not None
    with open(model_path_list) as f:
        yield from (
            clean_line
            for line in f
            for clean_line in [line.strip()]
            if len(clean_line) > 0
            if not clean_line.startswith("#")
        )


def execute_paconvert(command):
    # Run padconvet to get log from torch to paddle.
    try:
        subprocess.run(command, check=True)
        print("Convert successfully")
    except subprocess.CalledProcessError as e:
        print(f"Convert failed: {e}")
    except FileNotFoundError:
        print("Error: File not be found.")


def get_convert_log(model_path, log_dir):
    # Get paconvert log of sample.
    input_file = os.path.join(model_path, "model.py")
    log_file = os.path.join(log_dir, "conversion.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "paconvert",
        "-i",
        input_file,
        "--log_dir",
        log_file,
        "--show_unsupport_api",
    ]

    execute_paconvert(cmd)


def filter_unconverted_api(log_file):
    # Filter unconvered api from log.
    unconverted_api = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("torch."):
                    name = line.split()[0]
                    unconverted_api.append((name))
    except FileNotFoundError:
        print(f"Not found: {log_file}")

    return unconverted_api


def save_unconverted_api(rel_model_path, unconverted_api, summary_file):
    # Save result of sample to summary.json.
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    if not summary_file.exists():
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({}, f)

    with open(summary_file, "r", encoding="utf-8") as json_f:
        all_data = json.load(json_f)

    all_data[rel_model_path] = {"api_unsupported_list": unconverted_api}

    with open(summary_file, "w", encoding="utf-8") as json_f:
        json.dump(all_data, json_f, indent=4, ensure_ascii=False)


def filter_and_save_unconverted_api(rel_model_path, log_dir, summary_dir):
    # Get unconverted api of sample.
    log_file = Path(os.path.join(log_dir, "conversion.log"))
    summary_file = Path(os.path.join(summary_dir, "summary.json"))

    unconverted_api = filter_unconverted_api(log_file)
    save_unconverted_api(rel_model_path, unconverted_api, summary_file)


def get_api_convert_rate(log_file):
    # Get api convert rate of sample.
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(r"Convert Rate is:\s*(\d+\.?\d*)%", line)
                if match:
                    rate = match.group(1)
                    return rate

    except FileNotFoundError:
        print(f"Not found: {log_file}")


def save_api_convert_rate(rel_model_path, summary_file, api_convert_rate):
    # Save api convert rate to summary.json.
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, "r", encoding="utf-8") as json_f:
        all_data = json.load(json_f)

    all_data[rel_model_path]["api_convert_rate"] = api_convert_rate

    with open(summary_file, "w", encoding="utf-8") as json_f:
        json.dump(all_data, json_f, indent=4, ensure_ascii=False)


def save_sample_api_convert_rate(rel_model_path, log_dir, summary_dir):
    # Save api convet rate of sample.
    summary_file = Path(os.path.join(summary_dir, "summary.json"))
    log_file = Path(os.path.join(log_dir, "conversion.log"))
    api_convert_rate = get_api_convert_rate(log_file)
    save_api_convert_rate(rel_model_path, summary_file, api_convert_rate)
