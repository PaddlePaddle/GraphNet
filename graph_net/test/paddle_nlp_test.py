import os
import sys
import subprocess
import shutil
import traceback
import time
import glob
import logging

os.environ["ENABLE_CINN_IN_DY2ST"] = "0"
os.environ["FLAGS_logging_trunc_pir_py_code"] = "1"
os.environ["FLAGS_logging_pir_py_code_int_tensor_element_limit"] = "64"
os.environ["FLAGS_logging_pir_py_code_dir"] = "/tmp/dump"

import paddle
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger()


def run_nlp_model(model_name: str, dump_path: str) -> None:
    device = (
        "cuda:0"
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        else "cpu"
    )
    print(f"\nTesting NLP model: {model_name} on {device}")

    if not os.path.exists(dump_path):
        os.makedirs(dump_path, exist_ok=True)

    paddle.set_flags(
        {
            "FLAGS_logging_trunc_pir_py_code": 1,
            "FLAGS_logging_pir_py_code_int_tensor_element_limit": 64,
            "FLAGS_logging_pir_py_code_dir": dump_path,
        }
    )

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config, dtype="float16")
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    text = "Hello, my name is Bob. I am learning about large language models and their architectures. "
    inputs = tokenizer(
        text, return_tensors="pd", padding=True, truncation=True, max_length=512
    )
    input_data = {key: val.to(device) for key, val in inputs.items()}
    input_data["use_cache"] = False

    # model(**input_data)

    input_specs = []
    for name, value in input_data.items():
        if isinstance(value, paddle.Tensor):
            input_specs.append(
                paddle.static.InputSpec(value.shape, value.dtype, name=name)
            )

    static_model = paddle.jit.to_static(model, input_spec=input_specs, full_graph=True)
    static_model(**input_data)


def process_model(model_name, dump_dir):
    logger.info(f"Starting processing for: {model_name}")

    # Step 1: dump pir program
    dump_path = os.path.join(dump_dir, model_name)
    run_nlp_model(model_name, dump_path)
    logger.info(f"Dump {model_name} to {dump_path}")

    # Step 2: extract GraphNet sample
    workspace = os.getenv("GRAPH_NET_EXTRACT_WORKSPACE", "./workspace")
    output_dir = os.path.join(workspace, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    generate_sample_cmd = [
        "python",
        "-m",
        "athena.module_op_unittests_for_graphnet",
        f"--model_name={model_name}",
        f"--ir_programs={dump_path}/exec_programs.py",
        f"--example_inputs={dump_path}/programs_example_input_tensor_meta.py",
        f"--output_dir={output_dir}",
        "--max_depth_output_only=True",
    ]

    result = subprocess.run(
        generate_sample_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
    )
    if result.returncode == 0:
        logger.info(f"Generate samples for {model_name} to {dump_path}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dir = os.path.join(current_dir, "dump")

    model_name = "facebook/llama-7b"
    process_model(model_name, dump_dir)


if __name__ == "__main__":
    main()
