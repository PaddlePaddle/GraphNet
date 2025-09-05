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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger()


def get_auto_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        text, return_tensors="pd", padding=True, truncation=True, max_length=512
    )
    return model, inputs


def get_gpt_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import GPTModel, GPTTokenizer

    model = GPTModel.from_pretrained(model_name)
    model.eval()

    tokenizer = GPTTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_ernie_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import ErnieModel, ErnieTokenizer

    model = ErnieModel.from_pretrained(model_name)
    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def get_bert_model_and_inputs(model_name, text, dtype):
    from paddlenlp.transformers import BertModel, BertTokenizer

    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pd")
    return model, inputs


def run_nlp_model(model_name, get_model_func, dump_path, text):
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

    model, inputs = get_model_func(model_name, text, dtype="float16")
    input_dict = {key: val.to(device) for key, val in inputs.items()}
    # input_dict["use_cache"] = False
    print(input_dict)

    model(**input_dict)

    input_specs = []
    for name, value in input_dict.items():
        if isinstance(value, paddle.Tensor):
            input_specs.append(
                paddle.static.InputSpec(value.shape, value.dtype, name=name)
            )

    static_model = paddle.jit.to_static(model, input_spec=input_specs, full_graph=True)
    static_model(**input_dict)


def process_model(model_name, get_gpt_model_and_inputs, dump_dir, text):
    logger.info(f"Starting processing for: {model_name}")

    # Step 1: dump pir program
    dump_path = os.path.join(dump_dir, model_name)
    run_nlp_model(model_name, get_gpt_model_and_inputs, dump_path, text)
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

    text_en = "Hello, my name is Bob. I am learning about large language models and their architectures. "
    text_cn = "欢迎使用百度飞桨!"

    # failed models: xlnet
    # success models: as follows

    # auto models
    model_name = "facebook/llama-7b"
    # process_model(model_name, get_auto_model_and_inputs, dump_dir, text_en)

    # gpt models
    model_name = "gpt2-medium-en"
    # process_model(model_name, get_gpt_model_and_inputs, dump_dir, text_en)

    # ernie models
    model_name = "ernie-1.0-base-zh"
    model_name = "ernie-1.0-base-zh-cw"
    model_name = "ernie-1.0-large-zh-cw"
    # more ..
    # process_model(model_name, get_ernie_model_and_inputs, dump_dir, text_cn)
    model_name = "ernie-1.0"
    # process_model(model_name, get_ernie_model_and_inputs, dump_dir, text_en)

    # bert models
    model_name = "bert-base-cased"
    process_model(model_name, get_bert_model_and_inputs, dump_dir, text_en)


if __name__ == "__main__":
    main()
