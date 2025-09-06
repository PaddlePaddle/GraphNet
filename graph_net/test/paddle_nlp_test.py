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
import nlp_model_getter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger()


def clear_transformers_cache():
    paddlenlp_models_cache = "/root/.paddlenlp/models"
    logger.info(f"Cleaning paddlenlp cache: {paddlenlp_models_cache}")

    if not os.path.exists(paddlenlp_models_cache):
        return

    for item in os.listdir(paddlenlp_models_cache):
        item_path = os.path.join(paddlenlp_models_cache, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            logger.error(f"Failed to delete {item_path}. Reason: {e}")

    logger.info("PaddleNLP cache cleared successfully")


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

    clear_transformers_cache()


def process_model(model_name, get_gpt_model_and_inputs, dump_dir, text):
    logger.info(f"Starting processing for: {model_name}")

    # Step 1: dump pir program
    dump_path = os.path.join(dump_dir, model_name.replace("/", "_"))
    run_nlp_model(model_name, get_gpt_model_and_inputs, dump_path, text)
    logger.info(f"Dump {model_name} to {dump_path}")

    # Step 2: extract GraphNet sample
    workspace = os.getenv("GRAPH_NET_EXTRACT_WORKSPACE", "./workspace")
    output_dir = os.path.join(workspace, model_name.replace("/", "_"))
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

    # auto models
    model_name = "facebook/llama-7b"
    process_model(
        model_name, nlp_model_getter.get_auto_model_and_inputs, dump_dir, text_en
    )

    # bert models
    bert_model_dict = {
        "bert-base-cased": text_en,
        "bert-base-chinese": text_cn,
        "bert-base-multilingual-cased": text_en,
        "bert-base-multilingual-uncased": text_en,
        "bert-base-uncased": text_en,
        "bert-large-cased": text_en,
        "bert-large-uncased": text_en,
        # "bert-wwm-chinese": text_cn, # no params
        "bert-wwm-ext-chinese": text_cn,
        # "macbert-base-chinese": text_cn, # no params
        # "macbert-large-chinese": text_cn, # no params
        # "simbert-base-chinese": text_cn, # no params
        "uer/chinese-roberta-6l-768h": text_cn,
        "uer/chinese-roberta-base": text_cn,
        "uer/chinese-roberta-medium": text_cn,
        "uer/chinese-roberta-mini": text_cn,
        "uer/chinese-roberta-small": text_cn,
        "uer/chinese-roberta-tiny": text_cn,
    }
    # for model_name, text in bert_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_bert_model_and_inputs, dump_dir, text)

    # convbert models
    convbert_model_dict = {
        "convbert-base": text_en,
        "convbert-medium-small": text_en,
        "convbert-small": text_en,
    }
    # for model_name, text in convbert_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_convbert_model_and_inputs, dump_dir, text)

    # ernie models
    ernie_1_model_dict = {
        "ernie-1.0-base-zh-cw": text_cn,
        "ernie-1.0-base-zh": text_cn,
        "ernie-1.0-large-zh-cw": text_cn,
        "ernie-1.0": text_en,
    }
    # for model_name, text in ernie_1_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_ernie_model_and_inputs, dump_dir, text)

    ernie_2_model_dict = {
        # "ernie-2.0-base-en-finetuned-squad": text_en, # no params
        # "ernie-2.0-base-en": text_en, # no params
        "ernie-2.0-base-zh": text_cn,
        # "ernie-2.0-large-en": text_en, # no params
        "ernie-2.0-large-zh": text_cn,
    }
    # for model_name, text in ernie_2_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_ernie_model_and_inputs, dump_dir, text)

    ernie_3_model_dict = {
        "ernie-3.0-base-zh": text_cn,
        "ernie-3.0-medium-zh": text_cn,
        "ernie-3.0-micro-zh": text_cn,
        "ernie-3.0-mini-zh": text_cn,
        "ernie-3.0-nano-zh": text_cn,
        "ernie-3.0-tiny-base-v1-zh": text_cn,
        "ernie-3.0-tiny-base-v2-zh": text_cn,
        "ernie-3.0-tiny-medium-v1-zh": text_cn,
        "ernie-3.0-tiny-medium-v2-zh": text_cn,
        "ernie-3.0-tiny-micro-v1-zh": text_cn,
        "ernie-3.0-tiny-micro-v2-zh": text_cn,
        "ernie-3.0-tiny-mini-v1-zh": text_cn,
        "ernie-3.0-tiny-mini-v2-zh": text_cn,
        "ernie-3.0-tiny-nano-v1-zh": text_cn,
        "ernie-3.0-tiny-nano-v2-zh": text_cn,
        "ernie-3.0-tiny-pico-v2-zh": text_cn,
        "ernie-3.0-xbase-zh": text_cn,
    }
    # for model_name, text in ernie_3_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_ernie_model_and_inputs, dump_dir, text)

    ernie_other_mode_dict = {
        "ernie-search-base-dual-encoder-marco-en": text_en,
        "ernie-search-large-cross-encoder-marco-en": text_en,
        "ernie-tiny": text_en,
    }
    # for model_name, text in ernie_other_mode_dict.items():
    #    process_model(model_name, nlp_model_getter.get_ernie_model_and_inputs, dump_dir, text)

    ernie_rocketqa_model_dict = {
        "rocketqa-base-cross-encoder": text_en,
        "rocketqa-medium-cross-encoder": text_en,
        "rocketqa-micro-cross-encoder": text_en,
        "rocketqa-mini-cross-encoder": text_en,
        "rocketqa-nano-cross-encoder": text_en,
        "rocketqav2-en-marco-cross-encoder": text_en,
        "rocketqav2-en-marco-para-encoder": text_en,
        "rocketqav2-en-marco-query-encoder": text_en,
        "rocketqa-zh-base-para-encoder": text_cn,
        "rocketqa-zh-base-query-encoder": text_cn,
        "rocketqa-zh-dureader-cross-encoder": text_cn,
        "rocketqa-zh-dureader-para-encoder": text_cn,
        "rocketqa-zh-dureader-query-encoder": text_cn,
        "rocketqa-zh-medium-para-encoder": text_cn,
        "rocketqa-zh-medium-query-encoder": text_cn,
        "rocketqa-zh-micro-para-encoder": text_cn,
        "rocketqa-zh-micro-query-encoder": text_cn,
        "rocketqa-zh-mini-para-encoder": text_cn,
        "rocketqa-zh-mini-query-encoder": text_cn,
        "rocketqa-zh-nano-para-encoder": text_cn,
        "rocketqa-zh-nano-query-encoder": text_cn,
    }
    # for model_name, text in ernie_rocketqa_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_ernie_model_and_inputs, dump_dir, text)

    ernie_uie_model_dict = {
        "uie-base-answer-extractor": text_en,
        "uie-base-en": text_en,
        "uie-base": text_en,
        "uie-base-qa-filter": text_en,
        "uie-medium": text_en,
        "uie-micro": text_en,
        "uie-mini": text_en,
        "uie-nano": text_en,
        "uie-senta-base": text_en,
        "uie-senta-medium": text_en,
        "uie-senta-micro": text_en,
        "uie-senta-mini": text_en,
        "uie-senta-nano": text_en,
    }
    # for model_name, text in ernie_uie_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_ernie_model_and_inputs, dump_dir, text)

    ernie_utc_model_dict = {
        "utc-base": text_en,
        "utc-large": text_en,
        "utc-medium": text_en,
        "utc-micro": text_en,
        "utc-mini": text_en,
        "utc-nano": text_en,
        "utc-pico": text_en,
        "utc-xbase": text_en,
    }
    # for model_name, text in ernie_utc_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_ernie_model_and_inputs, dump_dir, text)

    # ernie_m models
    ernie_m_model_dict = {
        "ernie-m-base": text_en,
        "ernie-m-large": text_en,
        "uie-m-base": text_en,
        "uie-m-large": text_en,
    }
    # for model_name, text in ernie_m_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_ernie_m_model_and_inputs, dump_dir, text)

    # gpt models
    model_name = "gpt2-medium-en"
    # process_model(model_name, nlp_model_getter.get_gpt_model_and_inputs, dump_dir, text_en)

    # nezha models
    nezha_model_dict = {
        "nezha-base-chinese": text_cn,
        "nezha-base-wwm-chinese": text_cn,
        "nezha-large-chinese": text_cn,
        "nezha-large-wwm-chinese": text_cn,
    }
    # for model_name, text in nezha_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_nezha_model_and_inputs, dump_dir, text)

    # ppminilm models
    model_name = "ppminilm-6l-768h"
    # process_model(model_name, nlp_model_getter.get_ppminilm_model_and_inputs, dump_dir, text_en)

    # reformer models
    roformer_model_dict = {
        "roformer-chinese-base": text_cn,
        "roformer-chinese-char-base": text_cn,
        "roformer-chinese-char-small": text_cn,
        "roformer-chinese-sim-char-base": text_cn,
        "roformer-chinese-sim-char-ft-base": text_cn,
        "roformer-chinese-sim-char-ft-small": text_cn,
        "roformer-chinese-sim-char-small": text_cn,
        "roformer-chinese-small": text_cn,
        "roformer-english-small-discriminator": text_en,
        "roformer-english-small-generator": text_en,
    }
    # for model_name, text in roformer_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_reformer_model_and_inputs, dump_dir, text)

    # skep models
    skep_model_dict = {
        "skep_ernie_2.0_large_en": text_en,
        "skep_ernie_1.0_large_ch": text_cn,
    }
    # for model_name, text in skep_model_dict.items():
    #    process_model(model_name, nlp_model_getter.get_skep_model_and_inputs, dump_dir, text)


if __name__ == "__main__":
    main()
