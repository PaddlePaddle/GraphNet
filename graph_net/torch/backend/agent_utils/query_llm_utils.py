import time
from typing import Dict, List
from openai import OpenAI
from dataclasses import dataclass


@dataclass
class LLMQueryConfig:
    api_url: str = None
    api_key: str = None

    # common settings
    llm_name: str = None
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096
    parallel_query_nums: int = 1
    iterative_query_nums: int = 10

    dsl: str = "CUDA"

    # cache settings
    # responses will be saved in <tmp_llm_cache>/<llm_name>/
    top_save_dir: str = "./tmp_llm_cache"

    # network settings
    timeout_seconds: float = 600.0
    max_retries: int = 3
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 16.0

    # local Inference settings
    is_local_inference: bool = False
    local_inference_vendor: str = "vllm"  # only for print logs
    local_inference_port: int = 30000
    local_inference_address: str = "localhost"


# TODO conside token budget
# TODO fetch parallel query results
def query_llm_service(
    prompt: str | List[Dict], system_prompt: str, query_config: LLMQueryConfig
):
    """Query LLM service with retries."""

    if query_config.is_local_inference:
        query_config.api_url = (
            f"http://{query_config.local_inference_address}:"
            f"{query_config.local_inference_port}/v1/chat/completions"
        )

    client = OpenAI(
        api_key=query_config.api_key,
        base_url=query_config.api_url,
        timeout=query_config.timeout_seconds,
    )

    # format prompt messages
    query_messages = []
    if isinstance(prompt, list) and prompt and prompt[0].get("role") is not None:
        query_messages = prompt
    elif isinstance(prompt, str):
        if system_prompt:
            query_messages.append({"role": "system", "content": system_prompt})
        query_messages.append({"role": "user", "content": prompt})
    else:
        raise ValueError(
            "Invalid prompt format. Only 'str' or 'list[dict]' are supported."
        )

    # Query with retries
    attempt = 0
    backoff_seconds = query_config.backoff_initial_seconds

    while attempt < query_config.max_retries:
        try:
            response = client.chat.completions.create(
                model=query_config.llm_name,
                messages=query_messages,
                temperature=query_config.temperature,
                top_p=query_config.top_p,
                max_tokens=query_config.max_tokens,
            )
            outputs = [choice.message.content for choice in response.choices]
            token_usage = response.usage if hasattr(response, "usage") else None

            if token_usage:
                token_usage = token_usage.to_dict()

            if len(outputs) == 1:
                return outputs[0], token_usage
            else:
                return outputs, token_usage

        except Exception as e:
            attempt += 1
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, query_config.backoff_max_seconds)
            print(
                f"LLM query attempt {attempt} failed with error: {e}. "
                f"Retrying in {backoff_seconds} seconds...",
                flush=True,
            )
            continue

    raise RuntimeError(f"LLM query failed after {query_config.max_retries} attempts.")


def add_token_usage(a, b):
    if a is None:
        return b
    if b is None:
        return a

    result = {}

    for key in set(a.keys()).union(b.keys()):
        va = a.get(key)
        vb = b.get(key)

        if isinstance(va, dict) and isinstance(vb, dict):
            result[key] = add_token_usage(va, vb)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            result[key] = va + vb
        elif va is None:
            result[key] = vb
        elif vb is None:
            result[key] = va
        else:
            result[key] = va

    return result
