import os
import torch
import warnings
from graph_net.config.fetch_agent_config import get_llm_config
from graph_net.torch.backend.agent_utils.query_llm_utils import (
    LLMQueryConfig,
    query_llm_service,
    add_token_usage,
)
from .prompt_cuda import (
    generate_default_cuda_prompt,
    judge_optimize_prompt,
    judge_correct_prompt,
    coder_optimize_prompt,
    coder_correct_prompt,
)
from .kernel_text_util import (
    extract_cuda_code,
    remove_pybind_module,
    compile_kernel,
    exec_eval_cuda,
    exec_eval_cuda_with_ncu,
    format_with_kernelbench_style,
)
from .prompt_cuda import PROMPT_SYSTEM


def generate(prompt, system_prompt, llm_query_config: LLMQueryConfig):
    try:
        query_result = query_llm_service(
            prompt=prompt, system_prompt=system_prompt, query_config=llm_query_config
        )
        return query_result
    except Exception as e:
        raise RuntimeError(f"LLM query failed with exception: {e}")


def optimize(module, model_inputs=None, task_name: str = "default_task"):
    """Optimize the given PyTorch module using custom DSL operators."""

    llm_config = get_llm_config()
    llm_query_config = LLMQueryConfig(**llm_config)
    traced_module = torch.fx.symbolic_trace(module)

    if "cuda" == llm_config.dsl.lower():
        return cuda_optimize(traced_module, model_inputs, task_name, llm_query_config)
    elif "triton" == llm_config.dsl.lower():
        return torch.compile(module)  # TODO add custom triton optimize
    else:
        raise NotImplementedError(f"Unsupported language: {llm_config.dsl}")

    # return the best of optimized models


def agent_fix_cuda_error(
    error_info, torch_model_code, cuda_code, work_dir, llm_query_config
):
    """Fix CUDA code based on error information from compilation or execution."""

    prompt = judge_correct_prompt(error_info, torch_model_code, cuda_code)
    modify_text, tokens_judge_correct = generate(
        prompt, PROMPT_SYSTEM, llm_query_config
    )
    prompt = coder_correct_prompt(error_info, cuda_code, modify_text)
    with open(os.path.join(work_dir, "fix_cuda_error_prompt.txt"), "a") as f:
        f.write(prompt)
    cuda_code, tokens_cuda_code_fix = generate(prompt, PROMPT_SYSTEM, llm_query_config)
    cuda_code = extract_cuda_code(cuda_code)
    cuda_code = remove_pybind_module(cuda_code)
    return cuda_code, add_token_usage(tokens_judge_correct, tokens_cuda_code_fix)


def cuda_optimize(
    gm,
    model_inputs,
    task_name: str = "default_task",
    llm_query_config: LLMQueryConfig = None,
):
    best_model = gm
    max_iters = llm_query_config.iterative_query_nums
    store_dir = os.path.join(llm_query_config.top_save_dir, task_name)

    torch_model_code = format_with_kernelbench_style(gm, model_inputs)

    # iterative optimization
    cur_iter_token_usage = None
    for iter in range(max_iters):
        print(f"=== Optimize {task_name}, Iteration {iter} ===", flush=True)

        context_dir_path = os.path.join(store_dir, f"iter_{iter}")
        already_done: bool = os.path.exists(
            os.path.join(context_dir_path, "model_new.py")
        )

        # Skip already done iterations
        if already_done:
            print(f"Iteration {iter} already done, skipping...", flush=True)
            with open(os.path.join(context_dir_path, "model_new.py"), "r") as fin:
                cuda_code = fin.read()
            continue
        os.makedirs(context_dir_path, exist_ok=True)

        is_success_compilable = False
        is_success_functional = False

        # Generate initial kernel code
        if iter == 0:
            prompt = generate_default_cuda_prompt(torch_model_code)
            text_response, cur_iter_token_usage = generate(
                prompt, PROMPT_SYSTEM, llm_query_config
            )
            raw_cuda_code = extract_cuda_code(text_response)
            cuda_code = remove_pybind_module(raw_cuda_code)

        # compile
        try:
            is_success_compilable, compile_info = compile_kernel(
                cuda_code=cuda_code, work_dir=context_dir_path
            )

            with open(os.path.join(context_dir_path, "log.log"), "a") as f:
                f.write(
                    f"[Token Usage] Iteration {iter} cost: {cur_iter_token_usage}\n"
                )
            cur_iter_token_usage = None

        except Exception as e:  # wrapper compile_kernel to catch all exceptions
            print(f"Compilation failed with exception: {e}")
            continue

        # [Eval Result] compile failed
        if not is_success_compilable:
            cuda_code, fix_error_token = agent_fix_cuda_error(
                compile_info["msg"][:4096],  # save tokens
                torch_model_code,
                cuda_code,
                work_dir=context_dir_path,
                llm_query_config=llm_query_config,
            )
            cur_iter_token_usage = add_token_usage(
                cur_iter_token_usage, fix_error_token
            )
            continue

        # [Result-Compile] compile success
        try:
            is_success_functional, eval_msg = exec_eval_cuda(
                compile_info["exec_filename"],  # .so filename
                compile_info["exec_content"],  # .so binary content
                torch_model_code,
                work_dir=context_dir_path,
            )
        except Exception as e:
            print(f"Execution failed with exception: {e}", flush=True)
            continue

        # [Result-Execute] functional failed
        if not is_success_functional:
            cuda_code, fix_error_token = agent_fix_cuda_error(
                eval_msg[:4096],
                torch_model_code,
                cuda_code,
                work_dir=context_dir_path,
                llm_query_config=llm_query_config,
            )
            cur_iter_token_usage = add_token_usage(
                cur_iter_token_usage, fix_error_token
            )
            continue

        # [Result-Execute] functional success: optimization with NCU analysis
        else:
            is_ncu_success, ncu_metric_info = exec_eval_cuda_with_ncu(
                compile_info["exec_filename"],
                compile_info["exec_content"],
                work_dir=context_dir_path,
            )
            if not is_ncu_success:
                warnings.warn("NCU analysis failed.", RuntimeWarning)
                continue
            optimize_prompt = judge_optimize_prompt(
                torch_model_code, cuda_code, ncu_metric_info
            )
            optimize_strategy, strategy_token = generate(
                optimize_prompt, PROMPT_SYSTEM, llm_query_config
            )
            optimize_prompt = coder_optimize_prompt(cuda_code, optimize_strategy)
            cuda_code, cuda_gen_token = generate(
                optimize_prompt, PROMPT_SYSTEM, llm_query_config
            )
            cuda_code = extract_cuda_code(cuda_code)
            cuda_code = remove_pybind_module(cuda_code)
            cur_iter_token_usage = add_token_usage(cur_iter_token_usage, strategy_token)
            cur_iter_token_usage = add_token_usage(cur_iter_token_usage, cuda_gen_token)

    return best_model
