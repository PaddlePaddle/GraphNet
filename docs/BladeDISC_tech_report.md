# 1 - Introduction

BladeDISC is an end-to-end **Dynamic Shape Compiler** project for machine learning workloads, which is one of the key components of Alibaba's [PAI-Blade](https://www.aliyun.com/activity/bigdata/blade). For more information, please refer to  [Github BladeDISC | TorchBlade Overview](https://github.com/alibaba/BladeDISC/blob/main/docs/developers/bladedisc_torch_overview.md).

This technical report demonstrates that `graph_net.torch.test_compiler` supports using the BladeDISC compiler as a backend, i.e., it supports configuring `--compiler "bladedisc"`, reads subgraphs from the `GraphNet/samples` directory, and successfully executes and obtains correct evaluation results.

Taking BERT as an example  [Optimize and Inference BERT with TorchBlade](https://github.com/alibaba/BladeDISC/blob/main/docs/tutorials/torch_bert_inference.md), the main execution process is as follows:

1. Convert the PyTorch model to TorchScript using `torch.jit.trace` or `torch.jit.script`.
2. Compile and optimize the model using BladeDISC's `torch_blade.optimize` to generate the compiled model `compiled_model`.
3. Combine the compiled model with input parameters `compiled_model(input)` to execute the forward pass.

The process of compiling and optimizing with `torch.jit.trace` or `torch.jit.script` can be abstracted as follows:

```shell
# allow_tracing=True   using torch.jit.trace(model, inputs)
compiled_model = torch_blade.optimize(model, allow_tracing=True, model_inputs=tuple(inputs))
# allow_tracing=False  using torch.jit.script(model)
compiled_model = torch_blade.optimize(model, allow_tracing=False)
```

In the test of this report, `torch.jit.trace` was used.


# 2 - Installation Instructions

> The installation environment in this section is also the test environment used in Chapter 3.

Official quick deployment options include [Install BladeDISC With Docker](https://github.com/alibaba/BladeDISC/blob/main/docs/install_with_docker.md) or  [Build BladeDISC from Source](https://github.com/alibaba/BladeDISC/blob/main/docs/build_from_source.md).

However, BladeDISC's last official support ended in 2022, when it was adapted for PyTorch 1.X series. Compiling from source requires specific modifications to adapt to PyTorch 2.X. Therefore, it is recommended to use the official image `bladedisc/bladedisc:latest-runtime-torch1.12.0-cu113` to quickly obtain compiler performance evaluation data.

```shell
docker run -itd --gpus all --name torch_bladedisc_test -v /your_path:/your_path registry.cn-shanghai.aliyuncs.com/bladedisc/bladedisc:latest-runtime-torch1.12.0-cu113 /bin/bash
```

**Note**: Since BladeDISC is not adapted for PyTorch 2.X, certain parts of GraphNet that depend on higher versions of PyTorch should be commented out before execution. For example, `GraphNet/graph_net/torch/__init__.py` should be modified as follows:

```shell
"""
GraphNet PyTorch Implementation
"""
# from .extractor import extract
# from .samples_util import get_default_samples_directory
# __all__ = ["extract", "get_default_samples_directory"]
```



### 3 - Test Report

- BladeDISC for torch (import torch_blade) does not exhibit any entire category of models failing to run in the existing `/samples` (as of 2025.08.30).

- For all models under `/samples/cosyvoice`, batch performance testing on GPU A100-SXM-40GB is documented in `BladeDISC_batch_test.txt`.

- For each category in `/samples`, one model was tested. The validation report can be found in `BladeDISC_validation_report.txt`, with a performance overview as follows:

  | Model                                                        | Eager (ms) | Compiled (ms) |
  | ------------------------------------------------------------ | ---------- | ------------- |
  | cosyvoice/CosyVoice-300M                                     | 8.4000     | 8.3600        |
  | mmpose/2xmspn_50                                             | 17.1000    | 14.1000       |
  | mmseg/ANN_R50                                                | 21.7000    | 21.8000       |
  | nemo/parakeet-ctc-0.6b                                       | 55.3000    | 54.4000       |
  | torchaudio/convtasnet_base_libri2mix                         | 99.4000    | 99.6000       |
  | torchgeometric/LINKX                                         | 1.0300     | 0.7280        |
  | timm/darknet17                                               | 2.1500     | 2.1300        |
  | torchvision/deeplabv3_resnet50                               | 8.4300     | 7.6200        |
  | transformers-auto-model/hf-tiny-model-private_tiny-random-AltCLIPModel | 6.0000     | 4.4200        |
  | ultralytics/yolo11l-cls                                      | 17.6000    | 14.8000       |



# 4 - Execution Issue Analysis

### Issue 1: Unsupported Operators

The PyTorch version is too old (1.X), and some operators are only available in newer versions. For example:

```shell
Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 490, in <module>
    main(args=args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 442, in main
    test_single_model(args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 243, in test_single_model
    model = get_model(args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 102, in get_model
    model_class = load_class_from_file(args, class_name="GraphModule")
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 90, in load_class_from_file
    exec(compiled_code, module.__dict__)
  File "/daiwenhao/GraphNet/samples/torchvision/alexnet/model.py", line 4, in <module>
    class GraphModule(torch.nn.Module):
  File "/daiwenhao/GraphNet/samples/torchvision/alexnet/model.py", line 9, in GraphModule
    s1: torch.SymInt,
AttributeError: module 'torch' has no attribute 'SymInt'
```

Another example:

```shell
weight should have at least three dimensions
Failed! Try to export it through torch.jit.script:
object has no attribute scaled_dot_product_attention:
  File "/daiwenhao/GraphNet/samples/torchaudio/hubert_base/model.py", line 609
        v = view_2.transpose(2, 1)
        view_2 = None
        attn_output = torch._C._nn.scaled_dot_product_attention(
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
Fail to export torchscript on the top level of the model, We will iterate over the submodules and replace those that can be successfully exported by the torch.jit.script
Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 494, in <module>
    main(args=args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 446, in main
    test_single_model(args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 290, in test_single_model
    eager_stats = measure_performance(eager_model_call, args, compiler)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 228, in measure_performance
    times = time_execution_with_cuda_event(
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 167, in time_execution_with_cuda_event
    kernel_fn(*args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 287, in <lambda>
    eager_model_call = lambda: model(**input_dict)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/daiwenhao/GraphNet/samples/torchaudio/hubert_base/model.py", line 609, in forward
    attn_output = torch._C._nn.scaled_dot_product_attention(
AttributeError: module 'torch._C._nn' has no attribute 'scaled_dot_product_attention'
```

### Issue 2: Unsupported Dynamic Types

Still due to the outdated PyTorch version (1.X), dynamic types in models are not supported.

```shell
object has no attribute sym_size:
  File "/daiwenhao/GraphNet/samples/torchgeometric/GAT/model.py", line 114
        edge_index = l_edge_index_[(slice(None, None, None), mask)]
        mask = None
        sym_size_int = torch.ops.aten.sym_size.int(edge_index, 1)
                       ~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
        _check_is_size = torch._check_is_size(sym_size_int)
        _check_is_size = None
Fail to export torchscript on the top level of the model, We will iterate over the submodules and replace those that can be successfully exported by the torch.jit.script
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/torch/_ops.py", line 198, in __getattr__
    op, overload_names = torch._C._jit_get_operation(qualified_op_name)
RuntimeError: No such operator aten::sym_size
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 492, in <module>
    main(args=args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 444, in main
    test_single_model(args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 288, in test_single_model
    eager_stats = measure_performance(eager_model_call, args, compiler)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 226, in measure_performance
    times = time_execution_with_cuda_event(
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 165, in time_execution_with_cuda_event
    kernel_fn(*args)
  File "/daiwenhao/GraphNet/graph_net/torch/test_compiler.py", line 285, in <lambda>
    eager_model_call = lambda: model(**input_dict)
  File "/usr/local/lib/python3.8/dist-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/daiwenhao/GraphNet/samples/torchgeometric/GAT/model.py", line 114, in forward
    sym_size_int = torch.ops.aten.sym_size.int(edge_index, 1)
  File "/usr/local/lib/python3.8/dist-packages/torch/_ops.py", line 202, in __getattr__
    raise AttributeError(f"'_OpNamespace' object has no attribute '{op_name}'") from e
AttributeError: '_OpNamespace' object has no attribute 'sym_size'
```

### Issue 3: Unsupported `device(type="cuda", index=0)`

In torch.jit.script execution mode, the BladeDISCBackend does not require input specifications, but `device(type="cuda", index=0)` is not supported by TorchScript; only `torch.device("cuda")` is supported.

```shell
The following variants are available:
  aten::device(str a) -> (Device):
  Argument a not provided.
  
  device(str type) -> (Device):
  Keyword argument index unknown.

The original call is:
  File "/daiwenhao/GraphNet/samples/ultralytics/yolo11l/model.py", line 6511
        l_self_modules_model_modules_23_stride = None
        arange = torch.arange(
            end=80, device=device(type="cuda", index=0), dtype=torch.float32
                           ~~~~~~ <--- HERE
        )
        sx = arange + 0.5

Fail to export torchscript on the top level of the model, We will iterate over the submodules and replace those that can be successfully exported by the torch.jit.script
```