import torch
import torch.nn as nn
import os
import sys
import inspect
import importlib.util
from typing import List, Dict


class ComposedModel(nn.Module):
    def __init__(self, submodules: List[nn.Module]):
        super().__init__()
        self.submodules = nn.ModuleList(submodules)
        self.submodule_param_names = [
            list(inspect.signature(sm.forward).parameters.keys())
            for sm in self.submodules
        ]

    def forward(self, **kwargs):
        current_args = kwargs
        for i, (sm, param_names) in enumerate(
            zip(self.submodules, self.submodule_param_names)
        ):
            # 准备当前子图的输入字典
            call_kwargs = {}
            if i > 0:
                # 对于后续子图，第一个参数是上一个子图的输出
                first_param_name = param_names[0]
                call_kwargs[first_param_name] = current_args  # current_args 此时是上一个子图的输出

            # 从主输入字典中筛选出当前子图需要的权重参数
            for name in param_names:
                if name in current_args:
                    call_kwargs[name] = current_args[name]

            outputs = sm(**call_kwargs)
            # 假设每个子图只有一个输出，并且返回的是一个元组
            current_args = outputs[0]

        return (current_args,)


class RangeDecomposerValidatorBackend:
    def _load_model_instance(self, path: str, device: str) -> torch.nn.Module:
        class_name = "GraphModule"
        model_file = os.path.join(path, "model.py")

        spec = importlib.util.spec_from_file_location(class_name, model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ModelClass = getattr(module, class_name)
        instance = ModelClass().to(device)
        return instance

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        model_file_path = inspect.getfile(
            model.__class__
        )  # e.g., /test/simple_CNN/model.py
        model_dir = os.path.dirname(model_file_path)  # e.g., /test/simple_CNN

        decomposed_parent_dir = (
            model_dir + "_decomposed"
        )  # e.g., /test/simple_CNN_decomposed
        subgraph_paths = []
        for name in sorted(os.listdir(decomposed_parent_dir)):
            full_path = os.path.join(decomposed_parent_dir, name)
            if os.path.isdir(full_path) and name.startswith("subgraph_"):
                subgraph_paths.append(full_path)

        print(
            f"[RangeDecomposerValidatorBackend] Found subgraphs: {[os.path.basename(p) for p in subgraph_paths]}"
        )

        submodule_instances = []
        device = next(model.parameters()).device  # 从传入的model获取device信息

        for path in subgraph_paths:
            instance = self._load_model_instance(path, device)
            submodule_instances.append(instance)
            dir_name = os.path.basename(path)
            print(
                f"[RangeDecomposerValidatorBackend] Loaded and instantiated '{dir_name}'"
            )

        composed_model = ComposedModel(submodule_instances)
        return composed_model.eval()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
