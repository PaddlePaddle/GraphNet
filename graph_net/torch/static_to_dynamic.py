import torch
import torch.fx as fx
from graph_net.torch.utils import convert_tensor_meta_attrs_list_to_named_tensors
from torch.fx.passes.shape_prop import ShapeProp
from graph_net.torch.utils import apply_templates
from pathlib import Path
import inspect
from typing import Any
from contextlib import contextmanager
from torch.export import export
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module
from graph_net.imp_util import load_module
import os


# used as configuration of python3 -m graph_net.torch.run_model
class StaticToDynamic:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config

    def __call__(self, module, dim_axes_pairs):
        return StaticToDynamicModule(self.config, module, dim_axes_pairs)


class StaticToDynamicModule(torch.nn.Module):
    def __init__(self, config, module, dim_axes_pairs):
        super().__init__()
        config = {} if config is None else config
        self.config = self.make_config(**config)
        self.module = module
        self.dim_axes_pairs = dim_axes_pairs

    def make_config(self, pass_names=()):
        if pass_names == ():
            pass_names = (
                "naive_call_method_view_pass",
                "naive_call_method_reshape_pass",
                "naive_call_method_expand_pass",
            )
        return {
            "pass_names": pass_names,
        }

    def need_rewrite(self):
        try:
            traced_module = torch.fx.symbolic_trace(self.module)
        except:
            return False
        return any(
            predicator(traced_module) for predicator, _ in self.get_conditional_passes()
        )

    def save_graph_module(self, graph_module, model_path):
        py_code = apply_templates(graph_module.code)
        (Path(model_path) / "model.py").write_text(py_code)

    def rewrite_with_tensor_meta_attrs_list(self, tensor_meta_attrs_list):
        named_tensors = convert_tensor_meta_attrs_list_to_named_tensors(
            tensor_meta_attrs_list
        )
        ret = self.rewrite(**{k: v for k, v in named_tensors})
        return ret

    def rewrite(self, *args, **kwargs):
        assert len(args) == 0
        inputs = tuple(
            kwargs[name] for name in inspect.signature(self.module.forward).parameters
        )
        traced_module = parse_sole_graph_module(self.module, inputs)

        for predicator, pass_fn in self.get_conditional_passes():
            if predicator(traced_module):
                ShapeProp(traced_module).propagate(*inputs)
                traced_module = pass_fn(traced_module)

        return traced_module

    def get_conditional_passes(self):
        from graph_net.torch.dim_gen_passes.pass_mgr import get_dim_gen_pass

        return [
            (pass_obj.need_rewrite, pass_obj.rewrite)
            for pass_name in self.config["pass_names"]
            for dim, axes in self.dim_axes_pairs
            for pass_cls in [get_dim_gen_pass(pass_name)]
            for pass_obj in [pass_cls(dim=dim, axes=axes)]
        ]

    def forward(self, *args, **kwargs):
        traced_module = self.rewrite(*args, **kwargs)

        inputs = [
            kwargs[name] for name in inspect.signature(self.module.forward).parameters
        ]
        ShapeProp(traced_module).propagate(*inputs)

        # return traced_module(*args, **kwargs)
