import logging
import torch
import torch.fx as fx
from graph_net.torch.utils import get_dummy_named_tensors
from torch.fx.passes.shape_prop import ShapeProp
from graph_net.torch.utils import apply_templates
from pathlib import Path
import inspect
from typing import Any
from contextlib import contextmanager
from torch.export import export
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module
from graph_net.torch.fx_graph_cache_util import (
    parse_immutable_model_path_into_sole_graph_module,
)
from graph_net.imp_util import load_module
import os


# used as configuration of python3 -m graph_net.torch.run_model
class StaticToDynamic:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config

    def __call__(self, module, dim_axes_pairs):
        return StaticToDynamicModulePass(self.config, module, dim_axes_pairs)

    def create_inputs_by_metas(self, module, tensor_meta_attrs_list):
        named_tensors = get_dummy_named_tensors(tensor_meta_attrs_list)
        name2tensor = {k: v for k, v in named_tensors}
        return tuple(
            name2tensor[name] for name in inspect.signature(module.forward).parameters
        )


class StaticToDynamicModulePass(torch.nn.Module):
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
                "non_batch_call_method_expand_pass",
                "non_batch_call_function_arange_pass",
                "non_batch_call_function_getitem_slice_pass",
                "non_batch_call_function_full_pass",
                "non_batch_call_function_full_plus_one_pass",
                "non_batch_call_function_zeros_pass",
                "non_batch_call_function_arange_plus_one_pass",
            )
        return {
            "pass_names": pass_names,
        }

    def get_pass_names(self):
        return self.config["pass_names"]

    def need_rewrite(self, inputs):
        try:
            logging.warning("before _create_fx_graph_module")
            traced_module = self._create_fx_graph_module(inputs)
            logging.warning("after _create_fx_graph_module")
            ShapeProp(traced_module).propagate(*inputs)
        except:
            return False
        return any(
            pass_obj.need_rewrite(traced_module)
            for pass_obj in self.get_conditional_passes()
        )

    def save_graph_module(self, graph_module, model_path):
        py_code = apply_templates(graph_module.code)
        (Path(model_path) / "model.py").write_text(py_code)

    def rewrite(self, inputs):
        traced_module = self._create_fx_graph_module(inputs)
        ShapeProp(traced_module).propagate(*inputs)
        for pass_obj in self.get_conditional_passes():
            if pass_obj.need_rewrite(traced_module):
                ShapeProp(traced_module).propagate(*inputs)
                traced_module = pass_obj.rewrite(traced_module)

        return traced_module

    def _create_fx_graph_module(self, inputs):
        if hasattr(self.module, "__graph_net_file_path__"):
            model_path = os.path.dirname(self.module.__graph_net_file_path__)
            return parse_immutable_model_path_into_sole_graph_module(model_path)
        else:
            return parse_sole_graph_module(self.module, inputs)

    def get_conditional_passes(self):
        from graph_net.torch.dim_gen_passes.pass_mgr import get_dim_gen_pass

        return [
            pass_obj
            for pass_name in self.config["pass_names"]
            for dim, axes in self.dim_axes_pairs
            for pass_cls in [get_dim_gen_pass(pass_name)]
            for pass_obj in [pass_cls(dim=dim, axes=axes)]
        ]

    def forward(self, *args, **kwargs):
        print(f"Do nothing.")
