import logging
import torch
import copy
import os
import inspect
from graph_net.tensor_meta import TensorMeta
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module
from graph_net.imp_util import load_module
from dataclasses import asdict


def parse_immutable_model_path_into_sole_graph_module(model_path):
    model_path = os.path.realpath(model_path)
    if model_path not in g_model_path2graph_module:
        module = _get_torch_module(model_path)
        tensor_metas = _get_tensor_metas(model_path)
        logging.warning("before _create_inputs_by_metas")
        inputs = _create_inputs_by_metas(module, tensor_metas)
        logging.warning("after _create_inputs_by_metas")
        logging.warning("before parse_sole_graph_module")
        g_model_path2graph_module[model_path] = parse_sole_graph_module(module, inputs)
        logging.warning("after parse_sole_graph_module")
    return copy.deepcopy(g_model_path2graph_module[model_path])


def _get_torch_module(model_path):
    py_module = load_module(f"{model_path}/model.py")
    torch_module_cls = py_module.GraphModule
    return torch_module_cls()


def _get_tensor_metas(model_path):
    make = TensorMeta.unserialize_from_py_file
    return [
        *make(os.path.join(model_path, "input_meta.py")),
        *make(os.path.join(model_path, "weight_meta.py")),
    ]


def _create_inputs_by_metas(module, tensor_metas):
    tensor_meta_attrs_list = [asdict(tensor_meta) for tensor_meta in tensor_metas]
    from graph_net.torch.utils import get_dummy_named_tensors

    named_tensors = get_dummy_named_tensors(tensor_meta_attrs_list)
    name2tensor = {k: v for k, v in named_tensors}
    return tuple(
        name2tensor[name] for name in inspect.signature(module.forward).parameters
    )


g_model_path2graph_module = {}
