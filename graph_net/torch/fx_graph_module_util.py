import os
import inspect
from graph_net.tensor_meta import TensorMeta
from graph_net.imp_util import load_module
from dataclasses import asdict


def get_fx_graph_num_ops(fx_graph_module):
    def get_num_ops(node):
        return 0 if node.op in {"placeholder", "output"} else 1

    return sum(map(get_num_ops, fx_graph_module.graph.nodes))


def get_torch_module_and_inputs(model_path, use_dummy_inputs=True):
    module = _get_torch_module(model_path)
    tensor_metas = _get_tensor_metas(model_path)
    inputs = _create_inputs_by_metas(module, tensor_metas, use_dummy_inputs)
    return module, inputs


def _get_torch_module(model_path):
    py_module = load_module(f"{model_path}/model.py")
    torch_module_cls = py_module.GraphModule
    torch_module_cls.__graph_net_file_path__ = model_path
    return torch_module_cls()


def _get_tensor_metas(model_path):
    make = TensorMeta.unserialize_from_py_file
    return [
        *make(os.path.join(model_path, "input_meta.py")),
        *make(os.path.join(model_path, "weight_meta.py")),
    ]


def _create_inputs_by_metas(module, tensor_metas, use_dummy_inputs):
    tensor_meta_attrs_list = [asdict(tensor_meta) for tensor_meta in tensor_metas]
    from graph_net.torch.utils import get_named_tensors

    named_tensors = get_named_tensors(tensor_meta_attrs_list, use_dummy_inputs)
    name2tensor = {k: v for k, v in named_tensors}
    return tuple(
        name2tensor[name] for name in inspect.signature(module.forward).parameters
    )
