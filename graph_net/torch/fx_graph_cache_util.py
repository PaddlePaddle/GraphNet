import copy
import os
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs


def parse_immutable_model_path_into_sole_graph_module(model_path):
    model_path = os.path.realpath(model_path)
    if model_path not in g_model_path2graph_module:
        module, inputs = get_torch_module_and_inputs(model_path)
        g_model_path2graph_module[model_path] = parse_sole_graph_module(module, inputs)
    return copy.deepcopy(g_model_path2graph_module[model_path])


g_model_path2graph_module = {}
