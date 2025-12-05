from graph_net.imp_util import load_module
from graph_net.torch.sym_dim_reifiers.reifier_base import ReifierBase
import os


def get_reifier(reifier_name) -> ReifierBase:
    import graph_net.torch.sym_dim_reifiers as sym_dim_reifiers

    py_module = load_module(
        f"{os.path.dirname(sym_dim_reifiers.__file__)}/{reifier_name}.py"
    )
    return py_module.ConcreteReifier
