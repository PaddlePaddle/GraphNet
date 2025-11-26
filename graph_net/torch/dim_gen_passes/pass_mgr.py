from graph_net.imp_util import load_module
import os


def get_dim_gen_pass(pass_name) -> "DimensionGeneralizationPass":
    import graph_net.torch.dim_gen_passes as dgpass

    py_module = load_module(f"{os.path.dirname(dgpass.__file__)}/{pass_name}.py")
    return py_module.ConcretePass
