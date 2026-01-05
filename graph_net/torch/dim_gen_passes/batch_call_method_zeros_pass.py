import torch
import torch.fx as fx
from graph_net.torch.dim_gen_passes import DimensionGeneralizationPass
import os


class ConcretePass(DimensionGeneralizationPass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_functions = {
            torch.zeros,
            torch.ones,
            torch.empty,
            torch.randn,
            torch.rand,
        }

    def get_pass_name(cls) -> str:
        return os.path.basename(__file__)[:-3]

    def need_rewrite(self, traced_module: fx.GraphModule) -> bool:
        if 0 not in self.axes:
            return False
        for node in traced_module.graph.nodes:
            if node.op == "call_function" and node.target in self.target_functions:
                return True
        return False

    def _node_need_rewrite(self, node) -> bool:
        if not (node.op == "call_function"):
            return False
        if node.target not in self.target_functions:
            return False
        if len(node.args) < 1:
            return False
        if not isinstance(node.args[0], (tuple, list)):
            return False
        if self.dim not in node.args[0]:
            return False
        return True

    def rewrite(self, traced_module: fx.GraphModule) -> fx.GraphModule:
        ref_node = next(
            node for node in traced_module.graph.nodes if node.op == "placeholder"
        )

        with traced_module.graph.inserting_after(ref_node):
            size_0 = traced_module.graph.call_method("size", args=(ref_node, 0))

        for node in traced_module.graph.nodes:
            if self._node_need_rewrite(node):
                zero_args = list(node.args[0])
                for axis_idx, target_dim in enumerate(zero_args):
                    if (
                        axis_idx == 0
                        and isinstance(target_dim, int)
                        and target_dim == self.dim
                    ):
                        zero_args[axis_idx] = size_0
                        new_args = list(node.args)
                        new_args[0] = (
                            tuple(zero_args)
                            if isinstance(node.args[0], tuple)
                            else zero_args
                        )
                        node.args = tuple(new_args)

        traced_module.graph.lint()
        traced_module.recompile()
        return traced_module
