import os
import torch
import unittest

from torch.fx import symbolic_trace
from graph_net.torch.extractor import extract
from graph_net.torch.decompose_util import fold_range_to_submodule


# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        y = x + 1
        z = y * 2
        return z.clamp(min=0.0, max=1.0)


class WrapperModule(torch.nn.Module):
    def __init__(self, submodule, seq_no):
        super().__init__()
        self.submodule = submodule
        self.seq_no = seq_no

    def forward(self, *args):
        print("Args:")
        print(args)
        return self.submodule(*args)


def submodule_hook(submodule: torch.fx.GraphModule, seq_no):
    print(f"{'-'*8} [submodule-{seq_no}] {'-'*8}\n")
    print(submodule.graph)
    print(submodule.code)
    return WrapperModule(submodule, seq_no)


class TestExtractorSubmodule(unittest.TestCase):
    """Test extraction of submodules from traced GraphModule."""

    def test_sample(self):
        module = MyModule()

        # Symbolic tracing frontend - captures the semantics of the module
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

        # High-level intermediate representation (IR) - Graph representation
        print(symbolic_traced.graph)
        """
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %add : [num_users=1] = call_function[target=operator.add](args = (%x, 1), kwargs = {})
            %mul : [num_users=1] = call_function[target=operator.mul](args = (%add, 2), kwargs = {})
            %clamp : [num_users=1] = call_method[target=clamp](args = (%mul,), kwargs = {min: 0.0, max: 1.0})
            return clamp
        """

        # Code generation - valid Python code
        print(symbolic_traced.code)
        """
        def forward(self, x):
            add = x + 1;  x = None
            mul = add * 2;  add = None
            clamp = mul.clamp(min = 0.0, max = 1.0);  mul = None
            return clamp
        """

        inp = torch.tensor([1.0, 2.0, 3.0, 4.0])
        source_output = module(inp)
        traced_output = symbolic_traced(inp)

        folded = fold_range_to_submodule(
            symbolic_traced,
            start_node_idx=0,
            end_node_idx=2,
            submodule_hook=submodule_hook,
            # group_head_and_tail=False,
        )
        folded_output = folded(inp)

        print(f"{'-'*8} [folded] {'-'*8}\n")
        print(folded.graph)
        """
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %add : [num_users=1] = call_function[target=operator.add](args = (%x, 1), kwargs = {})
            %extraced_submodule : [num_users=1] = call_module[target=extraced_submodule](args = (%add,), kwargs = {})
            %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%extraced_submodule, 0), kwargs = {})
            return getitem

        """
        print(folded.code)
        """
        def forward(self, x):
            add = x + 1;  x = None
            extraced_submodule = self.extraced_submodule(add);  add = None
            getitem = extraced_submodule[0];  extraced_submodule = None
            return getitem
        """

        # Save to workspace, assumed workspace is ./tmp/
        os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "./tmp/"
        folded = extract("demo_test", False)(folded)
        wrapper_output = folded(inp)

        self.assertTrue(torch.allclose(source_output, traced_output))
        self.assertTrue(torch.allclose(source_output, folded_output))
        self.assertTrue(torch.allclose(source_output, wrapper_output))


if __name__ == "__main__":
    unittest.main()
