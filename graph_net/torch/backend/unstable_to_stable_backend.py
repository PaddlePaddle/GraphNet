import os
import torch
import sys
import inspect
from .graph_compiler_backend import GraphCompilerBackend
from ..fx_graph_serialize_util import serialize_graph_module_to_str


class UnstableToStableBackend(GraphCompilerBackend):
    def __call__(self, model):
        # Perform unstable API check before running the model
        unstable_api = os.getenv("DISALLOWED_UNSTABLE_API", "").strip()
        self.unstable_api = unstable_api

        def my_backend(gm, sample_inputs):
            gm = self.unstable_to_stable(gm)
            self.check_unstable_api(gm)
            return gm.forward

        return torch.compile(backend=my_backend)(model)

    """
    TODO: Implement logic to convert unstable APIs in `self.model` into their stable counterparts.
    This API is responsible for traversing `self.model` and replacing any calls to experimental or unstable interfaces with the corresponding stable versions.
    Note: This logic is a critical component of the model compilation safety mechanism—do not modify or remove it without caution.

    **API naming convention:**
    `<unstable>_to_<stable>`

    **Stable API reference link:**
    """

    def _impl_unstable_to_stable_irfft(self, gm):
        def replace_in_graph(graph_mod):
            # Register stable implementation on GraphModule, codegen can use self.irfft
            try:
                setattr(graph_mod, "irfft", torch.fft.irfft)
            except Exception:
                pass

            for node in graph_mod.graph.nodes:
                if node.op == "call_function":
                    # Match for all forms of target names
                    if "fft_irfft" in str(node.target):
                        # Directly point target to Python layer function
                        node.target = torch.fft.irfft
            # Validate and recompile the graph
            graph_mod.graph.lint()
            graph_mod.recompile()

        # Process main gm and all nested GraphModules
        modules = [gm]
        modules += [
            m
            for _, m in gm.named_modules()
            if isinstance(m, torch.fx.GraphModule) and m is not gm
        ]
        for m in modules:
            replace_in_graph(m)

        return gm

    def _impl_unstable_to_stable_avg_pool2d(self, gm):
        """
        Convert torch._C._nn.avg_pool2d to torch.nn.functional.avg_pool2d
        """
        import torch.nn.functional as F

        # Update graph nodes: replace torch._C._nn.avg_pool2d with F.avg_pool2d
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if (
                    hasattr(node.target, "__module__")
                    and hasattr(node.target, "__name__")
                    and node.target.__module__ == "torch._C._nn"
                    and node.target.__name__ == "avg_pool2d"
                ):
                    node.target = F.avg_pool2d

        # Recompile the graph
        gm.recompile()

        return gm

    def _impl_unstable_to_stable_rfft(self, gm):
        """
        Convert torch._C._fft.fft_rfft to torch.fft.rfft
        """
        # Update graph nodes: replace torch._C._fft.fft_rfft with torch.fft.rfft
        issue_nodes = (
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            if hasattr(node.target, "__module__")
            if node.target.__module__ == "torch._C._fft"
            if hasattr(node.target, "__name__")
            if node.target.__name__ == "fft_rfft"
        )
        for node in issue_nodes:
            node.target = torch.fft.rfft

        # Recompile the graph
        gm.recompile()

        return gm

    def unstable_to_stable(self, gm):
        methods = (
            name
            for name in vars(type(self)).keys()
            if name.startswith("_impl_unstable_to_stable")
        )
        for method in methods:
            gm = getattr(self, method)(gm)
        return gm

    def check_unstable_api(self, gm):
        """
        Check whether gm contains the API specified in the environment
        variable DISALLOWED_UNSTABLE_API. If it does, raise an exception and stop
        execution immediately.

        IMPORTANT:
        This logic is part of the GraphNet compiler safety mechanism.
        Do NOT modify, remove, or bypass this check under any circumstances.
        """

        # Use serialized code to check for unstable APIs
        graph_text = serialize_graph_module_to_str(gm)
        # Search for the unstable API substring
        if self.unstable_api in graph_text:
            count = graph_text.count(self.unstable_api)
            print(f"❌unstable_api:{self.unstable_api} occurs {count} times")
            sys.exit(-1)
        else:
            print(f"✅ Model passed: no occurrence of '{self.unstable_api}' found.")

    def synchronize(self):
        # Synchronize CUDA operations if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
