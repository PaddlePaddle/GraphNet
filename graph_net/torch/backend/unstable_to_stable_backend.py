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

    def avg_pool2d_to_avg_pool2d(self, gm):
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

    def unstable_to_stable(self, gm):
        # Convert based on unstable_api environment variable
        if self.unstable_api == "torch._C._nn.avg_pool2d":
            gm = self.avg_pool2d_to_avg_pool2d(gm)
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
