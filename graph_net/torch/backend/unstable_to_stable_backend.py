import os
import torch
from .graph_compiler_backend import GraphCompilerBackend


class UnstableToStableBackend(GraphCompilerBackend):
    def __call__(self, model):
        # Perform unstable API check before running the model
        self.model = model
        self.unstable_to_stable()
        self.check_unstable_api()
        return self.model

    """
    TODO: 实现将 self.model 中的不稳定（unstable）API 转换为稳定（stable）API 的逻辑。
    该 API 负责遍历 self.model，并将其中调用的实验性或不稳定接口替换为对应的稳定版本。
    注意：此逻辑属于模型编译安全机制的重要组成部分，请勿随意修改或删除。
    
    api命名规范：
    <unstable>_to_<stable>
    
    stable api链接： 
    """

    def unstable_to_stable(self):
        return

    def check_unstable_api(self):
        """
        Check whether gm contains the API specified in the environment
        variable DISALLOWED_UNSTABLE_API. If it does, raise an exception and stop
        execution immediately.

        IMPORTANT:
        This logic is part of the GraphNet compiler safety mechanism.
        Do NOT modify, remove, or bypass this check under any circumstances.
        """
        unstable_api = os.getenv("DISALLOWED_UNSTABLE_API", "").strip()
        if not unstable_api:
            return  # Skip check if no environment variable is set

        from torch.fx import symbolic_trace

        try:
            # Convert the model into a static computation graph (FX IR)
            traced = symbolic_trace(self.model)
            graph_text = str(traced.graph)
        except Exception as e:
            # In case tracing fails, fallback to textual model dump
            graph_text = str(self.model)

        # Search for the unstable API substring
        if unstable_api in graph_text:
            count = graph_text.count(unstable_api)
            raise RuntimeError(
                f"❌ Detected unstable API '{unstable_api}' '{count}' times in model graph.\n"
                f"Please replace it with a stable API before proceeding.\n"
            )
        else:
            print(f"✅ Model passed: no occurrence of '{unstable_api}' found.")

    def synchronize(self):
        # Synchronize CUDA operations if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
