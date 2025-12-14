import torch
from graph_net.torch.backend.graph_compiler_backend import GraphCompilerBackend
from graph_net.torch.backend.agent_ncu import agent_compile


class AgentCompiledModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.count_compile = 0

    def forward(self, *args, **kwargs):
        self.module = self.compile(*args, **kwargs)
        return self.module

    def compile(self, *args, **kwargs):
        dummy_input = tuple([*args, *kwargs.values()])
        optimized_module = agent_compile.optimize(
            self.module,
            model_inputs=dummy_input,
            task_name=f"default_task_{self.count_compile}",
            language="cuda",
        )
        self.count_compile += 1
        return optimized_module


class AgentBackend(GraphCompilerBackend):
    def __call__(self, model):
        return AgentCompiledModule(model)

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
