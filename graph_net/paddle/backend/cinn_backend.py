import paddle
from graph_net.paddle.backend.graph_compiler_backend import GraphCompilerBackend


class CinnBackend(GraphCompilerBackend):
    def __call__(self, model, input_spec=None):
        build_strategy = paddle.static.BuildStrategy()
        compiled_model = paddle.jit.to_static(
            model,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )
        compiled_model.eval()
        program = compiled_model.forward.concrete_program.main_program
        return compiled_model

    def synchronize(self):
        if (
            paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
        ):
            paddle.device.synchronize()
