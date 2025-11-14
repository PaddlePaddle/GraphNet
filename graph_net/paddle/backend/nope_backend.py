import paddle
from graph_net.paddle.backend.graph_compiler_backend import GraphCompilerBackend


class NopeBackend(GraphCompilerBackend):
    def __call__(self, model, input_spec=None):
        return model

    def synchronize(self):
        if (
            paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_rocm()
            or paddle.device.is_compiled_with_xpu()
        ):
            paddle.device.synchronize()
