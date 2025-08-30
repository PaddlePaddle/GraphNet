import torch

try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None


class GraphCompilerBackend:
    def __call__(self, model):
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()


class InductorBackend(GraphCompilerBackend):
    def __call__(self, model):
        return torch.compile(model, backend="inductor")

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class TensorRTBackend(GraphCompilerBackend):
    def __call__(self, model):
        return torch.compile(model, backend="tensorrt")

    def synchronize(self):
        torch.cuda.synchronize()
