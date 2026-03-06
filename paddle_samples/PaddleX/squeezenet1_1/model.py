import paddle
from paddle.vision.models import squeezenet1_1


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(pretrained=False)

    def forward(self, x=None, **kwargs):
        # If x is not passed directly, try to find any tensor in kwargs
        if x is None:
            for val in kwargs.values():
                if isinstance(val, (paddle.Tensor, paddle.static.Variable)):
                    x = val
                    break

        if x is None:
            raise ValueError(
                f"No input tensor found. kwargs keys: {list(kwargs.keys())}"
            )

        return self.model(x)
