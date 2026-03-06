import paddle
from paddle.vision.models import squeezenet1_1


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(pretrained=False)

    def forward(self, *args, **kwargs):
        # 1. Try to get tensor from positional arguments
        if args:
            x = args[0]
        # 2. Try to get tensor from keyword arguments
        else:
            x = None
            for val in kwargs.values():
                if isinstance(val, (paddle.Tensor, paddle.static.Variable)):
                    x = val
                    break

            if x is None:
                # Log keys for debugging in CI if it fails again
                raise ValueError(
                    f"No input tensor found. args_len: {len(args)}, kwargs_keys: {list(kwargs.keys())}"
                )

        return self.model(x)
