import paddle
from paddle.vision.models import squeezenet1_1

class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(pretrained=False)

    def forward(self, *args, **kwargs):
        # 1. Try to get from positional args
        if args:
            x = args[0]
        # 2. Try to get from kwargs (looking for any Tensor)
        else:
            x = None
            for val in kwargs.values():
                if isinstance(val, (paddle.Tensor, paddle.static.Variable)):
                    x = val
                    break
            
            if x is None:
                raise ValueError(f"No input tensor found. args: {len(args)}, kwargs keys: {list(kwargs.keys())}")
            
        return self.model(x)
