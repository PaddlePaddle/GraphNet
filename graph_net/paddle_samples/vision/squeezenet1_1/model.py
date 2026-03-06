import paddle
from paddle.vision.models import squeezenet1_1

class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(pretrained=False)

    def forward(self, *args, **kwargs):
        # 1. Try to get from positional args
        # 2. Try to get from kwargs with common names
        # 3. If all else fails, take the first value in kwargs dictionary
        if args:
            x = args[0]
        elif "inputs" in kwargs:
            x = kwargs["inputs"]
        elif kwargs:
            x = list(kwargs.values())[0]
        else:
            raise ValueError("No input tensor found in forward arguments")
            
        return self.model(x)
