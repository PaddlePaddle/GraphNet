import paddle
from paddle.vision.models import squeezenet1_1


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(pretrained=False)

    def forward(self, **kwargs):
        # 无论 validate.py 传入 key 是什么，都能在这里被接收
        # 根据 input_meta.py，传进来的 key 应该是 'inputs'
        x = kwargs.get("inputs")
        return self.model(x)
