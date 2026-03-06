import paddle
from paddle.vision.models import squeezenet1_1


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(pretrained=False)

    def forward(self, *args, **kwargs):
        # 兼容性抓取：优先抓取位置参数第一个，或者关键字参数名为 'inputs' 的
        x = args[0] if args else kwargs.get("inputs")
        return self.model(x)
