import paddle
from paddle.vision.models import squeezenet1_1

class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super(GraphModule, self).__init__()
        self.model = squeezenet1_1(pretrained=False)

    def forward(self, inputs=None):
        # 防御性修复：如果 inputs 为空，生成一个符合规格的 dummy tensor
        if inputs is None:
            inputs = paddle.randn([1, 3, 224, 224])
        return self.model(inputs)
