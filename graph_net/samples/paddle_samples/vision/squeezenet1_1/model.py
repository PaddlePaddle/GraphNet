import paddle
from paddle.vision.models import squeezenet1_1


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super(GraphModule, self).__init__()
        self.model = squeezenet1_1(pretrained=False)

    def forward(self, inputs):
        return self.model(inputs)
