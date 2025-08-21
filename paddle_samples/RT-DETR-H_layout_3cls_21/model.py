import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
        data_11,
    ):
        # builtin.combine: ([1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32]) <- (1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32)
        combine_0 = [data_0, data_1, data_2, data_3, data_4, data_5]

        # pd_op.stack: (6x1x498x4xf32) <- ([1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32, 1x498x4xf32])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # builtin.combine: ([1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32]) <- (1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32)
        combine_1 = [data_6, data_7, data_8, data_9, data_10, data_11]

        # pd_op.stack: (6x1x498x11xf32) <- ([1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32, 1x498x11xf32])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        return stack_0, stack_1
