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
        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        combine_0 = [
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
        ]

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        add_n_0 = paddle._C_ops.add_n(combine_0)
        return add_n_0
