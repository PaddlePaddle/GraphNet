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
        data_12,
        data_13,
        data_14,
    ):
        # builtin.combine: ([xf32, xf32, xf32, xf32, xf32]) <- (xf32, xf32, xf32, xf32, xf32)
        combine_0 = [data_0, data_1, data_2, data_3, data_4]
        del data_0, data_1, data_2, data_3, data_4

        # pd_op.add_n: (xf32) <- ([xf32, xf32, xf32, xf32, xf32])
        add_n_0 = paddle._C_ops.add_n(combine_0)
        del combine_0

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        combine_1 = [data_5, data_6, data_7, data_8, data_9]
        del data_5, data_6, data_7, data_8, data_9

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        add_n_1 = paddle._C_ops.add_n(combine_1)
        del combine_1

        # builtin.combine: ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32]) <- (1xf32, 1xf32, 1xf32, 1xf32, 1xf32)
        combine_2 = [data_10, data_11, data_12, data_13, data_14]
        del data_10, data_11, data_12, data_13, data_14

        # pd_op.add_n: (1xf32) <- ([1xf32, 1xf32, 1xf32, 1xf32, 1xf32])
        add_n_2 = paddle._C_ops.add_n(combine_2)
        del combine_2

        return add_n_0, add_n_1, add_n_2
