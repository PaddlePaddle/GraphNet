import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8
    ):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([576x4xf32, 2304x4xf32, 9216x4xf32]) <- (576x4xf32, 2304x4xf32, 9216x4xf32)
        combine_0 = [data_0, data_1, data_2]

        # pd_op.concat: (12096x4xf32) <- ([576x4xf32, 2304x4xf32, 9216x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([576x2xf32, 2304x2xf32, 9216x2xf32]) <- (576x2xf32, 2304x2xf32, 9216x2xf32)
        combine_1 = [data_3, data_4, data_5]

        # pd_op.concat: (12096x2xf32) <- ([576x2xf32, 2304x2xf32, 9216x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_0)

        # builtin.combine: ([576x1xf32, 2304x1xf32, 9216x1xf32]) <- (576x1xf32, 2304x1xf32, 9216x1xf32)
        combine_2 = [data_6, data_7, data_8]

        # pd_op.concat: (12096x1xf32) <- ([576x1xf32, 2304x1xf32, 9216x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_0)
        return concat_0, concat_1, concat_2
