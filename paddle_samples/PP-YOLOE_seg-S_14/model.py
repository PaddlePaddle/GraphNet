import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8
    ):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # builtin.combine: ([1x400x2xf32, 1x1600x2xf32, 1x6400x2xf32]) <- (1x400x2xf32, 1x1600x2xf32, 1x6400x2xf32)
        combine_0 = [data_0, data_1, data_2]

        # pd_op.concat: (1x8400x2xf32) <- ([1x400x2xf32, 1x1600x2xf32, 1x6400x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([1x400x32xf32, 1x1600x32xf32, 1x6400x32xf32]) <- (1x400x32xf32, 1x1600x32xf32, 1x6400x32xf32)
        combine_1 = [data_3, data_4, data_5]

        # pd_op.concat: (1x8400x32xf32) <- ([1x400x32xf32, 1x1600x32xf32, 1x6400x32xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_1, full_0)

        # builtin.combine: ([1x400x68xf32, 1x1600x68xf32, 1x6400x68xf32]) <- (1x400x68xf32, 1x1600x68xf32, 1x6400x68xf32)
        combine_2 = [data_6, data_7, data_8]

        # pd_op.concat: (1x8400x68xf32) <- ([1x400x68xf32, 1x1600x68xf32, 1x6400x68xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_2, full_0)
        return full_0, assign_0, assign_1, concat_0, concat_1, concat_2
