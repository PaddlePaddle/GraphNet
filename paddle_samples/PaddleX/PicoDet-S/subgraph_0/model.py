import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([8x1xf32, 8x1xf32]) <- (8x2xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(data_1, 2, full_0)
        del data_1, full_0

        # builtin.split: (8x1xf32, 8x1xf32) <- ([8x1xf32, 8x1xf32])
        (
            split_0,
            split_1,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([8x1xf32, 8x1xf32, 8x1xf32, 8x1xf32]) <- (8x1xf32, 8x1xf32, 8x1xf32, 8x1xf32)
        combine_0 = [split_1, split_0, split_1, split_0]
        del split_0, split_1

        # pd_op.concat: (8x4xf32) <- ([8x1xf32, 8x1xf32, 8x1xf32, 8x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)
        del combine_0, full_1

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_0 = [-1, 1, 4]

        # pd_op.reshape: (8x1x4xf32) <- (8x4xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(concat_0, full_int_array_0)
        del concat_0, full_int_array_0

        # pd_op.divide: (8x2125x4xf32) <- (8x2125x4xf32, 8x1x4xf32)
        divide_0 = paddle._C_ops.divide(data_0, reshape_0)
        del data_0, reshape_0

        return divide_0
