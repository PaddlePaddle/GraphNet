import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(data_1, [0, 2, 1])
        del data_1

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("3"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("80"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, data_0, full_1, full_2]
        del data_0, full_0, full_1, full_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x-1x3x80xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, stack_0)
        del stack_0, transpose_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 2]

        # pd_op.pool2d: (-1x-1x1x40xf32) <- (-1x-1x3x80xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(
            reshape_0,
            full_int_array_0,
            [3, 2],
            [0, 0],
            False,
            True,
            "NCHW",
            "avg",
            False,
            False,
            "EXPLICIT",
        )
        del full_int_array_0, reshape_0

        return pool2d_0
