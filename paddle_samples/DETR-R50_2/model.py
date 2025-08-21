import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("4"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_0 = [full_0, full_1, full_1, data_0, data_1]

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.reshape: (4x1x1x-1x-1xf32) <- (4x1x1x-1xf32, 5xi64)
        reshape_1 = paddle._C_ops.reshape(data_3, stack_0)

        # pd_op.transpose: (4x256x-1xf32) <- (4x-1x256xf32)
        transpose_0 = paddle._C_ops.transpose(data_2, [0, 2, 1])

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("256"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_0, full_2, data_0, data_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.reshape: (4x256x-1x-1xf32) <- (4x256x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, stack_1)
        return transpose_0, reshape_0, reshape_1
