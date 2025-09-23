import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("10"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("7"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_0 = [full_0, full_1, full_2, full_2, full_2, data_0]
        del full_1, full_2

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x10x7x7x7x-1xf32) <- (140x7x7x-1xf32, 6xi64)
        reshape_1 = paddle._C_ops.reshape(data_1, stack_0)
        del data_1, stack_0

        # pd_op.transpose: (-1x10x7x7x7x-1xf32) <- (-1x10x7x7x7x-1xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_1, [0, 1, 3, 2, 4, 5])
        del reshape_1

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("70"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("49"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_0, full_3, full_4, data_0]
        del data_0, full_0, full_3, full_4

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x70x49x-1xf32) <- (-1x10x7x7x7x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, stack_1)
        del stack_1, transpose_0

        return reshape_0
