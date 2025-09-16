import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [full_1, data_1]
        del data_1, full_1

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 2147483647]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.set_value_: (8x26x1x40x1xf32) <- (8x26x1x40x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__0 = paddle._C_ops.set_value_(
            data_0,
            stack_0,
            full_int_array_0,
            full_int_array_1,
            [0, 3],
            [0],
            [],
            [1],
            [float("-inf")],
        )
        del data_0, full_int_array_0, full_int_array_1, stack_0

        return set_value__0
