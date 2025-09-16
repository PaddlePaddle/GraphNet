import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(data_0, full_0, float("1"), True)
        del full_0

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2.14748e+09"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("2.14748e+09"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_0 = [data_0, data_2]
        del data_0, data_2

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_1 = [scale_0, full_2]
        del full_2, scale_0

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.set_value_: (8x26x1x40x1xf32) <- (8x26x1x40x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__0 = paddle._C_ops.set_value_(
            data_1,
            stack_0,
            stack_1,
            full_int_array_0,
            [0, 3],
            [0],
            [],
            [1],
            [float("-inf")],
        )
        del data_1, full_int_array_0, stack_0, stack_1

        return set_value__0
