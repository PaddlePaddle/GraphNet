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

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [data_0]
        del data_0

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [scale_0]
        del scale_0

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (xf64) <- (8xf64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_2, [0], stack_0, stack_1, [-1], [0])
        del data_2, stack_0, stack_1

        # pd_op.cast: (xf64) <- (xi64)
        cast_0 = paddle._C_ops.cast(data_1, paddle.float64)

        # pd_op.multiply: (xf64) <- (xf64, xf64)
        multiply_0 = paddle._C_ops.multiply(slice_0, cast_0)
        del cast_0, slice_0

        # pd_op.ceil: (xf64) <- (xf64)
        ceil_0 = paddle._C_ops.ceil(multiply_0)
        del multiply_0

        # pd_op.cast: (xi64) <- (xf64)
        cast_1 = paddle._C_ops.cast(ceil_0, paddle.int64)
        del ceil_0

        # pd_op.minimum: (xi64) <- (xi64, xi64)
        minimum_0 = paddle._C_ops.minimum(data_1, cast_1)
        del cast_1

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(minimum_0, data_1)
        del data_1, minimum_0

        return less_than_0
