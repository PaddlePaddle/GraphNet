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

        # pd_op.slice: (xi32) <- (5xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [0], stack_0, stack_1, [-1], [0])
        del data_1, stack_0, stack_1

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)

        # pd_op.remainder: (xi64) <- (xi64, xi64)
        remainder_0 = paddle._C_ops.remainder(data_2, cast_0)
        del cast_0, data_2

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(remainder_0, full_1)
        del full_1, remainder_0, slice_0

        return not_equal_0
