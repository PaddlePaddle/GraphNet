import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (5x2xf32) <- ()
        full_0 = paddle._C_ops.full(
            [5, 2],
            float("0"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (5x2xi64) <- (5x2xf32)
        cast_0 = paddle._C_ops.cast(full_0, paddle.int64)
        del full_0

        # pd_op.full: (5x2xf32) <- ()
        full_1 = paddle._C_ops.full(
            [5, 2],
            float("1"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.cast: (5x2xi64) <- (5x2xf32)
        cast_1 = paddle._C_ops.cast(full_1, paddle.int64)
        del full_1

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([5x2xi64, 5x124xi64]) <- (5x2xi64, 5x124xi64)
        combine_0 = [cast_0, data_0]
        del cast_0, data_0

        # pd_op.concat: (5x126xi64) <- ([5x2xi64, 5x124xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)
        del combine_0

        # builtin.combine: ([5x2xi64, 5x124xi64]) <- (5x2xi64, 5x124xi64)
        combine_1 = [cast_1, data_1]
        del cast_1, data_1

        # pd_op.concat: (5x126xi64) <- ([5x2xi64, 5x124xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)
        del combine_1, full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (5x126xi64) <- (5x126xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(concat_0, full_3, float("0"), True)
        del full_3

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (5x126xb) <- (5x126xi64, xi64)
        equal_0 = paddle._C_ops.equal(scale_0, full_4)
        del full_4

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("-100"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (5x126xb) <- (5x126xb)
        cast_2 = paddle._C_ops.cast(equal_0, paddle.bool)
        del equal_0

        # pd_op.masked_fill: (5x126xi64) <- (5x126xi64, 5x126xb, xi64)
        masked_fill_0 = paddle._C_ops.masked_fill(scale_0, cast_2, full_5)
        del cast_2, full_5, scale_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-3]

        # pd_op.slice: (5x123xi64) <- (5x126xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            concat_0, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del concat_0

        # pd_op.slice: (5x123xi64) <- (5x126xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            concat_1, [1], full_int_array_0, full_int_array_1, [1], []
        )
        del concat_1, full_int_array_0, full_int_array_1

        return slice_0, slice_1, masked_fill_0
