import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi32) <- (5xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_0, full_int_array_0, full_int_array_1

        # pd_op.cast: (xi64) <- (xi32)
        cast_0 = paddle._C_ops.cast(slice_0, paddle.int64)

        # pd_op.remainder: (xi64) <- (xi64, xi64)
        remainder_0 = paddle._C_ops.remainder(data_1, cast_0)
        del cast_0, data_1

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(remainder_0, full_0)
        del full_0, remainder_0, slice_0

        return not_equal_0
