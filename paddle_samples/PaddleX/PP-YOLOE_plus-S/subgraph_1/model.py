import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (xi32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("4"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (8x7581xb) <- (8x7581xi32, xi32)
        not_equal_0 = paddle._C_ops.not_equal(data_0, full_0)
        del data_0, full_0

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xi64) <- (8x7581xb, 0xi64)
        sum_0 = paddle._C_ops.sum(not_equal_0, full_int_array_0, None, False)
        del full_int_array_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(sum_0, full_1)
        del full_1, not_equal_0, sum_0

        return greater_than_0
