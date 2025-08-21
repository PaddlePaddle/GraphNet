import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (xf32) <- (xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(data_0, full_0, full_1)

        # pd_op.divide: (xf32) <- (xf32, xf32)
        divide_0 = paddle._C_ops.divide(data_1, clip_0)

        # pd_op.full: (xi32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("2"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (1x8400xb) <- (1x8400xi32, xi32)
        not_equal_0 = paddle._C_ops.not_equal(data_2, full_2)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.sum: (xi64) <- (1x8400xb, 0xi64)
        sum_0 = paddle._C_ops.sum(not_equal_0, full_int_array_0, None, False)

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(sum_0, full_3)
        return greater_than_0, not_equal_0, clip_0, divide_0
