import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.shape64: (2xi64) <- (-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("10"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(slice_0, full_0)
        return slice_0
