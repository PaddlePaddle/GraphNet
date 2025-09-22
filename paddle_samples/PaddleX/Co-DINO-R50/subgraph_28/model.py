import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (900xb) <- (900xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(data_0, full_1)

        # pd_op.nonzero: (-1x1xi64) <- (900xb)
        nonzero_0 = paddle._C_ops.nonzero(greater_than_0)
        del greater_than_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        squeeze_0 = paddle._C_ops.squeeze(nonzero_0, full_int_array_0)
        del nonzero_0

        # pd_op.equal: (900xb) <- (900xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_1)
        del data_0, full_1

        # pd_op.nonzero: (-1x1xi64) <- (900xb)
        nonzero_1 = paddle._C_ops.nonzero(equal_0)
        del equal_0

        # pd_op.squeeze: (-1xi64) <- (-1x1xi64, 1xi64)
        squeeze_1 = paddle._C_ops.squeeze(nonzero_1, full_int_array_0)
        del full_int_array_0, nonzero_1

        # pd_op.full: (900xi32) <- ()
        full_0 = paddle._C_ops.full(
            [900], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        return squeeze_0, squeeze_1, full_0
