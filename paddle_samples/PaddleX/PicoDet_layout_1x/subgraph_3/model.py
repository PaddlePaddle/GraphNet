import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_1 = paddle._C_ops.equal(data_1, data_0)

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_1, paddle.int64)
        del equal_1

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_0)
        del cast_0

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(data_1, data_0)
        del data_0, data_1

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_2 = paddle._C_ops.equal(multiply_0, full_0)
        del multiply_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(equal_2, paddle.int64)
        del equal_2

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_1 = paddle._C_ops.not_equal(cast_1, full_0)
        del cast_1

        # pd_op.cast: (xi64) <- (xb)
        cast_2 = paddle._C_ops.cast(not_equal_1, paddle.int64)
        del not_equal_1

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_2, full_0)
        del cast_2, full_0

        return equal_0
