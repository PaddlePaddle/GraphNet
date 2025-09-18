import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("100"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(data_0, full_0)
        del data_0, full_0

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_1)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_1, full_1)
        del cast_1, full_1

        # pd_op.argsort: (-1xf32, -1xi64) <- (-1xf32)
        argsort_1, argsort_0 = (lambda x, f: f(x))(
            paddle._C_ops.argsort(data_1, -1, True, False),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None),
        )
        del data_1

        return argsort_0
