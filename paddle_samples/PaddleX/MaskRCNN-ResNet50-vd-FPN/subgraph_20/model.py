import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (xi32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (512xb) <- (512xi32, xi32)
        not_equal_0 = paddle._C_ops.not_equal(data_0, full_0)
        del full_0

        # pd_op.full: (xi32) <- ()
        full_1 = paddle._C_ops.full(
            [], float("2"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (512xb) <- (512xi32, xi32)
        not_equal_1 = paddle._C_ops.not_equal(data_0, full_1)
        del data_0, full_1

        # pd_op.logical_and: (512xb) <- (512xb, 512xb)
        logical_and_0 = paddle._C_ops.logical_and(not_equal_0, not_equal_1)
        del not_equal_0, not_equal_1

        # pd_op.nonzero: (-1x1xi64) <- (512xb)
        nonzero_0 = paddle._C_ops.nonzero(logical_and_0)
        del logical_and_0

        # pd_op.numel: (xi64) <- (-1x1xi64)
        numel_0 = paddle._C_ops.numel(nonzero_0)

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(numel_0, full_2)
        del full_2, nonzero_0, numel_0

        return equal_0
