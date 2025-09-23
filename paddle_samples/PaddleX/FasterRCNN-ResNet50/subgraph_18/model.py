import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full: (xi32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("-1"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (-1xb) <- (-1xi32, xi32)
        not_equal_0 = paddle._C_ops.not_equal(data_1, full_0)
        del full_0

        # pd_op.cast: (xi32) <- (xi64)
        cast_0 = paddle._C_ops.cast(data_0, paddle.int32)
        del data_0

        # pd_op.not_equal: (-1xb) <- (-1xi32, xi32)
        not_equal_1 = paddle._C_ops.not_equal(data_1, cast_0)

        # pd_op.logical_and: (-1xb) <- (-1xb, -1xb)
        logical_and_0 = paddle._C_ops.logical_and(not_equal_0, not_equal_1)
        del not_equal_0, not_equal_1

        # pd_op.nonzero: (-1x1xi64) <- (-1xb)
        nonzero_1 = paddle._C_ops.nonzero(logical_and_0)
        del logical_and_0

        # pd_op.equal: (-1xb) <- (-1xi32, xi32)
        equal_0 = paddle._C_ops.equal(data_1, cast_0)
        del cast_0, data_1

        # pd_op.nonzero: (-1x1xi64) <- (-1xb)
        nonzero_0 = paddle._C_ops.nonzero(equal_0)
        del equal_0

        # pd_op.numel: (xi64) <- (-1x1xi64)
        numel_0 = paddle._C_ops.numel(nonzero_1)

        # pd_op.less_than: (xb) <- (xi64, xi64)
        less_than_0 = paddle._C_ops.less_than(numel_0, numel_0)
        del nonzero_1

        return numel_0, nonzero_0
