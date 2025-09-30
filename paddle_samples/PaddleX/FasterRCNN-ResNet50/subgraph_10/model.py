import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (512x4xf32) <- (-1x4xf32, 512xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(data_2, data_1, full_0)
        del data_2

        # pd_op.gather: (512xi64) <- (-1xi64, 512xi32, 1xi32)
        gather_2 = paddle._C_ops.gather(data_3, data_1, full_0)
        del data_3

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(data_0, full_1)
        del data_0

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_1)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_1, full_1)
        del cast_1, full_1

        # pd_op.gather: (512x4xf32) <- (-1x4xf32, 512xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(data_4, gather_2, full_0)
        del data_4, full_0

        # pd_op.shape64: (1xi64) <- (512xi32)
        shape64_0 = paddle._C_ops.shape64(data_1)
        del data_1, gather_2

        return gather_0, gather_1, shape64_0
