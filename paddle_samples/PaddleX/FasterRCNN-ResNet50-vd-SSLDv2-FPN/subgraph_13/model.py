import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(data_0, full_0)
        del data_0

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.int64)
        del equal_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_0)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_1 = paddle._C_ops.equal(cast_1, full_0)
        del cast_1, full_0

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1xi32, -1xi32]) <- (-1xi32, -1xi32)
        combine_0 = [data_1, data_2]
        del data_1, data_2

        # pd_op.concat: (-1xi32) <- ([-1xi32, -1xi32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)
        del combine_0

        # pd_op.gather: (-1xi32) <- (-1xi32, -1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(data_3, concat_0, full_1)
        del concat_0, data_3, full_1

        return gather_0
