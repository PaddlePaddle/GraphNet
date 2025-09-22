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

        # pd_op.gather: (-1xi32) <- (-1xi32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_3, data_1, full_1)
        del data_1, data_3, full_1

        # pd_op.full: (xi32) <- ()
        full_2 = paddle._C_ops.full(
            [], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (-1xb) <- (-1xi32, xi32)
        equal_2 = paddle._C_ops.equal(data_2, full_2)
        del full_2

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi32) <- (-1xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            gather_0, full_3, paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xi32) <- (-1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(full_like_0, full_4, float("0"), True)
        del full_4, full_like_0

        # pd_op.where: (-1xi32) <- (-1xb, -1xi32, -1xi32)
        where_1 = paddle._C_ops.where(equal_2, scale_0, gather_0)
        del equal_2, gather_0, scale_0

        # pd_op.full: (xi32) <- ()
        full_5 = paddle._C_ops.full(
            [], float("-1"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (-1xb) <- (-1xi32, xi32)
        equal_3 = paddle._C_ops.equal(data_2, full_5)
        del data_2, full_5

        # pd_op.full_like: (-1xi32) <- (-1xi32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            where_1, full_3, paddle.int32, paddle.framework._current_expected_place()
        )
        del full_3

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xi32) <- (-1xi32, 1xf32)
        scale_1 = paddle._C_ops.scale(full_like_1, full_6, float("0"), True)
        del full_6, full_like_1

        # pd_op.where: (-1xi32) <- (-1xb, -1xi32, -1xi32)
        where_0 = paddle._C_ops.where(equal_3, scale_1, where_1)
        del equal_3, scale_1, where_1

        return where_0
