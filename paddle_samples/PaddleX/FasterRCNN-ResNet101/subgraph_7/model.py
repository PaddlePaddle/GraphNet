import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.gather: (2001xi32) <- (1xi32, 2001xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_2, data_0, full_0)
        del data_0, data_2, full_0

        # pd_op.full: (xi32) <- ()
        full_1 = paddle._C_ops.full(
            [], float("0"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (2001xb) <- (2001xi32, xi32)
        equal_0 = paddle._C_ops.equal(data_1, full_1)
        del full_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (2001xi32) <- (2001xi32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            gather_0, full_2, paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2001xi32) <- (2001xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(full_like_0, full_3, float("0"), True)
        del full_3, full_like_0

        # pd_op.where: (2001xi32) <- (2001xb, 2001xi32, 2001xi32)
        where_1 = paddle._C_ops.where(equal_0, scale_0, gather_0)
        del equal_0, gather_0, scale_0

        # pd_op.full: (xi32) <- ()
        full_4 = paddle._C_ops.full(
            [], float("-1"), paddle.int32, paddle.framework._current_expected_place()
        )

        # pd_op.equal: (2001xb) <- (2001xi32, xi32)
        equal_1 = paddle._C_ops.equal(data_1, full_4)
        del data_1, full_4

        # pd_op.full_like: (2001xi32) <- (2001xi32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            where_1, full_2, paddle.int32, paddle.framework._current_expected_place()
        )
        del full_2

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2001xi32) <- (2001xi32, 1xf32)
        scale_1 = paddle._C_ops.scale(full_like_1, full_5, float("0"), True)
        del full_5, full_like_1

        # pd_op.where: (2001xi32) <- (2001xb, 2001xi32, 2001xi32)
        where_0 = paddle._C_ops.where(equal_1, scale_1, where_1)
        del equal_1, scale_1, where_1

        return where_0
