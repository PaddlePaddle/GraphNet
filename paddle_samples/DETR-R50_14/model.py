import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_6,
        data_7,
        data_8,
        data_9,
        data_10,
        data_11,
    ):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_4, full_0, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_6, full_1, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_8, full_2, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            data_10, full_3, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1xi64, -1xi64, -1xi64, -1xi64]) <- (-1xi64, -1xi64, -1xi64, -1xi64)
        combine_0 = [full_like_0, full_like_1, full_like_2, full_like_3]

        # pd_op.concat: (-1xi64) <- ([-1xi64, -1xi64, -1xi64, -1xi64], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_4)

        # builtin.combine: ([-1xi64, -1xi64, -1xi64, -1xi64]) <- (-1xi64, -1xi64, -1xi64, -1xi64)
        combine_1 = [data_4, data_6, data_8, data_10]

        # pd_op.concat: (-1xi64) <- ([-1xi64, -1xi64, -1xi64, -1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_4)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("100"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xi64) <- (-1xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(concat_0, full_5, float("0"), True)

        # pd_op.add: (-1xi64) <- (-1xi64, -1xi64)
        add_0 = paddle._C_ops.add(concat_1, scale_0)

        # pd_op.gather: (-1x1xi32) <- (-1x1xi32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_0, data_5, full_4)

        # pd_op.gather: (-1x1xf32) <- (-1x1xf32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(data_1, data_7, full_4)

        # pd_op.gather: (-1x1xi32) <- (-1x1xi32, -1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(data_2, data_9, full_4)

        # pd_op.gather: (-1x1xi32) <- (-1x1xi32, -1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(data_3, data_11, full_4)
        return gather_0, gather_1, gather_2, gather_3, add_0
