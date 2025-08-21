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
        data_12,
        data_13,
        data_14,
        data_15,
        data_16,
        data_17,
        data_18,
        data_19,
        data_20,
        data_21,
        data_22,
        data_23,
        data_24,
    ):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_9, full_0, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_11, full_1, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            data_13, full_2, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            data_15, full_3, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            data_17, full_4, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_5 = paddle._C_ops.full_like(
            data_19, full_5, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("6"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_6 = paddle._C_ops.full_like(
            data_21, full_6, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("7"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1xi64) <- (-1xi64, 1xf32)
        full_like_7 = paddle._C_ops.full_like(
            data_23, full_7, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64]) <- (-1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64)
        combine_0 = [
            full_like_0,
            full_like_1,
            full_like_2,
            full_like_3,
            full_like_4,
            full_like_5,
            full_like_6,
            full_like_7,
        ]

        # pd_op.concat: (-1xi64) <- ([-1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_0, full_8)

        # builtin.combine: ([-1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64]) <- (-1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64)
        combine_1 = [
            data_9,
            data_11,
            data_13,
            data_15,
            data_17,
            data_19,
            data_21,
            data_23,
        ]

        # pd_op.concat: (-1xi64) <- ([-1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64, -1xi64], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_1, full_8)

        # pd_op.multiply: (-1xi64) <- (-1xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(concat_1, data_0)

        # pd_op.add: (-1xi64) <- (-1xi64, -1xi64)
        add_0 = paddle._C_ops.add(concat_2, multiply_0)

        # pd_op.gather: (-1x1xi32) <- (1x1xi32, -1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_1, data_10, full_8)

        # pd_op.gather: (-1x1xi32) <- (1x1xi32, -1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(data_2, data_12, full_8)

        # pd_op.gather: (-1x1xi32) <- (1x1xi32, -1xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(data_3, data_14, full_8)

        # pd_op.gather: (-1x1xi32) <- (1x1xi32, -1xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(data_4, data_16, full_8)

        # pd_op.gather: (-1x1xi32) <- (-1x1xi32, -1xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(data_5, data_18, full_8)

        # pd_op.gather: (-1x1xi32) <- (1x1xi32, -1xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(data_6, data_20, full_8)

        # pd_op.gather: (-1x1xi32) <- (-1x1xi32, -1xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(data_7, data_22, full_8)

        # pd_op.gather: (-1x1xi32) <- (-1x1xi32, -1xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(data_8, data_24, full_8)

        # builtin.combine: ([-1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32]) <- (-1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32)
        combine_2 = [
            gather_0,
            gather_1,
            gather_2,
            gather_3,
            gather_4,
            gather_5,
            gather_6,
            gather_7,
        ]

        # pd_op.concat: (-1x1xi32) <- ([-1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32, -1x1xi32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_8)
        return add_0, concat_0
