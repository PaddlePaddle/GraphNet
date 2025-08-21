import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5):
        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1xi64) <- (1xi64, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            data_2, full_0, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (1xi64) <- (1xi64, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            data_4, full_1, paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1xi64, 1xi64]) <- (1xi64, 1xi64)
        combine_0 = [full_like_0, full_like_1]

        # pd_op.concat: (2xi64) <- ([1xi64, 1xi64], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_0, full_2)

        # builtin.combine: ([1xi64, 1xi64]) <- (1xi64, 1xi64)
        combine_1 = [data_2, data_4]

        # pd_op.concat: (2xi64) <- ([1xi64, 1xi64], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_1, full_2)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("300"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (2xi64) <- (2xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(concat_1, full_3, float("0"), True)

        # pd_op.add: (2xi64) <- (2xi64, 2xi64)
        add_0 = paddle._C_ops.add(concat_2, scale_0)

        # pd_op.gather: (1x1xi32) <- (1x1xi32, 1xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(data_0, data_3, full_2)

        # pd_op.gather: (1x1xi32) <- (1x1xi32, 1xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(data_1, data_5, full_2)

        # builtin.combine: ([1x1xi32, 1x1xi32]) <- (1x1xi32, 1x1xi32)
        combine_2 = [gather_0, gather_1]

        # pd_op.concat: (2x1xi32) <- ([1x1xi32, 1x1xi32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_2, full_2)
        return add_0, concat_0
