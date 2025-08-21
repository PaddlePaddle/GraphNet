import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        # pd_op.full: (1xf64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("18"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (18xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_1, full_2, full_3, dtype="int64")

        # pd_op.cast: (18xf32) <- (18xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (18xf32) <- (18xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_4, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("32"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (18xf32) <- (18xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_5, float("0"), True)

        # builtin.combine: ([18xf32, 18xf32]) <- (18xf32, 18xf32)
        combine_0 = [scale_1, scale_1]

        # pd_op.meshgrid: ([18x18xf32, 18x18xf32]) <- ([18xf32, 18xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (18x18xf32, 18x18xf32) <- ([18x18xf32, 18x18xf32])
        (
            split_1,
            split_0,
        ) = meshgrid_0

        # pd_op.scale: (18x18xf32) <- (18x18xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(split_0, full_4, float("-80"), True)

        # pd_op.scale: (18x18xf32) <- (18x18xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_1, full_4, float("-80"), True)

        # pd_op.scale: (18x18xf32) <- (18x18xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_0, full_4, float("80"), True)

        # pd_op.scale: (18x18xf32) <- (18x18xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_1, full_4, float("80"), True)

        # builtin.combine: ([18x18xf32, 18x18xf32, 18x18xf32, 18x18xf32]) <- (18x18xf32, 18x18xf32, 18x18xf32, 18x18xf32)
        combine_1 = [scale_2, scale_3, scale_4, scale_5]

        # pd_op.stack: (18x18x4xf32) <- ([18x18xf32, 18x18xf32, 18x18xf32, 18x18xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # builtin.combine: ([18x18xf32, 18x18xf32]) <- (18x18xf32, 18x18xf32)
        combine_2 = [split_0, split_1]

        # pd_op.stack: (18x18x2xf32) <- ([18x18xf32, 18x18xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (324x4xf32) <- (18x18x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (324x2xf32) <- (18x18x2xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_1, full_int_array_1)

        # pd_op.full: (324x1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [324, 1],
            float("32"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        return split_0, split_1, stack_0, stack_1, reshape_0, reshape_1, full_0
