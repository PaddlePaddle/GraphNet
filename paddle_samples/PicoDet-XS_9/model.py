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
            [1], float("56"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.float64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (56xi64) <- (1xf64, 1xf64, 1xf64)
        arange_0 = paddle.arange(full_1, full_2, full_3, dtype="int64")

        # pd_op.cast: (56xf32) <- (56xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (56xf32) <- (56xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_4, float("0.5"), True)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (56xf32) <- (56xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_5, float("0"), True)

        # builtin.combine: ([56xf32, 56xf32]) <- (56xf32, 56xf32)
        combine_0 = [scale_1, scale_1]

        # pd_op.meshgrid: ([56x56xf32, 56x56xf32]) <- ([56xf32, 56xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (56x56xf32, 56x56xf32) <- ([56x56xf32, 56x56xf32])
        (
            split_1,
            split_0,
        ) = meshgrid_0

        # pd_op.scale: (56x56xf32) <- (56x56xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(split_0, full_4, float("-20"), True)

        # pd_op.scale: (56x56xf32) <- (56x56xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(split_1, full_4, float("-20"), True)

        # pd_op.scale: (56x56xf32) <- (56x56xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_0, full_4, float("20"), True)

        # pd_op.scale: (56x56xf32) <- (56x56xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_1, full_4, float("20"), True)

        # builtin.combine: ([56x56xf32, 56x56xf32, 56x56xf32, 56x56xf32]) <- (56x56xf32, 56x56xf32, 56x56xf32, 56x56xf32)
        combine_1 = [scale_2, scale_3, scale_4, scale_5]

        # pd_op.stack: (56x56x4xf32) <- ([56x56xf32, 56x56xf32, 56x56xf32, 56x56xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # builtin.combine: ([56x56xf32, 56x56xf32]) <- (56x56xf32, 56x56xf32)
        combine_2 = [split_0, split_1]

        # pd_op.stack: (56x56x2xf32) <- ([56x56xf32, 56x56xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (3136x4xf32) <- (56x56x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (3136x2xf32) <- (56x56x2xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_1, full_int_array_1)

        # pd_op.full: (3136x1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [3136, 1],
            float("8"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        return split_0, split_1, stack_0, stack_1, reshape_0, reshape_1, full_0
