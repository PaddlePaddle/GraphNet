import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2):
        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_0, data_1, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_2, float("0"), True)

        # pd_op.cast: (xf32) <- (xi64)
        cast_1 = paddle._C_ops.cast(data_2, paddle.float32)

        # pd_op.multiply: (-1xf32) <- (-1xf32, xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, cast_1)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_0, data_0, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_2 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(cast_2, full_2, float("0"), True)

        # pd_op.multiply: (-1xf32) <- (-1xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(scale_1, cast_1)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [multiply_1, multiply_0]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_1,
            split_0,
        ) = meshgrid_0

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_1 = [split_0, split_1]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 2]

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_0)
        return reshape_0, split_0, split_1
