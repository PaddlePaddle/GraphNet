import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.shape64: (4xi64) <- (4x-1x-1x-1xf32)
        shape64_0 = paddle._C_ops.shape64(data_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [3]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [4]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_2, full_int_array_3, [1], [0]
        )

        # pd_op.cast: (xf32) <- (xi64)
        cast_0 = paddle._C_ops.cast(data_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(cast_0, full_0, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.5"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xf32) <- (xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(scale_1, full_1, float("0"), True)

        # pd_op.full: (1xi64) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_2, slice_2, full_3, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_4, float("0.5"), True)

        # pd_op.multiply: (-1xf32) <- (-1xf32, xf32)
        multiply_0 = paddle._C_ops.multiply(scale_2, cast_0)

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_2, slice_1, full_3, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_2 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cast_2, full_4, float("0.5"), True)

        # pd_op.multiply: (-1xf32) <- (-1xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(scale_3, cast_0)

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [multiply_1, multiply_0]

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_1,
            split_0,
        ) = meshgrid_0

        # pd_op.subtract: (-1x-1xf32) <- (-1x-1xf32, xf32)
        subtract_0 = paddle._C_ops.subtract(split_0, scale_0)

        # pd_op.subtract: (-1x-1xf32) <- (-1x-1xf32, xf32)
        subtract_1 = paddle._C_ops.subtract(split_1, scale_0)

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, xf32)
        add_0 = paddle._C_ops.add(split_0, scale_0)

        # pd_op.add: (-1x-1xf32) <- (-1x-1xf32, xf32)
        add_1 = paddle._C_ops.add(split_1, scale_0)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32)
        combine_1 = [subtract_0, subtract_1, add_0, add_1]

        # pd_op.stack: (-1x-1x4xf32) <- ([-1x-1xf32, -1x-1xf32, -1x-1xf32, -1x-1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # builtin.combine: ([-1x-1xf32, -1x-1xf32]) <- (-1x-1xf32, -1x-1xf32)
        combine_2 = [split_0, split_1]

        # pd_op.stack: (-1x-1x2xf32) <- ([-1x-1xf32, -1x-1xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-1, 4]

        # pd_op.reshape: (-1x4xf32) <- (-1x-1x4xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(stack_0, full_int_array_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [-1, 2]

        # pd_op.reshape: (-1x2xf32) <- (-1x-1x2xf32, 2xi64)
        reshape_1 = paddle._C_ops.reshape(stack_1, full_int_array_5)
        return (
            reshape_0,
            slice_0,
            slice_1,
            slice_2,
            scale_0,
            split_0,
            split_1,
            stack_0,
            stack_1,
            reshape_1,
        )
