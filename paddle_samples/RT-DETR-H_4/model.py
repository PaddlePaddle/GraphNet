import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.shape64: (4xi64) <- (1x-1x8x32xf32)
        shape64_0 = paddle._C_ops.shape64(data_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_1

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_2

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_2

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_3

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_1, [0], full_int_array_3, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            slice_1, [0], full_int_array_3, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            slice_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_1 = paddle._C_ops.multiply(slice_2, slice_3)

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_1, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            slice_4, [0], full_int_array_3, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            slice_4, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_2 = paddle._C_ops.multiply(slice_5, slice_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_4

        # pd_op.slice: (2xi64) <- (3x2xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_1, [0], full_int_array_2, full_int_array_4, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            slice_7, [0], full_int_array_3, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            slice_7, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_3 = paddle._C_ops.multiply(slice_8, slice_9)

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [multiply_1, multiply_2, multiply_3]

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_0, 0)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x-1x8x32xf32, 3xi64, 1xi32)
        split_3 = paddle._C_ops.split(data_0, stack_1, full_0)

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_0,
            split_1,
            split_2,
        ) = split_3

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x500x8x3x4x2xf32) <- (1x500x8x3x4x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(data_2, full_1, float("0"), True)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x500x8x3x4x2xf32) <- (1x500x8x3x4x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(scale_1, full_2, float("-1"), True)

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(split_0, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_3, full_4, slice_2, slice_3]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_1, 0)

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(transpose_0, stack_2)

        # pd_op.slice: (1x500x8x4x2xf32) <- (1x500x8x3x4x2xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            scale_0, [3], full_int_array_3, full_int_array_1, [1], [3]
        )

        # pd_op.transpose: (1x8x500x4x2xf32) <- (1x500x8x4x2xf32)
        transpose_1 = paddle._C_ops.transpose(slice_10, [0, 2, 1, 3, 4])

        # pd_op.flatten: (8x500x4x2xf32) <- (1x8x500x4x2xf32)
        flatten_0 = paddle._C_ops.flatten(transpose_1, 0, 1)

        # pd_op.grid_sample: (8x32x500x4xf32) <- (8x32x-1x-1xf32, 8x500x4x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(
            reshape_0, flatten_0, "bilinear", "zeros", False
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_5 = paddle._C_ops.flatten(split_1, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_5, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [full_3, full_4, slice_5, slice_6]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_2, 0)

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_2, stack_3)

        # pd_op.slice: (1x500x8x4x2xf32) <- (1x500x8x3x4x2xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            scale_0, [3], full_int_array_1, full_int_array_2, [1], [3]
        )

        # pd_op.transpose: (1x8x500x4x2xf32) <- (1x500x8x4x2xf32)
        transpose_3 = paddle._C_ops.transpose(slice_11, [0, 2, 1, 3, 4])

        # pd_op.flatten: (8x500x4x2xf32) <- (1x8x500x4x2xf32)
        flatten_1 = paddle._C_ops.flatten(transpose_3, 0, 1)

        # pd_op.grid_sample: (8x32x500x4xf32) <- (8x32x-1x-1xf32, 8x500x4x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(
            reshape_1, flatten_1, "bilinear", "zeros", False
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_6 = paddle._C_ops.flatten(split_2, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [full_3, full_4, slice_8, slice_9]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_3, 0)

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(transpose_4, stack_4)

        # pd_op.slice: (1x500x8x4x2xf32) <- (1x500x8x3x4x2xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            scale_0, [3], full_int_array_2, full_int_array_4, [1], [3]
        )

        # pd_op.transpose: (1x8x500x4x2xf32) <- (1x500x8x4x2xf32)
        transpose_5 = paddle._C_ops.transpose(slice_12, [0, 2, 1, 3, 4])

        # pd_op.flatten: (8x500x4x2xf32) <- (1x8x500x4x2xf32)
        flatten_2 = paddle._C_ops.flatten(transpose_5, 0, 1)

        # pd_op.grid_sample: (8x32x500x4xf32) <- (8x32x-1x-1xf32, 8x500x4x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(
            reshape_2, flatten_2, "bilinear", "zeros", False
        )

        # pd_op.transpose: (1x8x500x3x4xf32) <- (1x500x8x3x4xf32)
        transpose_6 = paddle._C_ops.transpose(data_3, [0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [8, 1, 500, 12]

        # pd_op.reshape: (8x1x500x12xf32) <- (1x8x500x3x4xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(transpose_6, full_int_array_5)

        # builtin.combine: ([8x32x500x4xf32, 8x32x500x4xf32, 8x32x500x4xf32]) <- (8x32x500x4xf32, 8x32x500x4xf32, 8x32x500x4xf32)
        combine_4 = [grid_sample_0, grid_sample_1, grid_sample_2]

        # pd_op.stack: (8x32x500x3x4xf32) <- ([8x32x500x4xf32, 8x32x500x4xf32, 8x32x500x4xf32])
        stack_0 = paddle._C_ops.stack(combine_4, -2)

        # pd_op.flatten: (8x32x500x12xf32) <- (8x32x500x3x4xf32)
        flatten_3 = paddle._C_ops.flatten(stack_0, 3, 4)

        # pd_op.multiply: (8x32x500x12xf32) <- (8x32x500x12xf32, 8x1x500x12xf32)
        multiply_0 = paddle._C_ops.multiply(flatten_3, reshape_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.sum: (8x32x500xf32) <- (8x32x500x12xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_0, full_int_array_0, None, False)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [1, 256, 500]

        # pd_op.reshape: (1x256x500xf32) <- (8x32x500xf32, 3xi64)
        reshape_4 = paddle._C_ops.reshape(sum_0, full_int_array_6)

        # pd_op.transpose: (1x500x256xf32) <- (1x256x500xf32)
        transpose_7 = paddle._C_ops.transpose(reshape_4, [0, 2, 1])
        return (
            full_0,
            split_0,
            split_1,
            split_2,
            full_1,
            full_2,
            scale_0,
            transpose_0,
            reshape_0,
            assign_0,
            assign_1,
            transpose_1,
            flatten_0,
            grid_sample_0,
            transpose_2,
            reshape_1,
            assign_2,
            assign_3,
            transpose_3,
            flatten_1,
            grid_sample_1,
            transpose_4,
            reshape_2,
            assign_4,
            assign_5,
            transpose_5,
            flatten_2,
            grid_sample_2,
            transpose_6,
            reshape_3,
            stack_0,
            flatten_3,
            multiply_0,
            full_int_array_0,
            sum_0,
            transpose_7,
        )
