import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        parameter_0,
        parameter_1,
        parameter_2,
        parameter_3,
        parameter_4,
        parameter_5,
        parameter_6,
        parameter_7,
        data_0,
        data_1,
        data_2,
        data_3,
        data_4,
    ):
        # pd_op.matmul: (1x45640x256xf32) <- (1x45640x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(data_0, parameter_7, False, False)
        del data_0, parameter_7

        # pd_op.add: (1x45640x256xf32) <- (1x45640x256xf32, 256xf32)
        add_1 = paddle._C_ops.add(matmul_0, parameter_6)
        del parameter_6

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.unsqueeze: (1x45640x1xf32) <- (1x45640xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_0)
        del data_1

        # pd_op.multiply: (1x45640x256xf32) <- (1x45640x256xf32, 1x45640x1xf32)
        multiply_0 = paddle._C_ops.multiply(add_1, unsqueeze_0)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 45640, 8, 32]

        # pd_op.reshape: (1x45640x8x32xf32) <- (1x45640x256xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(multiply_0, full_int_array_1)
        del full_int_array_1

        # pd_op.matmul: (1x45640x320xf32) <- (1x45640x256xf32, 256x320xf32)
        matmul_1 = paddle._C_ops.matmul(data_2, parameter_5, False, False)
        del parameter_5

        # pd_op.add: (1x45640x320xf32) <- (1x45640x320xf32, 320xf32)
        add_2 = paddle._C_ops.add(matmul_1, parameter_4)
        del parameter_4

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_2 = [1, 45640, 8, 5, 4, 2]

        # pd_op.reshape: (1x45640x8x5x4x2xf32) <- (1x45640x320xf32, 6xi64)
        reshape_1 = paddle._C_ops.reshape(add_2, full_int_array_2)
        del full_int_array_2

        # pd_op.matmul: (1x45640x160xf32) <- (1x45640x256xf32, 256x160xf32)
        matmul_2 = paddle._C_ops.matmul(data_2, parameter_3, False, False)
        del data_2, parameter_3

        # pd_op.add: (1x45640x160xf32) <- (1x45640x160xf32, 160xf32)
        add_3 = paddle._C_ops.add(matmul_2, parameter_2)
        del parameter_2

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 45640, 8, 20]

        # pd_op.reshape: (1x45640x8x20xf32) <- (1x45640x160xf32, 4xi64)
        reshape_2 = paddle._C_ops.reshape(add_3, full_int_array_3)
        del full_int_array_3

        # pd_op.softmax: (1x45640x8x20xf32) <- (1x45640x8x20xf32)
        softmax_0 = paddle._C_ops.softmax(reshape_2, -1)
        del reshape_2

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [1, 45640, 8, 5, 4]

        # pd_op.reshape: (1x45640x8x5x4xf32) <- (1x45640x8x20xf32, 5xi64)
        reshape_3 = paddle._C_ops.reshape(softmax_0, full_int_array_4)
        del full_int_array_4

        # pd_op.flip: (5x2xi64) <- (5x2xi64)
        flip_0 = paddle._C_ops.flip(data_4, [1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_5 = [1, 1, 1, 5, 1, 2]

        # pd_op.reshape: (1x1x1x5x1x2xi64) <- (5x2xi64, 6xi64)
        reshape_4 = paddle._C_ops.reshape(flip_0, full_int_array_5)
        del flip_0, full_int_array_5

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_6 = [1, 45640, 1, 5, 1, 2]

        # pd_op.reshape: (1x45640x1x5x1x2xf32) <- (1x45640x5x2xf32, 6xi64)
        reshape_5 = paddle._C_ops.reshape(data_3, full_int_array_6)
        del data_3, full_int_array_6

        # pd_op.cast: (1x1x1x5x1x2xf32) <- (1x1x1x5x1x2xi64)
        cast_0 = paddle._C_ops.cast(reshape_4, paddle.float32)
        del reshape_4

        # pd_op.divide: (1x45640x8x5x4x2xf32) <- (1x45640x8x5x4x2xf32, 1x1x1x5x1x2xf32)
        divide_0 = paddle._C_ops.divide(reshape_1, cast_0)

        # pd_op.add: (1x45640x8x5x4x2xf32) <- (1x45640x1x5x1x2xf32, 1x45640x8x5x4x2xf32)
        add_4 = paddle._C_ops.add(reshape_5, divide_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [0]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_8

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_8

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_4, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            slice_0, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_4 = full_int_array_9

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_5 = full_int_array_9

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            slice_0, [0], full_int_array_8, full_int_array_9, [1], [0]
        )
        del slice_0

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_1 = paddle._C_ops.multiply(slice_1, slice_2)

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_4, [0], full_int_array_8, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            slice_3, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            slice_3, [0], full_int_array_8, full_int_array_9, [1], [0]
        )
        del slice_3

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_2 = paddle._C_ops.multiply(slice_4, slice_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [3]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_6 = full_int_array_10

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_7 = full_int_array_10

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_4, [0], full_int_array_9, full_int_array_10, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            slice_6, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            slice_6, [0], full_int_array_8, full_int_array_9, [1], [0]
        )
        del slice_6

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_3 = paddle._C_ops.multiply(slice_7, slice_8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [4]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_8 = full_int_array_11

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_9 = full_int_array_11

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_4, [0], full_int_array_10, full_int_array_11, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            slice_9, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            slice_9, [0], full_int_array_8, full_int_array_9, [1], [0]
        )
        del slice_9

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_4 = paddle._C_ops.multiply(slice_10, slice_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [5]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_10 = full_int_array_12

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            data_4, [0], full_int_array_11, full_int_array_12, [1], [0]
        )
        del data_4

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            slice_12, [0], full_int_array_7, full_int_array_8, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            slice_12, [0], full_int_array_8, full_int_array_9, [1], [0]
        )
        del slice_12

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_5 = paddle._C_ops.multiply(slice_13, slice_14)

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64)
        combine_0 = [multiply_1, multiply_2, multiply_3, multiply_4, multiply_5]
        del multiply_1, multiply_2, multiply_3, multiply_4, multiply_5

        # pd_op.stack: (5xi64) <- ([xi64, xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split: ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32]) <- (1x45640x8x32xf32, 5xi64, 1xi32)
        split_0 = paddle._C_ops.split(reshape_0, stack_0, full_0)
        del reshape_0, stack_0

        # builtin.split: (-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32) <- ([-1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32, -1x-1x-1x-1xf32])
        (
            split_1,
            split_2,
            split_3,
            split_4,
            split_5,
        ) = split_0
        del split_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x45640x8x5x4x2xf32) <- (1x45640x8x5x4x2xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(add_4, full_1, float("0"), True)
        del add_4

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x45640x8x5x4x2xf32) <- (1x45640x8x5x4x2xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_2, float("-1"), True)
        del scale_0

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(split_1, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        del flatten_0

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("8"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("32"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_3, full_4, slice_1, slice_2]
        del slice_1, slice_2

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(transpose_0, stack_1)
        del stack_1

        # pd_op.slice: (1x45640x8x4x2xf32) <- (1x45640x8x5x4x2xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            scale_1, [3], full_int_array_7, full_int_array_8, [1], [3]
        )
        del full_int_array_7

        # pd_op.transpose: (1x8x45640x4x2xf32) <- (1x45640x8x4x2xf32)
        transpose_1 = paddle._C_ops.transpose(slice_15, [0, 2, 1, 3, 4])
        del slice_15

        # pd_op.flatten: (8x45640x4x2xf32) <- (1x8x45640x4x2xf32)
        flatten_1 = paddle._C_ops.flatten(transpose_1, 0, 1)

        # pd_op.grid_sample: (8x32x45640x4xf32) <- (8x32x-1x-1xf32, 8x45640x4x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(
            reshape_6, flatten_1, "bilinear", "zeros", False
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_2 = paddle._C_ops.flatten(split_2, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_2 = paddle._C_ops.transpose(flatten_2, [0, 2, 1])
        del flatten_2

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_2 = [full_3, full_4, slice_4, slice_5]
        del slice_4, slice_5

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(transpose_2, stack_2)
        del stack_2

        # pd_op.slice: (1x45640x8x4x2xf32) <- (1x45640x8x5x4x2xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            scale_1, [3], full_int_array_8, full_int_array_9, [1], [3]
        )
        del full_int_array_8

        # pd_op.transpose: (1x8x45640x4x2xf32) <- (1x45640x8x4x2xf32)
        transpose_3 = paddle._C_ops.transpose(slice_16, [0, 2, 1, 3, 4])
        del slice_16

        # pd_op.flatten: (8x45640x4x2xf32) <- (1x8x45640x4x2xf32)
        flatten_3 = paddle._C_ops.flatten(transpose_3, 0, 1)

        # pd_op.grid_sample: (8x32x45640x4xf32) <- (8x32x-1x-1xf32, 8x45640x4x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(
            reshape_7, flatten_3, "bilinear", "zeros", False
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(split_3, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_4 = paddle._C_ops.transpose(flatten_4, [0, 2, 1])
        del flatten_4

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_3 = [full_3, full_4, slice_7, slice_8]
        del slice_7, slice_8

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_3, 0)
        del combine_3

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_8 = paddle._C_ops.reshape(transpose_4, stack_3)
        del stack_3

        # pd_op.slice: (1x45640x8x4x2xf32) <- (1x45640x8x5x4x2xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            scale_1, [3], full_int_array_9, full_int_array_10, [1], [3]
        )
        del full_int_array_9

        # pd_op.transpose: (1x8x45640x4x2xf32) <- (1x45640x8x4x2xf32)
        transpose_5 = paddle._C_ops.transpose(slice_17, [0, 2, 1, 3, 4])
        del slice_17

        # pd_op.flatten: (8x45640x4x2xf32) <- (1x8x45640x4x2xf32)
        flatten_5 = paddle._C_ops.flatten(transpose_5, 0, 1)

        # pd_op.grid_sample: (8x32x45640x4xf32) <- (8x32x-1x-1xf32, 8x45640x4x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(
            reshape_8, flatten_5, "bilinear", "zeros", False
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_6 = paddle._C_ops.flatten(split_4, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_6 = paddle._C_ops.transpose(flatten_6, [0, 2, 1])
        del flatten_6

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_4 = [full_3, full_4, slice_10, slice_11]
        del slice_10, slice_11

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(transpose_6, stack_4)
        del stack_4

        # pd_op.slice: (1x45640x8x4x2xf32) <- (1x45640x8x5x4x2xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            scale_1, [3], full_int_array_10, full_int_array_11, [1], [3]
        )
        del full_int_array_10

        # pd_op.transpose: (1x8x45640x4x2xf32) <- (1x45640x8x4x2xf32)
        transpose_7 = paddle._C_ops.transpose(slice_18, [0, 2, 1, 3, 4])
        del slice_18

        # pd_op.flatten: (8x45640x4x2xf32) <- (1x8x45640x4x2xf32)
        flatten_7 = paddle._C_ops.flatten(transpose_7, 0, 1)

        # pd_op.grid_sample: (8x32x45640x4xf32) <- (8x32x-1x-1xf32, 8x45640x4x2xf32)
        grid_sample_3 = paddle._C_ops.grid_sample(
            reshape_9, flatten_7, "bilinear", "zeros", False
        )

        # pd_op.flatten: (-1x-1x-1xf32) <- (-1x-1x-1x-1xf32)
        flatten_8 = paddle._C_ops.flatten(split_5, 2, 3)

        # pd_op.transpose: (-1x-1x-1xf32) <- (-1x-1x-1xf32)
        transpose_8 = paddle._C_ops.transpose(flatten_8, [0, 2, 1])
        del flatten_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_5 = [full_3, full_4, slice_13, slice_14]
        del full_3, full_4, slice_13, slice_14

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_5, 0)
        del combine_5

        # pd_op.reshape: (8x32x-1x-1xf32) <- (-1x-1x-1xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(transpose_8, stack_5)
        del stack_5

        # pd_op.slice: (1x45640x8x4x2xf32) <- (1x45640x8x5x4x2xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            scale_1, [3], full_int_array_11, full_int_array_12, [1], [3]
        )
        del full_int_array_11, full_int_array_12

        # pd_op.transpose: (1x8x45640x4x2xf32) <- (1x45640x8x4x2xf32)
        transpose_9 = paddle._C_ops.transpose(slice_19, [0, 2, 1, 3, 4])
        del slice_19

        # pd_op.flatten: (8x45640x4x2xf32) <- (1x8x45640x4x2xf32)
        flatten_9 = paddle._C_ops.flatten(transpose_9, 0, 1)

        # pd_op.grid_sample: (8x32x45640x4xf32) <- (8x32x-1x-1xf32, 8x45640x4x2xf32)
        grid_sample_4 = paddle._C_ops.grid_sample(
            reshape_10, flatten_9, "bilinear", "zeros", False
        )

        # pd_op.transpose: (1x8x45640x5x4xf32) <- (1x45640x8x5x4xf32)
        transpose_10 = paddle._C_ops.transpose(reshape_3, [0, 2, 1, 3, 4])
        del reshape_3

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [8, 1, 45640, 20]

        # pd_op.reshape: (8x1x45640x20xf32) <- (1x8x45640x5x4xf32, 4xi64)
        reshape_11 = paddle._C_ops.reshape(transpose_10, full_int_array_13)
        del full_int_array_13

        # builtin.combine: ([8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32]) <- (8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32)
        combine_6 = [
            grid_sample_0,
            grid_sample_1,
            grid_sample_2,
            grid_sample_3,
            grid_sample_4,
        ]

        # pd_op.stack: (8x32x45640x5x4xf32) <- ([8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32, 8x32x45640x4xf32])
        stack_6 = paddle._C_ops.stack(combine_6, -2)
        del combine_6

        # pd_op.flatten: (8x32x45640x20xf32) <- (8x32x45640x5x4xf32)
        flatten_10 = paddle._C_ops.flatten(stack_6, 3, 4)

        # pd_op.multiply: (8x32x45640x20xf32) <- (8x32x45640x20xf32, 8x1x45640x20xf32)
        multiply_6 = paddle._C_ops.multiply(flatten_10, reshape_11)

        # pd_op.sum: (8x32x45640xf32) <- (8x32x45640x20xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_6, full_int_array_0, None, False)
        del full_int_array_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_14 = [1, 256, 45640]

        # pd_op.reshape: (1x256x45640xf32) <- (8x32x45640xf32, 3xi64)
        reshape_12 = paddle._C_ops.reshape(sum_0, full_int_array_14)
        del full_int_array_14

        # pd_op.transpose: (1x45640x256xf32) <- (1x256x45640xf32)
        transpose_11 = paddle._C_ops.transpose(reshape_12, [0, 2, 1])
        del reshape_12

        # pd_op.matmul: (1x45640x256xf32) <- (1x45640x256xf32, 256x256xf32)
        matmul_3 = paddle._C_ops.matmul(transpose_11, parameter_1, False, False)
        del parameter_1

        # pd_op.add: (1x45640x256xf32) <- (1x45640x256xf32, 256xf32)
        add_0 = paddle._C_ops.add(matmul_3, parameter_0)
        del (
            add_1,
            add_2,
            add_3,
            assign_0,
            assign_1,
            assign_10,
            assign_2,
            assign_3,
            assign_4,
            assign_5,
            assign_6,
            assign_7,
            assign_8,
            assign_9,
            cast_0,
            divide_0,
            flatten_1,
            flatten_10,
            flatten_3,
            flatten_5,
            flatten_7,
            flatten_9,
            full_0,
            full_1,
            full_2,
            grid_sample_0,
            grid_sample_1,
            grid_sample_2,
            grid_sample_3,
            grid_sample_4,
            matmul_0,
            matmul_1,
            matmul_2,
            matmul_3,
            multiply_0,
            multiply_6,
            parameter_0,
            reshape_1,
            reshape_10,
            reshape_11,
            reshape_5,
            reshape_6,
            reshape_7,
            reshape_8,
            reshape_9,
            scale_1,
            softmax_0,
            split_1,
            split_2,
            split_3,
            split_4,
            split_5,
            stack_6,
            sum_0,
            transpose_0,
            transpose_1,
            transpose_10,
            transpose_11,
            transpose_2,
            transpose_3,
            transpose_4,
            transpose_5,
            transpose_6,
            transpose_7,
            transpose_8,
            transpose_9,
            unsqueeze_0,
        )

        return add_0
