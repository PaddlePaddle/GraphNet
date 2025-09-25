import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.unsqueeze: (1x1x5x2xf32) <- (1x5x2xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(data_1, full_int_array_0)
        del data_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_0, [0], full_int_array_1, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            slice_0, [0], full_int_array_1, full_int_array_0, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            slice_0, [0], full_int_array_0, full_int_array_2, [1], [0]
        )
        del slice_0

        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_0, slice_1, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)
        del arange_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_2, float("0.5"), True)
        del cast_0

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_0, slice_2, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)
        del arange_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(cast_1, full_2, float("0.5"), True)
        del cast_1

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_0 = [scale_0, scale_1]
        del scale_0, scale_1

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)
        del combine_0

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_0 = paddle._C_ops.flatten(split_0, 0, 1)
        del split_0

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(flatten_0, full_int_array_1)
        del flatten_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [0, 1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 2]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_3 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_3, full_int_array_4, [1, 1], [2, 3]
        )
        del full_int_array_3, full_int_array_4

        # pd_op.cast: (xf32) <- (xi64)
        cast_2 = paddle._C_ops.cast(slice_1, paddle.float32)
        del slice_1

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_1 = paddle._C_ops.multiply(slice_3, cast_2)
        del cast_2, slice_3

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_0 = paddle._C_ops.divide(unsqueeze_1, multiply_1)
        del multiply_1, unsqueeze_1

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_1 = paddle._C_ops.flatten(split_1, 0, 1)
        del split_1

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(flatten_1, full_int_array_1)
        del flatten_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [1, 1]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_4 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_5, full_int_array_6, [1, 1], [2, 3]
        )
        del full_int_array_5

        # pd_op.cast: (xf32) <- (xi64)
        cast_3 = paddle._C_ops.cast(slice_2, paddle.float32)
        del slice_2

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_2 = paddle._C_ops.multiply(slice_4, cast_3)
        del cast_3, slice_4

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_1 = paddle._C_ops.divide(unsqueeze_2, multiply_2)
        del multiply_2, unsqueeze_2

        # builtin.combine: ([1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32)
        combine_1 = [divide_1, divide_0]
        del divide_0, divide_1

        # pd_op.stack: (1x-1x2xf32) <- ([1x-1xf32, 1x-1xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)
        del combine_1

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_0, [0], full_int_array_0, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_1, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            slice_5, [0], full_int_array_0, full_int_array_2, [1], [0]
        )
        del slice_5

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_2 = paddle.arange(full_0, slice_6, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_4 = paddle._C_ops.cast(arange_2, paddle.float32)
        del arange_2

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_4, full_2, float("0.5"), True)
        del cast_4

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_3 = paddle.arange(full_0, slice_7, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_5 = paddle._C_ops.cast(arange_3, paddle.float32)
        del arange_3

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cast_5, full_2, float("0.5"), True)
        del cast_5

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_2 = [scale_2, scale_3]
        del scale_2, scale_3

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_2)
        del combine_2

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_2 = paddle._C_ops.flatten(split_2, 0, 1)
        del split_2

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(flatten_2, full_int_array_1)
        del flatten_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [2, 2]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_8 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_6, full_int_array_7, [1, 1], [2, 3]
        )
        del full_int_array_6, full_int_array_7

        # pd_op.cast: (xf32) <- (xi64)
        cast_6 = paddle._C_ops.cast(slice_6, paddle.float32)
        del slice_6

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_3 = paddle._C_ops.multiply(slice_8, cast_6)
        del cast_6, slice_8

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_2 = paddle._C_ops.divide(unsqueeze_3, multiply_3)
        del multiply_3, unsqueeze_3

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_3 = paddle._C_ops.flatten(split_3, 0, 1)
        del split_3

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(flatten_3, full_int_array_1)
        del flatten_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [1, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [2, 1]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_9 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_8, full_int_array_9, [1, 1], [2, 3]
        )
        del full_int_array_8

        # pd_op.cast: (xf32) <- (xi64)
        cast_7 = paddle._C_ops.cast(slice_7, paddle.float32)
        del slice_7

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_4 = paddle._C_ops.multiply(slice_9, cast_7)
        del cast_7, slice_9

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_3 = paddle._C_ops.divide(unsqueeze_4, multiply_4)
        del multiply_4, unsqueeze_4

        # builtin.combine: ([1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32)
        combine_3 = [divide_3, divide_2]
        del divide_2, divide_3

        # pd_op.stack: (1x-1x2xf32) <- ([1x-1xf32, 1x-1xf32])
        stack_1 = paddle._C_ops.stack(combine_3, -1)
        del combine_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [3]

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_0, [0], full_int_array_2, full_int_array_10, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            slice_10, [0], full_int_array_1, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            slice_10, [0], full_int_array_0, full_int_array_2, [1], [0]
        )
        del slice_10

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_4 = paddle.arange(full_0, slice_11, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_8 = paddle._C_ops.cast(arange_4, paddle.float32)
        del arange_4

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_8, full_2, float("0.5"), True)
        del cast_8

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_5 = paddle.arange(full_0, slice_12, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_9 = paddle._C_ops.cast(arange_5, paddle.float32)
        del arange_5

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(cast_9, full_2, float("0.5"), True)
        del cast_9

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_4 = [scale_4, scale_5]
        del scale_4, scale_5

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_4)
        del combine_4

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_4 = paddle._C_ops.flatten(split_4, 0, 1)
        del split_4

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(flatten_4, full_int_array_1)
        del flatten_4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [3, 2]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_13 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_9, full_int_array_11, [1, 1], [2, 3]
        )
        del full_int_array_11, full_int_array_9

        # pd_op.cast: (xf32) <- (xi64)
        cast_10 = paddle._C_ops.cast(slice_11, paddle.float32)
        del slice_11

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_5 = paddle._C_ops.multiply(slice_13, cast_10)
        del cast_10, slice_13

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_4 = paddle._C_ops.divide(unsqueeze_5, multiply_5)
        del multiply_5, unsqueeze_5

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_5 = paddle._C_ops.flatten(split_5, 0, 1)
        del split_5

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(flatten_5, full_int_array_1)
        del flatten_5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [2, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [3, 1]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_14 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_12, full_int_array_13, [1, 1], [2, 3]
        )
        del full_int_array_12

        # pd_op.cast: (xf32) <- (xi64)
        cast_11 = paddle._C_ops.cast(slice_12, paddle.float32)
        del slice_12

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_6 = paddle._C_ops.multiply(slice_14, cast_11)
        del cast_11, slice_14

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_5 = paddle._C_ops.divide(unsqueeze_6, multiply_6)
        del multiply_6, unsqueeze_6

        # builtin.combine: ([1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32)
        combine_5 = [divide_5, divide_4]
        del divide_4, divide_5

        # pd_op.stack: (1x-1x2xf32) <- ([1x-1xf32, 1x-1xf32])
        stack_2 = paddle._C_ops.stack(combine_5, -1)
        del combine_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [4]

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            data_0, [0], full_int_array_10, full_int_array_14, [1], [0]
        )
        del full_int_array_10

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(
            slice_15, [0], full_int_array_1, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(
            slice_15, [0], full_int_array_0, full_int_array_2, [1], [0]
        )
        del slice_15

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_6 = paddle.arange(full_0, slice_16, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_12 = paddle._C_ops.cast(arange_6, paddle.float32)
        del arange_6

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_12, full_2, float("0.5"), True)
        del cast_12

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_7 = paddle.arange(full_0, slice_17, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_13 = paddle._C_ops.cast(arange_7, paddle.float32)
        del arange_7

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(cast_13, full_2, float("0.5"), True)
        del cast_13

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_6 = [scale_6, scale_7]
        del scale_6, scale_7

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_6)
        del combine_6

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_6 = paddle._C_ops.flatten(split_6, 0, 1)
        del split_6

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(flatten_6, full_int_array_1)
        del flatten_6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_15 = [4, 2]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_18 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_13, full_int_array_15, [1, 1], [2, 3]
        )
        del full_int_array_13, full_int_array_15

        # pd_op.cast: (xf32) <- (xi64)
        cast_14 = paddle._C_ops.cast(slice_16, paddle.float32)
        del slice_16

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_7 = paddle._C_ops.multiply(slice_18, cast_14)
        del cast_14, slice_18

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_6 = paddle._C_ops.divide(unsqueeze_7, multiply_7)
        del multiply_7, unsqueeze_7

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_7 = paddle._C_ops.flatten(split_7, 0, 1)
        del split_7

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(flatten_7, full_int_array_1)
        del flatten_7

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [3, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [4, 1]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_19 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_16, full_int_array_17, [1, 1], [2, 3]
        )
        del full_int_array_16

        # pd_op.cast: (xf32) <- (xi64)
        cast_15 = paddle._C_ops.cast(slice_17, paddle.float32)
        del slice_17

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_8 = paddle._C_ops.multiply(slice_19, cast_15)
        del cast_15, slice_19

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_7 = paddle._C_ops.divide(unsqueeze_8, multiply_8)
        del multiply_8, unsqueeze_8

        # builtin.combine: ([1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32)
        combine_7 = [divide_7, divide_6]
        del divide_6, divide_7

        # pd_op.stack: (1x-1x2xf32) <- ([1x-1xf32, 1x-1xf32])
        stack_3 = paddle._C_ops.stack(combine_7, -1)
        del combine_7

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [5]

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            data_0, [0], full_int_array_14, full_int_array_18, [1], [0]
        )
        del data_0, full_int_array_14, full_int_array_18

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(
            slice_20, [0], full_int_array_1, full_int_array_0, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(
            slice_20, [0], full_int_array_0, full_int_array_2, [1], [0]
        )
        del full_int_array_0, slice_20

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_8 = paddle.arange(full_0, slice_21, full_1, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_16 = paddle._C_ops.cast(arange_8, paddle.float32)
        del arange_8

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_16, full_2, float("0.5"), True)
        del cast_16

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_9 = paddle.arange(full_0, slice_22, full_1, dtype="int64")
        del full_0, full_1

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_17 = paddle._C_ops.cast(arange_9, paddle.float32)
        del arange_9

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(cast_17, full_2, float("0.5"), True)
        del cast_17, full_2

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_8 = [scale_8, scale_9]
        del scale_8, scale_9

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_8)
        del combine_8

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_8,
            split_9,
        ) = meshgrid_4
        del meshgrid_4

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_8 = paddle._C_ops.flatten(split_8, 0, 1)
        del split_8

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(flatten_8, full_int_array_1)
        del flatten_8

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [5, 2]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_23 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_17, full_int_array_19, [1, 1], [2, 3]
        )
        del full_int_array_17, full_int_array_19

        # pd_op.cast: (xf32) <- (xi64)
        cast_18 = paddle._C_ops.cast(slice_21, paddle.float32)
        del slice_21

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_9 = paddle._C_ops.multiply(slice_23, cast_18)
        del cast_18, slice_23

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_8 = paddle._C_ops.divide(unsqueeze_9, multiply_9)
        del multiply_9, unsqueeze_9

        # pd_op.flatten: (-1xf32) <- (-1x-1xf32)
        flatten_9 = paddle._C_ops.flatten(split_9, 0, 1)
        del split_9

        # pd_op.unsqueeze: (1x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(flatten_9, full_int_array_1)
        del flatten_9, full_int_array_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [4, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [5, 1]

        # pd_op.slice: (1x1xf32) <- (1x1x5x2xf32, 2xi64, 2xi64)
        slice_24 = paddle._C_ops.slice(
            unsqueeze_0, [2, 3], full_int_array_20, full_int_array_21, [1, 1], [2, 3]
        )
        del full_int_array_20, full_int_array_21

        # pd_op.cast: (xf32) <- (xi64)
        cast_19 = paddle._C_ops.cast(slice_22, paddle.float32)
        del slice_22

        # pd_op.multiply: (1x1xf32) <- (1x1xf32, xf32)
        multiply_10 = paddle._C_ops.multiply(slice_24, cast_19)
        del cast_19, slice_24

        # pd_op.divide: (1x-1xf32) <- (1x-1xf32, 1x1xf32)
        divide_9 = paddle._C_ops.divide(unsqueeze_10, multiply_10)
        del multiply_10, unsqueeze_10

        # builtin.combine: ([1x-1xf32, 1x-1xf32]) <- (1x-1xf32, 1x-1xf32)
        combine_9 = [divide_9, divide_8]
        del divide_8, divide_9

        # pd_op.stack: (1x-1x2xf32) <- ([1x-1xf32, 1x-1xf32])
        stack_4 = paddle._C_ops.stack(combine_9, -1)
        del combine_9

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32]) <- (1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32)
        combine_10 = [stack_0, stack_1, stack_2, stack_3, stack_4]
        del stack_0, stack_1, stack_2, stack_3, stack_4

        # pd_op.concat: (1x-1x2xf32) <- ([1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32, 1x-1x2xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_10, full_3)
        del combine_10, full_3

        # pd_op.unsqueeze: (1x-1x1x2xf32) <- (1x-1x2xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(concat_0, full_int_array_2)
        del concat_0, full_int_array_2

        # pd_op.multiply: (1x-1x5x2xf32) <- (1x-1x1x2xf32, 1x1x5x2xf32)
        multiply_0 = paddle._C_ops.multiply(unsqueeze_11, unsqueeze_0)
        del unsqueeze_0, unsqueeze_11

        return multiply_0
