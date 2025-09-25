import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self, parameter_0, parameter_1, parameter_2, parameter_3, data_0, data_1, data_2
    ):
        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            slice_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            slice_0, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del slice_0

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(slice_1, slice_2)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_0 = paddle._C_ops.scale(multiply_0, full_0, float("0"), True)
        del multiply_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_0 = [scale_0]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.slice: (1x-1xf32) <- (1x45640xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_1, [1], full_int_array_0, stack_0, [-1], [])

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_1 = [full_1, slice_1, slice_2, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (1x-1x-1x1xf32) <- (1x-1xf32, 4xi64)
        reshape_0 = paddle._C_ops.reshape(slice_3, stack_1)
        del slice_3, stack_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_4 = paddle._C_ops.slice(
            reshape_0, [2, 3], full_int_array_3, full_int_array_4, [1, 1], [2, 3]
        )

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(slice_4, full_int_array_1, None, False)
        del slice_4

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_5 = paddle._C_ops.slice(
            reshape_0, [1, 3], full_int_array_3, full_int_array_4, [1, 1], [1, 3]
        )
        del reshape_0

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(slice_5, full_int_array_1, None, False)
        del slice_5

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_1, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_0 = paddle._C_ops.cast(slice_1, paddle.int32)
        del slice_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_0 = paddle._C_ops.linspace(
            full_2,
            scale_1,
            cast_0,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_0, scale_1

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_2 = paddle._C_ops.scale(slice_2, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_1 = paddle._C_ops.cast(slice_2, paddle.int32)
        del slice_2

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_1 = paddle._C_ops.linspace(
            full_2,
            scale_2,
            cast_1,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_1, scale_2

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_2 = [linspace_0, linspace_1]
        del linspace_0, linspace_1

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_2)
        del combine_2

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_0,
            split_1,
        ) = meshgrid_0
        del meshgrid_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-1]

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(split_1, full_int_array_5)
        del split_1

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(split_0, full_int_array_5)
        del split_0

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("-1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([-1x-1x1xf32, -1x-1x1xf32]) <- (-1x-1x1xf32, -1x-1x1xf32)
        combine_3 = [unsqueeze_0, unsqueeze_1]
        del unsqueeze_0, unsqueeze_1

        # pd_op.concat: (-1x-1x2xf32) <- ([-1x-1x1xf32, -1x-1x1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_3, full_3)
        del combine_3

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(sum_1, full_int_array_5)
        del sum_1

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_3 = paddle._C_ops.unsqueeze(sum_0, full_int_array_5)
        del sum_0

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # builtin.combine: ([1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32)
        combine_4 = [unsqueeze_2, unsqueeze_3]
        del unsqueeze_2, unsqueeze_3

        # pd_op.concat: (1x2xf32) <- ([1x1xf32, 1x1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_4, full_4)
        del combine_4

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 1, 1, 2]

        # pd_op.reshape: (1x1x1x2xf32) <- (1x2xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(concat_1, full_int_array_6)
        del concat_1

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        unsqueeze_4 = paddle._C_ops.unsqueeze(concat_0, full_int_array_0)
        del concat_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, -1, -1, -1]

        # pd_op.expand: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 4xi64)
        expand_0 = paddle._C_ops.expand(unsqueeze_4, full_int_array_7)
        del unsqueeze_4

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(expand_0, full_0, float("0.5"), True)
        del expand_0

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1x1x1x2xf32)
        divide_0 = paddle._C_ops.divide(scale_3, reshape_1)
        del reshape_1, scale_3

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            divide_0, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0.05"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(full_like_0, full_5, float("0"), True)
        del full_like_0

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_0, float("0"), True)
        del scale_4

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        combine_5 = [divide_0, scale_5]
        del divide_0, scale_5

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_5, full_3)
        del combine_5

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_8 = [1, -1, 4]

        # pd_op.reshape: (1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(concat_2, full_int_array_8)
        del concat_2

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_2, [0], full_int_array_1, full_int_array_2, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            slice_6, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            slice_6, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del slice_6

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_1 = paddle._C_ops.multiply(slice_7, slice_8)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_0 = paddle._C_ops.add(scale_0, multiply_1)
        del multiply_1, scale_0

        # builtin.combine: ([xi64]) <- (xi64)
        combine_6 = [add_0]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_2 = paddle._C_ops.stack(combine_6, 0)
        del combine_6

        # pd_op.slice: (1x-1xf32) <- (1x45640xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(data_1, [1], stack_0, stack_2, [-1], [])
        del stack_0

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_7 = [full_1, slice_7, slice_8, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_3 = paddle._C_ops.stack(combine_7, 0)
        del combine_7

        # pd_op.reshape: (1x-1x-1x1xf32) <- (1x-1xf32, 4xi64)
        reshape_3 = paddle._C_ops.reshape(slice_9, stack_3)
        del slice_9, stack_3

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_10 = paddle._C_ops.slice(
            reshape_3, [2, 3], full_int_array_3, full_int_array_4, [1, 1], [2, 3]
        )

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(slice_10, full_int_array_1, None, False)
        del slice_10

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_11 = paddle._C_ops.slice(
            reshape_3, [1, 3], full_int_array_3, full_int_array_4, [1, 1], [1, 3]
        )
        del reshape_3

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(slice_11, full_int_array_1, None, False)
        del slice_11

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_6 = paddle._C_ops.scale(slice_7, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_2 = paddle._C_ops.cast(slice_7, paddle.int32)
        del slice_7

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_2 = paddle._C_ops.linspace(
            full_2,
            scale_6,
            cast_2,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_2, scale_6

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_7 = paddle._C_ops.scale(slice_8, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_3 = paddle._C_ops.cast(slice_8, paddle.int32)
        del slice_8

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_3 = paddle._C_ops.linspace(
            full_2,
            scale_7,
            cast_3,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_3, scale_7

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_8 = [linspace_2, linspace_3]
        del linspace_2, linspace_3

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_8)
        del combine_8

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_2,
            split_3,
        ) = meshgrid_1
        del meshgrid_1

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_5 = paddle._C_ops.unsqueeze(split_3, full_int_array_5)
        del split_3

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_6 = paddle._C_ops.unsqueeze(split_2, full_int_array_5)
        del split_2

        # builtin.combine: ([-1x-1x1xf32, -1x-1x1xf32]) <- (-1x-1x1xf32, -1x-1x1xf32)
        combine_9 = [unsqueeze_5, unsqueeze_6]
        del unsqueeze_5, unsqueeze_6

        # pd_op.concat: (-1x-1x2xf32) <- ([-1x-1x1xf32, -1x-1x1xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_9, full_3)
        del combine_9

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_7 = paddle._C_ops.unsqueeze(sum_3, full_int_array_5)
        del sum_3

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_8 = paddle._C_ops.unsqueeze(sum_2, full_int_array_5)
        del sum_2

        # builtin.combine: ([1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32)
        combine_10 = [unsqueeze_7, unsqueeze_8]
        del unsqueeze_7, unsqueeze_8

        # pd_op.concat: (1x2xf32) <- ([1x1xf32, 1x1xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_10, full_4)
        del combine_10

        # pd_op.reshape: (1x1x1x2xf32) <- (1x2xf32, 4xi64)
        reshape_4 = paddle._C_ops.reshape(concat_4, full_int_array_6)
        del concat_4

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        unsqueeze_9 = paddle._C_ops.unsqueeze(concat_3, full_int_array_0)
        del concat_3

        # pd_op.expand: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 4xi64)
        expand_1 = paddle._C_ops.expand(unsqueeze_9, full_int_array_7)
        del unsqueeze_9

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(expand_1, full_0, float("0.5"), True)
        del expand_1

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1x1x1x2xf32)
        divide_1 = paddle._C_ops.divide(scale_8, reshape_4)
        del reshape_4, scale_8

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        full_like_1 = paddle._C_ops.full_like(
            divide_1, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(full_like_1, full_5, float("0"), True)
        del full_like_1

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(scale_9, full_6, float("0"), True)
        del full_6, scale_9

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        combine_11 = [divide_1, scale_10]
        del divide_1, scale_10

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_11, full_3)
        del combine_11

        # pd_op.reshape: (1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        reshape_5 = paddle._C_ops.reshape(concat_5, full_int_array_8)
        del concat_5

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [3]

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            data_2, [0], full_int_array_2, full_int_array_9, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            slice_12, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            slice_12, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del slice_12

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_2 = paddle._C_ops.multiply(slice_13, slice_14)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_1 = paddle._C_ops.add(add_0, multiply_2)
        del add_0, multiply_2

        # builtin.combine: ([xi64]) <- (xi64)
        combine_12 = [add_1]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_4 = paddle._C_ops.stack(combine_12, 0)
        del combine_12

        # pd_op.slice: (1x-1xf32) <- (1x45640xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(data_1, [1], stack_2, stack_4, [-1], [])
        del stack_2

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_13 = [full_1, slice_13, slice_14, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_5 = paddle._C_ops.stack(combine_13, 0)
        del combine_13

        # pd_op.reshape: (1x-1x-1x1xf32) <- (1x-1xf32, 4xi64)
        reshape_6 = paddle._C_ops.reshape(slice_15, stack_5)
        del slice_15, stack_5

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_16 = paddle._C_ops.slice(
            reshape_6, [2, 3], full_int_array_3, full_int_array_4, [1, 1], [2, 3]
        )

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(slice_16, full_int_array_1, None, False)
        del slice_16

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_17 = paddle._C_ops.slice(
            reshape_6, [1, 3], full_int_array_3, full_int_array_4, [1, 1], [1, 3]
        )
        del reshape_6

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(slice_17, full_int_array_1, None, False)
        del slice_17

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_11 = paddle._C_ops.scale(slice_13, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_4 = paddle._C_ops.cast(slice_13, paddle.int32)
        del slice_13

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_4 = paddle._C_ops.linspace(
            full_2,
            scale_11,
            cast_4,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_4, scale_11

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_12 = paddle._C_ops.scale(slice_14, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_5 = paddle._C_ops.cast(slice_14, paddle.int32)
        del slice_14

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_5 = paddle._C_ops.linspace(
            full_2,
            scale_12,
            cast_5,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_5, scale_12

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_14 = [linspace_4, linspace_5]
        del linspace_4, linspace_5

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_14)
        del combine_14

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_4,
            split_5,
        ) = meshgrid_2
        del meshgrid_2

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_10 = paddle._C_ops.unsqueeze(split_5, full_int_array_5)
        del split_5

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_11 = paddle._C_ops.unsqueeze(split_4, full_int_array_5)
        del split_4

        # builtin.combine: ([-1x-1x1xf32, -1x-1x1xf32]) <- (-1x-1x1xf32, -1x-1x1xf32)
        combine_15 = [unsqueeze_10, unsqueeze_11]
        del unsqueeze_10, unsqueeze_11

        # pd_op.concat: (-1x-1x2xf32) <- ([-1x-1x1xf32, -1x-1x1xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_15, full_3)
        del combine_15

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_12 = paddle._C_ops.unsqueeze(sum_5, full_int_array_5)
        del sum_5

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_13 = paddle._C_ops.unsqueeze(sum_4, full_int_array_5)
        del sum_4

        # builtin.combine: ([1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32)
        combine_16 = [unsqueeze_12, unsqueeze_13]
        del unsqueeze_12, unsqueeze_13

        # pd_op.concat: (1x2xf32) <- ([1x1xf32, 1x1xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_16, full_4)
        del combine_16

        # pd_op.reshape: (1x1x1x2xf32) <- (1x2xf32, 4xi64)
        reshape_7 = paddle._C_ops.reshape(concat_7, full_int_array_6)
        del concat_7

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        unsqueeze_14 = paddle._C_ops.unsqueeze(concat_6, full_int_array_0)
        del concat_6

        # pd_op.expand: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 4xi64)
        expand_2 = paddle._C_ops.expand(unsqueeze_14, full_int_array_7)
        del unsqueeze_14

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(expand_2, full_0, float("0.5"), True)
        del expand_2

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1x1x1x2xf32)
        divide_2 = paddle._C_ops.divide(scale_13, reshape_7)
        del reshape_7, scale_13

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        full_like_2 = paddle._C_ops.full_like(
            divide_2, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(full_like_2, full_5, float("0"), True)
        del full_like_2

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("4"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(scale_14, full_7, float("0"), True)
        del full_7, scale_14

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        combine_17 = [divide_2, scale_15]
        del divide_2, scale_15

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_17, full_3)
        del combine_17

        # pd_op.reshape: (1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        reshape_8 = paddle._C_ops.reshape(concat_8, full_int_array_8)
        del concat_8

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [4]

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(
            data_2, [0], full_int_array_9, full_int_array_10, [1], [0]
        )
        del full_int_array_9

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(
            slice_18, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(
            slice_18, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del slice_18

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_3 = paddle._C_ops.multiply(slice_19, slice_20)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_2 = paddle._C_ops.add(add_1, multiply_3)
        del add_1, multiply_3

        # builtin.combine: ([xi64]) <- (xi64)
        combine_18 = [add_2]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_6 = paddle._C_ops.stack(combine_18, 0)
        del combine_18

        # pd_op.slice: (1x-1xf32) <- (1x45640xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(data_1, [1], stack_4, stack_6, [-1], [])
        del stack_4

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_19 = [full_1, slice_19, slice_20, full_1]

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_7 = paddle._C_ops.stack(combine_19, 0)
        del combine_19

        # pd_op.reshape: (1x-1x-1x1xf32) <- (1x-1xf32, 4xi64)
        reshape_9 = paddle._C_ops.reshape(slice_21, stack_7)
        del slice_21, stack_7

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_22 = paddle._C_ops.slice(
            reshape_9, [2, 3], full_int_array_3, full_int_array_4, [1, 1], [2, 3]
        )

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_6 = paddle._C_ops.sum(slice_22, full_int_array_1, None, False)
        del slice_22

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_23 = paddle._C_ops.slice(
            reshape_9, [1, 3], full_int_array_3, full_int_array_4, [1, 1], [1, 3]
        )
        del reshape_9

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_7 = paddle._C_ops.sum(slice_23, full_int_array_1, None, False)
        del slice_23

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_16 = paddle._C_ops.scale(slice_19, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_6 = paddle._C_ops.cast(slice_19, paddle.int32)
        del slice_19

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_6 = paddle._C_ops.linspace(
            full_2,
            scale_16,
            cast_6,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_6, scale_16

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_17 = paddle._C_ops.scale(slice_20, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_7 = paddle._C_ops.cast(slice_20, paddle.int32)
        del slice_20

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_7 = paddle._C_ops.linspace(
            full_2,
            scale_17,
            cast_7,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_7, scale_17

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_20 = [linspace_6, linspace_7]
        del linspace_6, linspace_7

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_20)
        del combine_20

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_6,
            split_7,
        ) = meshgrid_3
        del meshgrid_3

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_15 = paddle._C_ops.unsqueeze(split_7, full_int_array_5)
        del split_7

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_16 = paddle._C_ops.unsqueeze(split_6, full_int_array_5)
        del split_6

        # builtin.combine: ([-1x-1x1xf32, -1x-1x1xf32]) <- (-1x-1x1xf32, -1x-1x1xf32)
        combine_21 = [unsqueeze_15, unsqueeze_16]
        del unsqueeze_15, unsqueeze_16

        # pd_op.concat: (-1x-1x2xf32) <- ([-1x-1x1xf32, -1x-1x1xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_21, full_3)
        del combine_21

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_17 = paddle._C_ops.unsqueeze(sum_7, full_int_array_5)
        del sum_7

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_18 = paddle._C_ops.unsqueeze(sum_6, full_int_array_5)
        del sum_6

        # builtin.combine: ([1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32)
        combine_22 = [unsqueeze_17, unsqueeze_18]
        del unsqueeze_17, unsqueeze_18

        # pd_op.concat: (1x2xf32) <- ([1x1xf32, 1x1xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_22, full_4)
        del combine_22

        # pd_op.reshape: (1x1x1x2xf32) <- (1x2xf32, 4xi64)
        reshape_10 = paddle._C_ops.reshape(concat_10, full_int_array_6)
        del concat_10

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        unsqueeze_19 = paddle._C_ops.unsqueeze(concat_9, full_int_array_0)
        del concat_9

        # pd_op.expand: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 4xi64)
        expand_3 = paddle._C_ops.expand(unsqueeze_19, full_int_array_7)
        del unsqueeze_19

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(expand_3, full_0, float("0.5"), True)
        del expand_3

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1x1x1x2xf32)
        divide_3 = paddle._C_ops.divide(scale_18, reshape_10)
        del reshape_10, scale_18

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        full_like_3 = paddle._C_ops.full_like(
            divide_3, full_0, paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(full_like_3, full_5, float("0"), True)
        del full_like_3

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("8"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(scale_19, full_8, float("0"), True)
        del full_8, scale_19

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        combine_23 = [divide_3, scale_20]
        del divide_3, scale_20

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_23, full_3)
        del combine_23

        # pd_op.reshape: (1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        reshape_11 = paddle._C_ops.reshape(concat_11, full_int_array_8)
        del concat_11

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [5]

        # pd_op.slice: (2xi64) <- (5x2xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(
            data_2, [0], full_int_array_10, full_int_array_11, [1], [0]
        )
        del data_2, full_int_array_10, full_int_array_11

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(
            slice_24, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(
            slice_24, [0], full_int_array_1, full_int_array_2, [1], [0]
        )
        del full_int_array_2, slice_24

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_4 = paddle._C_ops.multiply(slice_25, slice_26)

        # pd_op.add: (xi64) <- (xi64, xi64)
        add_3 = paddle._C_ops.add(add_2, multiply_4)
        del add_2, multiply_4

        # builtin.combine: ([xi64]) <- (xi64)
        combine_24 = [add_3]
        del add_3

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_8 = paddle._C_ops.stack(combine_24, 0)
        del combine_24

        # pd_op.slice: (1x-1xf32) <- (1x45640xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(data_1, [1], stack_6, stack_8, [-1], [])
        del stack_6, stack_8

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_25 = [full_1, slice_25, slice_26, full_1]
        del full_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_9 = paddle._C_ops.stack(combine_25, 0)
        del combine_25

        # pd_op.reshape: (1x-1x-1x1xf32) <- (1x-1xf32, 4xi64)
        reshape_12 = paddle._C_ops.reshape(slice_27, stack_9)
        del slice_27, stack_9

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_28 = paddle._C_ops.slice(
            reshape_12, [2, 3], full_int_array_3, full_int_array_4, [1, 1], [2, 3]
        )

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_8 = paddle._C_ops.sum(slice_28, full_int_array_1, None, False)
        del slice_28

        # pd_op.slice: (1x-1xf32) <- (1x-1x-1x1xf32, 2xi64, 2xi64)
        slice_29 = paddle._C_ops.slice(
            reshape_12, [1, 3], full_int_array_3, full_int_array_4, [1, 1], [1, 3]
        )
        del full_int_array_3, full_int_array_4, reshape_12

        # pd_op.sum: (1xf32) <- (1x-1xf32, 1xi64)
        sum_9 = paddle._C_ops.sum(slice_29, full_int_array_1, None, False)
        del full_int_array_1, slice_29

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_21 = paddle._C_ops.scale(slice_25, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_8 = paddle._C_ops.cast(slice_25, paddle.int32)
        del slice_25

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_8 = paddle._C_ops.linspace(
            full_2,
            scale_21,
            cast_8,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_8, scale_21

        # pd_op.scale: (xi64) <- (xi64, 1xf32)
        scale_22 = paddle._C_ops.scale(slice_26, full_0, float("-1"), True)

        # pd_op.cast: (xi32) <- (xi64)
        cast_9 = paddle._C_ops.cast(slice_26, paddle.int32)
        del slice_26

        # pd_op.linspace: (-1xf32) <- (1xf32, xi64, xi32)
        linspace_9 = paddle._C_ops.linspace(
            full_2,
            scale_22,
            cast_9,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del cast_9, full_2, scale_22

        # builtin.combine: ([-1xf32, -1xf32]) <- (-1xf32, -1xf32)
        combine_26 = [linspace_8, linspace_9]
        del linspace_8, linspace_9

        # pd_op.meshgrid: ([-1x-1xf32, -1x-1xf32]) <- ([-1xf32, -1xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_26)
        del combine_26

        # builtin.split: (-1x-1xf32, -1x-1xf32) <- ([-1x-1xf32, -1x-1xf32])
        (
            split_8,
            split_9,
        ) = meshgrid_4
        del meshgrid_4

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_20 = paddle._C_ops.unsqueeze(split_9, full_int_array_5)
        del split_9

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_21 = paddle._C_ops.unsqueeze(split_8, full_int_array_5)
        del split_8

        # builtin.combine: ([-1x-1x1xf32, -1x-1x1xf32]) <- (-1x-1x1xf32, -1x-1x1xf32)
        combine_27 = [unsqueeze_20, unsqueeze_21]
        del unsqueeze_20, unsqueeze_21

        # pd_op.concat: (-1x-1x2xf32) <- ([-1x-1x1xf32, -1x-1x1xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_27, full_3)
        del combine_27

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_22 = paddle._C_ops.unsqueeze(sum_9, full_int_array_5)
        del sum_9

        # pd_op.unsqueeze: (1x1xf32) <- (1xf32, 1xi64)
        unsqueeze_23 = paddle._C_ops.unsqueeze(sum_8, full_int_array_5)
        del sum_8

        # builtin.combine: ([1x1xf32, 1x1xf32]) <- (1x1xf32, 1x1xf32)
        combine_28 = [unsqueeze_22, unsqueeze_23]
        del unsqueeze_22, unsqueeze_23

        # pd_op.concat: (1x2xf32) <- ([1x1xf32, 1x1xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_28, full_4)
        del combine_28

        # pd_op.reshape: (1x1x1x2xf32) <- (1x2xf32, 4xi64)
        reshape_13 = paddle._C_ops.reshape(concat_13, full_int_array_6)
        del concat_13, full_int_array_6

        # pd_op.unsqueeze: (1x-1x-1x2xf32) <- (-1x-1x2xf32, 1xi64)
        unsqueeze_24 = paddle._C_ops.unsqueeze(concat_12, full_int_array_0)
        del concat_12, full_int_array_0

        # pd_op.expand: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 4xi64)
        expand_4 = paddle._C_ops.expand(unsqueeze_24, full_int_array_7)
        del full_int_array_7, unsqueeze_24

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(expand_4, full_0, float("0.5"), True)
        del expand_4

        # pd_op.divide: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1x1x1x2xf32)
        divide_4 = paddle._C_ops.divide(scale_23, reshape_13)
        del reshape_13, scale_23

        # pd_op.full_like: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        full_like_4 = paddle._C_ops.full_like(
            divide_4, full_0, paddle.float32, paddle.framework._current_expected_place()
        )
        del full_0

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(full_like_4, full_5, float("0"), True)
        del full_5, full_like_4

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full(
            [1], float("16"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x-1x2xf32) <- (1x-1x-1x2xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(scale_24, full_9, float("0"), True)
        del full_9, scale_24

        # builtin.combine: ([1x-1x-1x2xf32, 1x-1x-1x2xf32]) <- (1x-1x-1x2xf32, 1x-1x-1x2xf32)
        combine_29 = [divide_4, scale_25]
        del divide_4, scale_25

        # pd_op.concat: (1x-1x-1x4xf32) <- ([1x-1x-1x2xf32, 1x-1x-1x2xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_29, full_3)
        del combine_29, full_3

        # pd_op.reshape: (1x-1x4xf32) <- (1x-1x-1x4xf32, 3xi64)
        reshape_14 = paddle._C_ops.reshape(concat_14, full_int_array_8)
        del concat_14, full_int_array_8

        # builtin.combine: ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32]) <- (1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32)
        combine_30 = [reshape_2, reshape_5, reshape_8, reshape_11, reshape_14]
        del reshape_11, reshape_14, reshape_2, reshape_5, reshape_8

        # pd_op.concat: (1x-1x4xf32) <- ([1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32, 1x-1x4xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_30, full_4)
        del combine_30, full_4

        # pd_op.full: (xf32) <- ()
        full_10 = paddle._C_ops.full(
            [],
            float("0.01"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.greater_than: (1x-1x4xb) <- (1x-1x4xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(concat_15, full_10)
        del full_10

        # pd_op.full: (xf32) <- ()
        full_11 = paddle._C_ops.full(
            [],
            float("0.99"),
            paddle.float32,
            paddle.framework._current_expected_place(),
        )

        # pd_op.less_than: (1x-1x4xb) <- (1x-1x4xf32, xf32)
        less_than_0 = paddle._C_ops.less_than(concat_15, full_11)
        del full_11

        # pd_op.bitwise_and: (1x-1x4xb) <- (1x-1x4xb, 1x-1x4xb)
        bitwise_and_0 = paddle._C_ops.bitwise_and(greater_than_0, less_than_0)
        del greater_than_0, less_than_0

        # pd_op.all: (1x-1x1xb) <- (1x-1x4xb)
        all_0 = paddle._C_ops.all(bitwise_and_0, [-1], True)
        del bitwise_and_0

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (1x-1x4xf32) <- (1x-1x4xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(concat_15, full_12, float("1"), True)
        del full_12

        # pd_op.divide: (1x-1x4xf32) <- (1x-1x4xf32, 1x-1x4xf32)
        divide_5 = paddle._C_ops.divide(concat_15, scale_26)
        del concat_15, scale_26

        # pd_op.log: (1x-1x4xf32) <- (1x-1x4xf32)
        log_0 = paddle._C_ops.log(divide_5)
        del divide_5

        # pd_op.cast: (1x45640xb) <- (1x45640xf32)
        cast_10 = paddle._C_ops.cast(data_1, paddle.bool)
        del data_1

        # pd_op.unsqueeze: (1x45640x1xb) <- (1x45640xb, 1xi64)
        unsqueeze_25 = paddle._C_ops.unsqueeze(cast_10, full_int_array_5)
        del cast_10, full_int_array_5

        # pd_op.bitwise_not: (1x45640x1xb) <- (1x45640x1xb)
        bitwise_not_0 = paddle._C_ops.bitwise_not(unsqueeze_25)
        del unsqueeze_25

        # pd_op.full: (xf32) <- ()
        full_13 = paddle._C_ops.full(
            [], float("inf"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.cast: (1x45640x1xb) <- (1x45640x1xb)
        cast_11 = paddle._C_ops.cast(bitwise_not_0, paddle.bool)
        del bitwise_not_0

        # pd_op.assign: (1x45640x1xb) <- (1x45640x1xb)
        assign_0 = cast_11

        # pd_op.masked_fill: (1x45640x4xf32) <- (1x-1x4xf32, 1x45640x1xb, xf32)
        masked_fill_1 = paddle._C_ops.masked_fill(log_0, cast_11, full_13)
        del log_0

        # pd_op.bitwise_not: (1x-1x1xb) <- (1x-1x1xb)
        bitwise_not_1 = paddle._C_ops.bitwise_not(all_0)
        del all_0

        # pd_op.cast: (1x-1x1xb) <- (1x-1x1xb)
        cast_12 = paddle._C_ops.cast(bitwise_not_1, paddle.bool)
        del bitwise_not_1

        # pd_op.assign: (1x-1x1xb) <- (1x-1x1xb)
        assign_1 = cast_12

        # pd_op.masked_fill: (1x45640x4xf32) <- (1x45640x4xf32, 1x-1x1xb, xf32)
        masked_fill_0 = paddle._C_ops.masked_fill(masked_fill_1, cast_12, full_13)
        del full_13, masked_fill_1

        # pd_op.full: (xf32) <- ()
        full_14 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign: (xf32) <- (xf32)
        assign_2 = full_14

        # pd_op.masked_fill: (1x45640x256xf32) <- (1x45640x256xf32, 1x45640x1xb, xf32)
        masked_fill_2 = paddle._C_ops.masked_fill(data_0, cast_11, full_14)
        del cast_11, data_0

        # pd_op.masked_fill: (1x45640x256xf32) <- (1x45640x256xf32, 1x-1x1xb, xf32)
        masked_fill_3 = paddle._C_ops.masked_fill(masked_fill_2, cast_12, full_14)
        del cast_12

        # pd_op.matmul: (1x45640x256xf32) <- (1x45640x256xf32, 256x256xf32)
        matmul_0 = paddle._C_ops.matmul(masked_fill_3, parameter_3, False, False)
        del parameter_3

        # pd_op.add: (1x45640x256xf32) <- (1x45640x256xf32, 256xf32)
        add_4 = paddle._C_ops.add(matmul_0, parameter_2)
        del parameter_2

        # pd_op.layer_norm: (1x45640x256xf32, 1x45640xf32, 1x45640xf32) <- (1x45640x256xf32, 256xf32, 256xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(
            paddle._C_ops.layer_norm(
                add_4, parameter_1, parameter_0, float("1e-05"), 2
            ),
            lambda out: out if isinstance(out, (list, tuple)) else (out, None, None),
        )
        del (
            add_4,
            assign_0,
            assign_1,
            assign_2,
            full_14,
            masked_fill_2,
            masked_fill_3,
            matmul_0,
            parameter_0,
            parameter_1,
        )

        return layer_norm_0, masked_fill_0
