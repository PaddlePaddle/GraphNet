import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.cast: (6x2xi32) <- (6x2xf32)
        cast_0 = paddle._C_ops.cast(data_3, paddle.int32)
        del data_3

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (6xi32) <- (6x2xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            cast_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_2 = []

        # pd_op.max: (xi32) <- (6xi32, 0xi64)
        max_0 = paddle._C_ops.max(slice_0, full_int_array_2, False)
        del slice_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.slice: (6xi32) <- (6x2xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            cast_0, [1], full_int_array_1, full_int_array_3, [1], [1]
        )

        # pd_op.max: (xi32) <- (6xi32, 0xi64)
        max_1 = paddle._C_ops.max(slice_1, full_int_array_2, False)
        del full_int_array_2, slice_1

        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("6"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.cast: (xi64) <- (xi32)
        cast_1 = paddle._C_ops.cast(max_0, paddle.int64)
        del max_0

        # pd_op.cast: (xi64) <- (xi32)
        cast_2 = paddle._C_ops.cast(max_1, paddle.int64)
        del max_1

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [full_0, cast_1, cast_2]
        del cast_1, cast_2, full_0

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (6x-1x-1xi32) <- (1xf32, 3xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_1, stack_0, paddle.int32
        )
        del full_1, stack_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (6x-1x-1xi32) <- (6x-1x-1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(full_with_tensor_0, full_2, float("-1"), True)
        del full_with_tensor_0

        # pd_op.slice: (xi32) <- (1xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del data_1

        # pd_op.scale: (xi32) <- (xi32, 1xf32)
        scale_1 = paddle._C_ops.scale(slice_2, full_2, float("0"), True)
        del slice_2

        # pd_op.cast: (xi64) <- (xi32)
        cast_3 = paddle._C_ops.cast(scale_1, paddle.int64)
        del scale_1

        # builtin.combine: ([xi64]) <- (xi64)
        combine_1 = [cast_3]

        # pd_op.stack: (1xi64) <- ([xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.slice: (-1x6xf32) <- (6x6xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_0, [0], full_int_array_0, stack_1, [-1], [])
        del data_0

        # pd_op.slice: (-1x28x28xf32) <- (6x28x28xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(data_2, [0], full_int_array_0, stack_1, [-1], [])
        del data_2, stack_1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 1]

        # pd_op.slice: (xi32) <- (6x2xi32, 2xi64, 2xi64)
        slice_5 = paddle._C_ops.slice(
            cast_0, [0, 1], full_int_array_4, full_int_array_5, [1, 1], [0, 1]
        )
        del full_int_array_4, full_int_array_5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [0, 1]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [1, 2]

        # pd_op.slice: (xi32) <- (6x2xi32, 2xi64, 2xi64)
        slice_6 = paddle._C_ops.slice(
            cast_0, [0, 1], full_int_array_6, full_int_array_7, [1, 1], [0, 1]
        )
        del cast_0, full_int_array_6, full_int_array_7

        # pd_op.unsqueeze: (-1x1x28x28xf32) <- (-1x28x28xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(slice_4, full_int_array_1)
        del slice_4

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2147483647]

        # pd_op.slice: (-1x4xf32) <- (-1x6xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            slice_3, [1], full_int_array_3, full_int_array_8, [1], []
        )
        del full_int_array_8, slice_3

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("1"), paddle.int32, paddle.core.CPUPlace()
        )

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(slice_7, 4, full_3)
        del full_3, slice_7

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        (
            split_0,
            split_1,
            split_2,
            split_3,
        ) = split_with_num_0
        del split_with_num_0

        # pd_op.shape64: (4xi64) <- (-1x1x28x28xf32)
        shape64_0 = paddle._C_ops.shape64(unsqueeze_0)

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_0, full_int_array_1, [1], [0]
        )
        del shape64_0

        # pd_op.full: (1xi64) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.cast: (xi64) <- (xi32)
        cast_4 = paddle._C_ops.cast(slice_5, paddle.int64)
        del slice_5

        # pd_op.full: (1xi64) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_0 = paddle.arange(full_4, cast_4, full_5, dtype="int64")

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_5 = paddle._C_ops.cast(arange_0, paddle.float32)
        del arange_0

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_5, full_2, float("0.5"), True)
        del cast_5

        # pd_op.cast: (xi64) <- (xi32)
        cast_6 = paddle._C_ops.cast(slice_6, paddle.int64)
        del slice_6

        # pd_op.arange: (-1xi64) <- (1xi64, xi64, 1xi64)
        arange_1 = paddle.arange(full_4, cast_6, full_5, dtype="int64")
        del full_4, full_5

        # pd_op.cast: (-1xf32) <- (-1xi64)
        cast_7 = paddle._C_ops.cast(arange_1, paddle.float32)
        del arange_1

        # pd_op.scale: (-1xf32) <- (-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(cast_7, full_2, float("0.5"), True)
        del cast_7

        # pd_op.subtract: (-1x-1xf32) <- (-1xf32, -1x1xf32)
        subtract_0 = paddle._C_ops.subtract(scale_2, split_1)
        del scale_2

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_1 = paddle._C_ops.subtract(split_3, split_1)
        del split_1, split_3

        # pd_op.divide: (-1x-1xf32) <- (-1x-1xf32, -1x1xf32)
        divide_0 = paddle._C_ops.divide(subtract_0, subtract_1)
        del subtract_0, subtract_1

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("2"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(divide_0, full_6, float("0"), True)
        del divide_0

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(scale_4, full_2, float("-1"), True)
        del scale_4

        # pd_op.subtract: (-1x-1xf32) <- (-1xf32, -1x1xf32)
        subtract_2 = paddle._C_ops.subtract(scale_3, split_0)
        del scale_3

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_3 = paddle._C_ops.subtract(split_2, split_0)
        del split_0, split_2

        # pd_op.divide: (-1x-1xf32) <- (-1x-1xf32, -1x1xf32)
        divide_1 = paddle._C_ops.divide(subtract_2, subtract_3)
        del subtract_2, subtract_3

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(divide_1, full_6, float("0"), True)
        del divide_1, full_6

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_6, full_2, float("-1"), True)
        del full_2, scale_6

        # pd_op.unsqueeze: (-1x1x-1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(scale_7, full_int_array_1)

        # pd_op.shape64: (2xi64) <- (-1x-1xf32)
        shape64_1 = paddle._C_ops.shape64(scale_5)

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            shape64_1, [0], full_int_array_1, full_int_array_3, [1], [0]
        )
        del shape64_1

        # pd_op.shape64: (2xi64) <- (-1x-1xf32)
        shape64_2 = paddle._C_ops.shape64(scale_7)
        del scale_7

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_0, full_int_array_1, [1], [0]
        )

        # pd_op.slice: (xi64) <- (2xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            shape64_2, [0], full_int_array_1, full_int_array_3, [1], [0]
        )
        del shape64_2

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_2 = [slice_8, slice_10, slice_12]
        del slice_10, slice_12, slice_8

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.expand: (-1x-1x-1xf32) <- (-1x1x-1xf32, 3xi64)
        expand_0 = paddle._C_ops.expand(unsqueeze_1, stack_2)
        del unsqueeze_1

        # pd_op.unsqueeze: (-1x-1x1xf32) <- (-1x-1xf32, 1xi64)
        unsqueeze_2 = paddle._C_ops.unsqueeze(scale_5, full_int_array_3)
        del full_int_array_3, scale_5

        # pd_op.expand: (-1x-1x-1xf32) <- (-1x-1x1xf32, 3xi64)
        expand_1 = paddle._C_ops.expand(unsqueeze_2, stack_2)
        del stack_2, unsqueeze_2

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32)
        combine_3 = [expand_0, expand_1]
        del expand_0, expand_1

        # pd_op.stack: (-1x-1x-1x2xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32])
        stack_3 = paddle._C_ops.stack(combine_3, 3)
        del combine_3

        # pd_op.grid_sample: (-1x1x-1x-1xf32) <- (-1x1x28x28xf32, -1x-1x-1x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(
            unsqueeze_0, stack_3, "bilinear", "zeros", False
        )
        del stack_3, unsqueeze_0

        # pd_op.slice: (-1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            grid_sample_0, [1], full_int_array_0, full_int_array_1, [1], [1]
        )
        del full_int_array_0, full_int_array_1, grid_sample_0

        # pd_op.full: (xf32) <- ()
        full_7 = paddle._C_ops.full(
            [], float("0.5"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_equal: (-1x-1x-1xb) <- (-1x-1x-1xf32, xf32)
        greater_equal_0 = paddle._C_ops.greater_equal(slice_13, full_7)
        del full_7, slice_13

        # pd_op.cast: (-1x-1x-1xi32) <- (-1x-1x-1xb)
        cast_8 = paddle._C_ops.cast(greater_equal_0, paddle.int32)
        del greater_equal_0

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [0, 0, 0]

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_4 = [cast_3, cast_4, cast_6]
        del cast_3, cast_4, cast_6

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_4 = paddle._C_ops.stack(combine_4, 0)
        del combine_4

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_10 = [1, 1, 1]

        # pd_op.set_value_with_tensor_: (6x-1x-1xi32) <- (6x-1x-1xi32, -1x-1x-1xi32, 3xi64, 3xi64, 3xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(
            scale_0,
            cast_8,
            full_int_array_9,
            stack_4,
            full_int_array_10,
            [0, 1, 2],
            [],
            [],
        )
        del cast_8, full_int_array_10, full_int_array_9, scale_0, stack_4

        return set_value_with_tensor__0
