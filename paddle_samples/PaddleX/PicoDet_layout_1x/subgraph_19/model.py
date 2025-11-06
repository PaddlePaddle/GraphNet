import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_0 = paddle._C_ops.greater_than(data_0, full_0)

        # pd_op.cast: (xi64) <- (xb)
        cast_0 = paddle._C_ops.cast(greater_than_0, paddle.int64)
        del greater_than_0

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_0 = paddle._C_ops.not_equal(cast_0, full_0)
        del cast_0

        # pd_op.cast: (xi64) <- (xb)
        cast_1 = paddle._C_ops.cast(not_equal_0, paddle.int64)
        del not_equal_0

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_0 = paddle._C_ops.equal(cast_1, full_0)
        del cast_1

        # pd_op.greater_than: (xb) <- (xi64, xi64)
        greater_than_1 = paddle._C_ops.greater_than(data_1, full_0)

        # pd_op.cast: (xi64) <- (xb)
        cast_2 = paddle._C_ops.cast(greater_than_1, paddle.int64)
        del greater_than_1

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_1 = paddle._C_ops.not_equal(cast_2, full_0)
        del cast_2

        # pd_op.cast: (xi64) <- (xb)
        cast_3 = paddle._C_ops.cast(not_equal_1, paddle.int64)
        del not_equal_1

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_1 = paddle._C_ops.equal(cast_3, full_0)
        del cast_3

        # pd_op.multiply: (xi64) <- (xi64, xi64)
        multiply_0 = paddle._C_ops.multiply(data_0, data_1)

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_2 = paddle._C_ops.equal(multiply_0, full_0)
        del multiply_0

        # pd_op.cast: (xi64) <- (xb)
        cast_4 = paddle._C_ops.cast(equal_2, paddle.int64)
        del equal_2

        # pd_op.not_equal: (xb) <- (xi64, xi64)
        not_equal_2 = paddle._C_ops.not_equal(cast_4, full_0)
        del cast_4

        # pd_op.cast: (xi64) <- (xb)
        cast_5 = paddle._C_ops.cast(not_equal_2, paddle.int64)
        del not_equal_2

        # pd_op.equal: (xb) <- (xi64, xi64)
        equal_3 = paddle._C_ops.equal(cast_5, full_0)
        del cast_5, full_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [3]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_2, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_0 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0, slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [4]

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_2, [1], full_int_array_1, full_int_array_4, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_2, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_1 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2, slice_3

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_1 = paddle._C_ops.multiply(subtract_0, subtract_1)
        del subtract_0, subtract_1

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_3, [1], full_int_array_0, full_int_array_1, [1], [1]
        )

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_3, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_2 = paddle._C_ops.subtract(slice_4, slice_5)
        del slice_4, slice_5

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_3, [1], full_int_array_1, full_int_array_4, [1], [1]
        )
        del full_int_array_1, full_int_array_4

        # pd_op.slice: (-1xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_3, [1], full_int_array_3, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (-1xf32) <- (-1xf32, -1xf32)
        subtract_3 = paddle._C_ops.subtract(slice_6, slice_7)
        del slice_6, slice_7

        # pd_op.multiply: (-1xf32) <- (-1xf32, -1xf32)
        multiply_2 = paddle._C_ops.multiply(subtract_2, subtract_3)
        del subtract_2, subtract_3

        # pd_op.slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_2, [1], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_2 = paddle._C_ops.full(
            [], float("2"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64]) <- (xi64, xi64, xi64)
        combine_0 = [data_0, full_1, full_2]
        del full_2

        # pd_op.stack: (3xi64) <- ([xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.reshape: (-1x1x2xf32) <- (-1x2xf32, 3xi64)
        reshape_0 = paddle._C_ops.reshape(slice_8, stack_0)
        del slice_8

        # pd_op.slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_3, [1], full_int_array_2, full_int_array_0, [1], []
        )

        # pd_op.maximum: (-1x-1x2xf32) <- (-1x1x2xf32, -1x2xf32)
        maximum_0 = paddle._C_ops.maximum(reshape_0, slice_9)
        del reshape_0, slice_9

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_5, [1], []
        )
        del data_2

        # pd_op.reshape: (-1x1x2xf32) <- (-1x2xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(slice_10, stack_0)
        del slice_10, stack_0

        # pd_op.slice: (-1x2xf32) <- (-1x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_3, [1], full_int_array_0, full_int_array_5, [1], []
        )
        del data_3, full_int_array_5

        # pd_op.minimum: (-1x-1x2xf32) <- (-1x1x2xf32, -1x2xf32)
        minimum_0 = paddle._C_ops.minimum(reshape_1, slice_11)
        del reshape_1, slice_11

        # pd_op.subtract: (-1x-1x2xf32) <- (-1x-1x2xf32, -1x-1x2xf32)
        subtract_4 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (-1x-1x2xf32) <- (-1x-1x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_4, full_3, full_4)
        del full_3, full_4, subtract_4

        # pd_op.slice: (-1x-1xf32) <- (-1x-1x2xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            clip_0, [2], full_int_array_2, full_int_array_3, [1], [2]
        )
        del full_int_array_2

        # pd_op.slice: (-1x-1xf32) <- (-1x-1x2xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            clip_0, [2], full_int_array_3, full_int_array_0, [1], [2]
        )
        del clip_0, full_int_array_0, full_int_array_3

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        multiply_3 = paddle._C_ops.multiply(slice_12, slice_13)
        del slice_12, slice_13

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_1 = [data_0, full_1]
        del data_0

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x1xf32) <- (-1xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(multiply_1, stack_1)
        del multiply_1, stack_1

        # builtin.combine: ([xi64, xi64]) <- (xi64, xi64)
        combine_2 = [full_1, data_1]
        del data_1, full_1

        # pd_op.stack: (2xi64) <- ([xi64, xi64])
        stack_2 = paddle._C_ops.stack(combine_2, 0)
        del combine_2

        # pd_op.reshape: (1x-1xf32) <- (-1xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(multiply_2, stack_2)
        del multiply_2, stack_2

        # pd_op.add: (-1x-1xf32) <- (-1x1xf32, 1x-1xf32)
        add_0 = paddle._C_ops.add(reshape_2, reshape_3)
        del reshape_2, reshape_3

        # pd_op.subtract: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        subtract_5 = paddle._C_ops.subtract(add_0, multiply_3)
        del add_0

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (1xf32) <- (1xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_5,
            [1],
            paddle.float32,
            [float("1e-06")],
            paddle.framework._current_expected_place(),
        )
        del full_5

        # pd_op.maximum: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        maximum_1 = paddle._C_ops.maximum(subtract_5, assign_value__0)
        del assign_value__0, subtract_5

        # pd_op.divide: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        divide_0 = paddle._C_ops.divide(multiply_3, maximum_1)
        del maximum_1, multiply_3

        return divide_0
