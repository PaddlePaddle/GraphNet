import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0):
        # pd_op.full: (xi64) <- ()
        full_0 = paddle._C_ops.full(
            [], float("1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_1 = paddle._C_ops.full(
            [], float("49"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64)
        combine_0 = [full_0, data_0, full_1, full_0]
        del data_0, full_1

        # pd_op.stack: (4xi64) <- ([xi64, xi64, xi64, xi64])
        stack_0 = paddle._C_ops.stack(combine_0, 0)
        del combine_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_with_tensor: (1x-1x49x1xf32) <- (1xf32, 4xi64)
        full_with_tensor_0 = paddle._C_ops.full_with_tensor(
            full_2, stack_0, paddle.float32
        )
        del full_2, stack_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [0, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-7, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [1, 1]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__0 = paddle._C_ops.set_value_(
            full_with_tensor_0,
            full_int_array_0,
            full_int_array_1,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("0")],
        )
        del full_int_array_0, full_with_tensor_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [0, -7]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [-7, -3]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__1 = paddle._C_ops.set_value_(
            set_value__0,
            full_int_array_3,
            full_int_array_4,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("1")],
        )
        del full_int_array_3, set_value__0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [0, -3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [-7, 2147483647]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__2 = paddle._C_ops.set_value_(
            set_value__1,
            full_int_array_5,
            full_int_array_6,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("2")],
        )
        del full_int_array_5, full_int_array_6, set_value__1

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [-7, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [-3, -7]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__3 = paddle._C_ops.set_value_(
            set_value__2,
            full_int_array_7,
            full_int_array_8,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("3")],
        )
        del full_int_array_7, set_value__2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [-3, -3]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__4 = paddle._C_ops.set_value_(
            set_value__3,
            full_int_array_1,
            full_int_array_9,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("4")],
        )
        del full_int_array_1, set_value__3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [-3, 2147483647]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__5 = paddle._C_ops.set_value_(
            set_value__4,
            full_int_array_4,
            full_int_array_10,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("5")],
        )
        del full_int_array_10, full_int_array_4, set_value__4

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [-3, 0]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [2147483647, -7]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__6 = paddle._C_ops.set_value_(
            set_value__5,
            full_int_array_11,
            full_int_array_12,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("6")],
        )
        del full_int_array_11, full_int_array_12, set_value__5

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [2147483647, -3]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__7 = paddle._C_ops.set_value_(
            set_value__6,
            full_int_array_8,
            full_int_array_13,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("7")],
        )
        del full_int_array_13, full_int_array_8, set_value__6

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [2147483647, 2147483647]

        # pd_op.set_value_: (1x-1x49x1xf32) <- (1x-1x49x1xf32, 2xi64, 2xi64, 2xi64)
        set_value__8 = paddle._C_ops.set_value_(
            set_value__7,
            full_int_array_9,
            full_int_array_14,
            full_int_array_2,
            [1, 2],
            [],
            [],
            [1],
            [float("8")],
        )
        del full_int_array_14, full_int_array_2, full_int_array_9, set_value__7

        # pd_op.shape64: (4xi64) <- (1x-1x49x1xf32)
        shape64_0 = paddle._C_ops.shape64(set_value__8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [2]

        # pd_op.slice: (xi64) <- (4xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            shape64_0, [0], full_int_array_15, full_int_array_16, [1], [0]
        )
        del shape64_0

        # pd_op.full: (xi64) <- ()
        full_3 = paddle._C_ops.full(
            [], float("7"), paddle.int64, paddle.framework._current_expected_place()
        )

        # pd_op.floor_divide: (xi64) <- (xi64, xi64)
        floor_divide_0 = paddle._C_ops.floor_divide(slice_0, full_3)
        del full_3, slice_0

        # pd_op.full: (xi64) <- ()
        full_4 = paddle._C_ops.full(
            [], float("-1"), paddle.int64, paddle.core.CPUPlace()
        )

        # pd_op.full: (xi64) <- ()
        full_5 = paddle._C_ops.full(
            [], float("7"), paddle.int64, paddle.core.CPUPlace()
        )

        # builtin.combine: ([xi64, xi64, xi64, xi64, xi64, xi64]) <- (xi64, xi64, xi64, xi64, xi64, xi64)
        combine_1 = [full_4, floor_divide_0, full_5, full_5, full_5, full_0]
        del floor_divide_0, full_0, full_4, full_5

        # pd_op.stack: (6xi64) <- ([xi64, xi64, xi64, xi64, xi64, xi64])
        stack_1 = paddle._C_ops.stack(combine_1, 0)
        del combine_1

        # pd_op.reshape: (-1x-1x7x7x7x1xf32) <- (1x-1x49x1xf32, 6xi64)
        reshape_0 = paddle._C_ops.reshape(set_value__8, stack_1)
        del stack_1

        # pd_op.transpose: (-1x-1x7x7x7x1xf32) <- (-1x-1x7x7x7x1xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_0, [0, 1, 3, 2, 4, 5])
        del reshape_0

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [-1, 7, 7, 1]

        # pd_op.reshape: (-1x7x7x1xf32) <- (-1x-1x7x7x7x1xf32, 4xi64)
        reshape_1 = paddle._C_ops.reshape(transpose_0, full_int_array_17)
        del full_int_array_17, transpose_0

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [-1, 49]

        # pd_op.reshape: (-1x49xf32) <- (-1x7x7x1xf32, 2xi64)
        reshape_2 = paddle._C_ops.reshape(reshape_1, full_int_array_18)
        del full_int_array_18, reshape_1

        # pd_op.unsqueeze: (-1x1x49xf32) <- (-1x49xf32, 1xi64)
        unsqueeze_0 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_15)
        del full_int_array_15

        # pd_op.unsqueeze: (-1x49x1xf32) <- (-1x49xf32, 1xi64)
        unsqueeze_1 = paddle._C_ops.unsqueeze(reshape_2, full_int_array_16)
        del full_int_array_16, reshape_2

        # pd_op.subtract: (-1x49x49xf32) <- (-1x1x49xf32, -1x49x1xf32)
        subtract_0 = paddle._C_ops.subtract(unsqueeze_0, unsqueeze_1)
        del unsqueeze_0, unsqueeze_1

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full_like: (-1x49x49xf32) <- (-1x49x49xf32, 1xf32)
        full_like_0 = paddle._C_ops.full_like(
            subtract_0,
            full_6,
            paddle.float32,
            paddle.framework._current_expected_place(),
        )
        del full_6

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("-100"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (-1x49x49xf32) <- (-1x49x49xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(full_like_0, full_7, float("0"), True)
        del full_7, full_like_0

        # pd_op.full: (xf32) <- ()
        full_8 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.not_equal: (-1x49x49xb) <- (-1x49x49xf32, xf32)
        not_equal_0 = paddle._C_ops.not_equal(subtract_0, full_8)
        del full_8, subtract_0

        # pd_op.cast: (-1x49x49xf32) <- (-1x49x49xb)
        cast_0 = paddle._C_ops.cast(not_equal_0, paddle.float32)
        del not_equal_0

        # pd_op.multiply: (-1x49x49xf32) <- (-1x49x49xf32, -1x49x49xf32)
        multiply_0 = paddle._C_ops.multiply(scale_0, cast_0)
        del cast_0, scale_0, set_value__8

        return multiply_0
