import paddle


class GraphModule(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, data_0, data_1, data_2, data_3, data_4):
        # pd_op.cast: (571x1xf32) <- (571x1xf64)
        cast_0 = paddle._C_ops.cast(data_0, paddle.float32)
        del data_0

        # pd_op.full: (xf32) <- ()
        full_0 = paddle._C_ops.full(
            [], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.greater_than: (571x1xb) <- (571x1xf32, xf32)
        greater_than_0 = paddle._C_ops.greater_than(cast_0, full_0)

        # pd_op.cast: (571x1xf32) <- (571x1xb)
        cast_1 = paddle._C_ops.cast(greater_than_0, paddle.float32)
        del greater_than_0

        # pd_op.multiply: (571x1xf32) <- (571x1xf32, 571x1xf32)
        multiply_0 = paddle._C_ops.multiply(cast_0, cast_1)
        del cast_1

        # pd_op.subtract: (571x1xf32) <- (571x1xf32, 571x1xf32)
        subtract_0 = paddle._C_ops.subtract(data_1, cast_0)

        # pd_op.abs: (571x1xf32) <- (571x1xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)
        del subtract_0

        # pd_op.pow: (571x1xf32) <- (571x1xf32)
        pow_0 = paddle._C_ops.pow(abs_0, float("2"))
        del abs_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full(
            [1], float("0.75"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (571x1xf32) <- (571x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(pow_0, full_1, float("0"), True)
        del full_1, pow_0

        # pd_op.less_equal: (571x1xb) <- (571x1xf32, xf32)
        less_equal_0 = paddle._C_ops.less_equal(cast_0, full_0)
        del full_0

        # pd_op.cast: (571x1xf32) <- (571x1xb)
        cast_2 = paddle._C_ops.cast(less_equal_0, paddle.float32)
        del less_equal_0

        # pd_op.multiply: (571x1xf32) <- (571x1xf32, 571x1xf32)
        multiply_1 = paddle._C_ops.multiply(scale_0, cast_2)
        del cast_2, scale_0

        # pd_op.add: (571x1xf32) <- (571x1xf32, 571x1xf32)
        add_1 = paddle._C_ops.add(multiply_0, multiply_1)
        del multiply_0, multiply_1

        # pd_op.bce_loss: (571x1xf32) <- (571x1xf32, 571x1xf32)
        bce_loss_0 = paddle._C_ops.bce_loss(data_1, cast_0)
        del cast_0, data_1

        # pd_op.multiply: (571x1xf32) <- (571x1xf32, 571x1xf32)
        multiply_2 = paddle._C_ops.multiply(bce_loss_0, add_1)
        del add_1, bce_loss_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [1]

        # pd_op.sum: (571xf32) <- (571x1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_2, full_int_array_0, None, False)
        del multiply_2

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [571, 1]

        # pd_op.reshape: (571x1xf32) <- (571xf32, 2xi64)
        reshape_0 = paddle._C_ops.reshape(sum_0, full_int_array_1)
        del sum_0

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [3]

        # pd_op.slice: (571xf32) <- (571x4xf32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(
            data_2, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.slice: (571xf32) <- (571x4xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(
            data_2, [1], full_int_array_4, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (571xf32) <- (571xf32, 571xf32)
        subtract_1 = paddle._C_ops.subtract(slice_0, slice_1)
        del slice_0, slice_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [4]

        # pd_op.slice: (571xf32) <- (571x4xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(
            data_2, [1], full_int_array_3, full_int_array_5, [1], [1]
        )

        # pd_op.slice: (571xf32) <- (571x4xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(
            data_2, [1], full_int_array_0, full_int_array_2, [1], [1]
        )

        # pd_op.subtract: (571xf32) <- (571xf32, 571xf32)
        subtract_2 = paddle._C_ops.subtract(slice_2, slice_3)
        del slice_2, slice_3

        # pd_op.multiply: (571xf32) <- (571xf32, 571xf32)
        multiply_3 = paddle._C_ops.multiply(subtract_1, subtract_2)
        del subtract_1, subtract_2

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(
            data_3, [1], full_int_array_2, full_int_array_3, [1], [1]
        )

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(
            data_3, [1], full_int_array_4, full_int_array_0, [1], [1]
        )

        # pd_op.subtract: (1xf32) <- (1xf32, 1xf32)
        subtract_3 = paddle._C_ops.subtract(slice_4, slice_5)
        del slice_4, slice_5

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(
            data_3, [1], full_int_array_3, full_int_array_5, [1], [1]
        )
        del full_int_array_3, full_int_array_5

        # pd_op.slice: (1xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(
            data_3, [1], full_int_array_0, full_int_array_2, [1], [1]
        )

        # pd_op.subtract: (1xf32) <- (1xf32, 1xf32)
        subtract_4 = paddle._C_ops.subtract(slice_6, slice_7)
        del slice_6, slice_7

        # pd_op.multiply: (1xf32) <- (1xf32, 1xf32)
        multiply_4 = paddle._C_ops.multiply(subtract_3, subtract_4)
        del subtract_3, subtract_4

        # pd_op.slice: (571x2xf32) <- (571x4xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(
            data_2, [1], full_int_array_4, full_int_array_2, [1], []
        )

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_6 = [571, 1, 2]

        # pd_op.reshape: (571x1x2xf32) <- (571x2xf32, 3xi64)
        reshape_1 = paddle._C_ops.reshape(slice_8, full_int_array_6)
        del slice_8

        # pd_op.slice: (1x2xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(
            data_3, [1], full_int_array_4, full_int_array_2, [1], []
        )

        # pd_op.maximum: (571x1x2xf32) <- (571x1x2xf32, 1x2xf32)
        maximum_0 = paddle._C_ops.maximum(reshape_1, slice_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2147483647]

        # pd_op.slice: (571x2xf32) <- (571x4xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(
            data_2, [1], full_int_array_2, full_int_array_7, [1], []
        )
        del data_2

        # pd_op.reshape: (571x1x2xf32) <- (571x2xf32, 3xi64)
        reshape_2 = paddle._C_ops.reshape(slice_10, full_int_array_6)
        del full_int_array_6, slice_10

        # pd_op.slice: (1x2xf32) <- (1x4xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(
            data_3, [1], full_int_array_2, full_int_array_7, [1], []
        )
        del data_3, full_int_array_7

        # pd_op.minimum: (571x1x2xf32) <- (571x1x2xf32, 1x2xf32)
        minimum_0 = paddle._C_ops.minimum(reshape_2, slice_11)

        # pd_op.subtract: (571x1x2xf32) <- (571x1x2xf32, 571x1x2xf32)
        subtract_5 = paddle._C_ops.subtract(minimum_0, maximum_0)
        del maximum_0, minimum_0

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full(
            [1], float("3.40282e+38"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.clip: (571x1x2xf32) <- (571x1x2xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_5, full_2, full_3)
        del subtract_5

        # pd_op.slice: (571x1xf32) <- (571x1x2xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(
            clip_0, [2], full_int_array_4, full_int_array_0, [1], [2]
        )

        # pd_op.slice: (571x1xf32) <- (571x1x2xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(
            clip_0, [2], full_int_array_0, full_int_array_2, [1], [2]
        )
        del clip_0

        # pd_op.multiply: (571x1xf32) <- (571x1xf32, 571x1xf32)
        multiply_5 = paddle._C_ops.multiply(slice_12, slice_13)
        del slice_12, slice_13

        # pd_op.reshape: (571x1xf32) <- (571xf32, 2xi64)
        reshape_3 = paddle._C_ops.reshape(multiply_3, full_int_array_1)
        del full_int_array_1, multiply_3

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [1, 1]

        # pd_op.reshape: (1x1xf32) <- (1xf32, 2xi64)
        reshape_4 = paddle._C_ops.reshape(multiply_4, full_int_array_8)
        del full_int_array_8, multiply_4

        # pd_op.add: (571x1xf32) <- (571x1xf32, 1x1xf32)
        add_2 = paddle._C_ops.add(reshape_3, reshape_4)
        del reshape_3, reshape_4

        # pd_op.subtract: (571x1xf32) <- (571x1xf32, 571x1xf32)
        subtract_6 = paddle._C_ops.subtract(add_2, multiply_5)
        del add_2

        # pd_op.minimum: (571x1x2xf32) <- (571x1x2xf32, 1x2xf32)
        minimum_1 = paddle._C_ops.minimum(reshape_1, slice_9)
        del reshape_1, slice_9

        # pd_op.maximum: (571x1x2xf32) <- (571x1x2xf32, 1x2xf32)
        maximum_1 = paddle._C_ops.maximum(reshape_2, slice_11)
        del reshape_2, slice_11

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full(
            [1], float("0"), paddle.float32, paddle.framework._current_expected_place()
        )

        # pd_op.assign_value_: (1xf32) <- (1xf32)
        assign_value__0 = paddle._C_ops.assign_value_(
            full_4,
            [1],
            paddle.float32,
            [float("1e-06")],
            paddle.framework._current_expected_place(),
        )
        del full_4

        # pd_op.maximum: (571x1xf32) <- (571x1xf32, 1xf32)
        maximum_2 = paddle._C_ops.maximum(subtract_6, assign_value__0)
        del subtract_6

        # pd_op.divide: (571x1xf32) <- (571x1xf32, 571x1xf32)
        divide_0 = paddle._C_ops.divide(multiply_5, maximum_2)
        del multiply_5

        # pd_op.subtract: (571x1x2xf32) <- (571x1x2xf32, 571x1x2xf32)
        subtract_7 = paddle._C_ops.subtract(maximum_1, minimum_1)
        del maximum_1, minimum_1

        # pd_op.clip: (571x1x2xf32) <- (571x1x2xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_7, full_2, full_3)
        del full_2, full_3, subtract_7

        # pd_op.slice: (571x1xf32) <- (571x1x2xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(
            clip_1, [2], full_int_array_4, full_int_array_0, [1], [2]
        )
        del full_int_array_4

        # pd_op.slice: (571x1xf32) <- (571x1x2xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(
            clip_1, [2], full_int_array_0, full_int_array_2, [1], [2]
        )
        del clip_1, full_int_array_0, full_int_array_2

        # pd_op.multiply: (571x1xf32) <- (571x1xf32, 571x1xf32)
        multiply_6 = paddle._C_ops.multiply(slice_14, slice_15)
        del slice_14, slice_15

        # pd_op.maximum: (571x1xf32) <- (571x1xf32, 1xf32)
        maximum_3 = paddle._C_ops.maximum(multiply_6, assign_value__0)
        del assign_value__0, multiply_6

        # pd_op.subtract: (571x1xf32) <- (571x1xf32, 571x1xf32)
        subtract_8 = paddle._C_ops.subtract(maximum_3, maximum_2)
        del maximum_2

        # pd_op.divide: (571x1xf32) <- (571x1xf32, 571x1xf32)
        divide_1 = paddle._C_ops.divide(subtract_8, maximum_3)
        del maximum_3, subtract_8

        # pd_op.subtract: (571x1xf32) <- (571x1xf32, 571x1xf32)
        subtract_9 = paddle._C_ops.subtract(divide_0, divide_1)
        del divide_0, divide_1

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full(
            [1], float("-1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (571x1xf32) <- (571x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(subtract_9, full_5, float("1"), True)
        del full_5, subtract_9

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full(
            [1], float("1"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (571x1xf32) <- (571x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(reshape_0, full_6, float("0"), True)
        del full_6, reshape_0

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full(
            [1], float("6"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (571x1xf32) <- (571x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_1, full_7, float("0"), True)
        del full_7, scale_1

        # pd_op.add: (571x1xf32) <- (571x1xf32, 571x1xf32)
        add_3 = paddle._C_ops.add(scale_2, scale_3)
        del scale_2, scale_3

        # pd_op.logical_not: (571x1xb) <- (571x1xb)
        logical_not_0 = paddle._C_ops.logical_not(data_4)
        del data_4

        # pd_op.cast: (571x1xf32) <- (571x1xb)
        cast_3 = paddle._C_ops.cast(logical_not_0, paddle.float32)
        del logical_not_0

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full(
            [1], float("1e+08"), paddle.float32, paddle.core.CPUPlace()
        )

        # pd_op.scale: (571x1xf32) <- (571x1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_3, full_8, float("0"), True)
        del cast_3, full_8

        # pd_op.add: (571x1xf32) <- (571x1xf32, 571x1xf32)
        add_0 = paddle._C_ops.add(add_3, scale_4)
        del add_3, scale_4

        return add_0
